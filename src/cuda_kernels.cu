#include "cuda_bridge.hpp"

#ifdef TENSOR3D_USE_CUDA

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace tensor3d {
namespace {

#define CUDA_CHECK(call) \
    do { \
        const cudaError_t status = (call); \
        if (status != cudaSuccess) { \
            throw std::runtime_error(cudaGetErrorString(status)); \
        } \
    } while (0)

#define CUBLAS_CHECK(call) \
    do { \
        const cublasStatus_t status = (call); \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            throw std::runtime_error("cuBLAS call failed"); \
        } \
    } while (0)

template <typename T>
void fill_dense_buffer_from_slice(const SparseSlice<T>& slice, double* buffer, std::size_t rows, std::size_t cols) {
    std::fill(buffer, buffer + rows * cols, 0.0);
    const CooSlice<T> coo = slice.to_coo();
    for (std::size_t index = 0; index < coo.nnz(); ++index) {
        const std::size_t row = coo.row_indices[index];
        const std::size_t col = coo.col_indices[index];
        buffer[row * cols + col] = static_cast<double>(coo.values[index]);
    }
}

inline std::size_t choose_dense_batch_size(std::size_t rows_left,
                                           std::size_t cols_left,
                                           std::size_t cols_right,
                                           std::size_t total_dense_slices) {
    constexpr std::size_t kTargetBytes = 512ull * 1024ull * 1024ull;
    const std::size_t elements_per_slice =
        rows_left * cols_left + cols_left * cols_right + rows_left * cols_right;
    if (elements_per_slice == 0) {
        return 1;
    }

    const std::size_t bytes_per_slice = elements_per_slice * sizeof(double);
    std::size_t batch_size = kTargetBytes / bytes_per_slice;
    if (batch_size == 0) {
        batch_size = 1;
    }
    return std::max<std::size_t>(1, std::min(batch_size, total_dense_slices));
}

template <typename T>
void load_dense_slice_from_buffer(Tensor3D<T>& tensor, std::size_t slice_index, const std::vector<double>& buffer, std::size_t rows, std::size_t cols) {
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t col = 0; col < cols; ++col) {
            const double value = buffer[row * cols + col];
            if (std::abs(value) <= 1e-12) {
                continue;
            }
            tensor.slice(slice_index).add_element(row, col, static_cast<T>(value));
        }
    }
}

} // namespace

template <typename T>
Tensor3D<T> hadamard_product_cuda(const Tensor3D<T>& left, const Tensor3D<T>& right, RunMetrics* metrics) {
    if (left.depth() != right.depth() || left.second_dim() != right.first_dim()) {
        throw std::invalid_argument("CUDA multiplication dimension mismatch");
    }

    const auto start = std::chrono::steady_clock::now();
    RunMetrics local_metrics;
    Tensor3D<T> result(left.first_dim(), right.second_dim(), left.depth());
    std::vector<std::size_t> dense_slices;
    std::vector<std::size_t> sparse_slices;
    dense_slices.reserve(left.depth());
    sparse_slices.reserve(left.depth());

    for (std::size_t slice = 0; slice < left.depth(); ++slice) {
        const double slice_work = estimate_hadamard_slice_work(left.slice(slice), right.slice(slice));
        const bool use_dense = should_use_dense_path(
            left.first_dim(),
            left.second_dim(),
            right.second_dim(),
            left.slice(slice).nnz(),
            right.slice(slice).nnz());

        if (use_dense) {
            dense_slices.push_back(slice);
            local_metrics.work_units_dense += slice_work;
        } else {
            sparse_slices.push_back(slice);
            local_metrics.work_units_sparse += slice_work;
        }
        local_metrics.work_units_total += slice_work;
    }

    local_metrics.gpu_assigned_slices = static_cast<std::uint64_t>(dense_slices.size());
    local_metrics.cpu_assigned_slices = static_cast<std::uint64_t>(sparse_slices.size());
    local_metrics.gpu_assigned_work = local_metrics.work_units_dense;
    local_metrics.cpu_assigned_work = local_metrics.work_units_sparse;
    const double partition_total_work = local_metrics.gpu_assigned_work + local_metrics.cpu_assigned_work;
    if (partition_total_work > 0.0) {
        local_metrics.gpu_assigned_ratio = local_metrics.gpu_assigned_work / partition_total_work;
        const double min_work = std::min(local_metrics.gpu_assigned_work, local_metrics.cpu_assigned_work);
        const double max_work = std::max(local_metrics.gpu_assigned_work, local_metrics.cpu_assigned_work);
        local_metrics.partition_load_imbalance = min_work > 0.0 ? (max_work - min_work) / min_work : 0.0;
    }

    if (!sparse_slices.empty()) {
#ifdef _OPENMP
#pragma omp parallel for schedule(runtime)
#endif
        for (std::ptrdiff_t index = 0; index < static_cast<std::ptrdiff_t>(sparse_slices.size()); ++index) {
            const auto slice_begin = std::chrono::steady_clock::now();
            const std::size_t slice = sparse_slices[static_cast<std::size_t>(index)];
            const CooSlice<T> left_coo = left.slice(slice).to_coo();
            const CooSlice<T> right_coo = right.slice(slice).to_coo();

            std::uint64_t iterations_local = 0;
            double flops_local = 0.0;
            result.slice(slice).coo = compact_coo(
                multiply_sparse_slices_serial(left_coo, right_coo, &iterations_local, &flops_local));
            result.slice(slice).preprocess();

            const double sparse_time_local =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - slice_begin).count();

#ifdef _OPENMP
#pragma omp critical
#endif
            {
                ++local_metrics.slices_total;
                ++local_metrics.slices_sparse;
                local_metrics.iterations += iterations_local;
                local_metrics.flops += flops_local;
                local_metrics.sparse_seconds += sparse_time_local;
            }
        }
    }

    if (!dense_slices.empty()) {
        const auto dense_begin = std::chrono::steady_clock::now();

        const std::size_t m = left.first_dim();
        const std::size_t k = left.second_dim();
        const std::size_t n = right.second_dim();
        const std::size_t batch_count = dense_slices.size();
        const std::size_t max_batch = choose_dense_batch_size(m, k, n, batch_count);
        std::vector<double> host_left(max_batch * m * k, 0.0);
        std::vector<double> host_right(max_batch * k * n, 0.0);
        std::vector<double> host_result(max_batch * m * n, 0.0);

        double* device_left = nullptr;
        double* device_right = nullptr;
        double* device_result = nullptr;

        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void**>(&device_left),
            host_left.size() * sizeof(double)));
        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void**>(&device_right),
            host_right.size() * sizeof(double)));
        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void**>(&device_result),
            host_result.size() * sizeof(double)));

        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

        cublasHandle_t handle;
        CUBLAS_CHECK(cublasCreate(&handle));
        CUBLAS_CHECK(cublasSetStream(handle, stream));

        const double alpha = 1.0;
        const double beta = 0.0;
        const long long int stride_a = static_cast<long long int>(m * k);
        const long long int stride_b = static_cast<long long int>(k * n);
        const long long int stride_c = static_cast<long long int>(m * n);

        for (std::size_t base = 0; base < batch_count; base += max_batch) {
            const std::size_t current_batch = std::min(max_batch, batch_count - base);

            for (std::size_t b = 0; b < current_batch; ++b) {
                const std::size_t slice = dense_slices[base + b];
                fill_dense_buffer_from_slice(
                    left.slice(slice),
                    host_left.data() + b * m * k,
                    m,
                    k);
                fill_dense_buffer_from_slice(
                    right.slice(slice),
                    host_right.data() + b * k * n,
                    k,
                    n);
            }

            CUDA_CHECK(cudaMemcpyAsync(
                device_left,
                host_left.data(),
                current_batch * m * k * sizeof(double),
                cudaMemcpyHostToDevice,
                stream));
            CUDA_CHECK(cudaMemcpyAsync(
                device_right,
                host_right.data(),
                current_batch * k * n * sizeof(double),
                cudaMemcpyHostToDevice,
                stream));

            // Row-major C = A * B is computed as column-major C^T = B^T * A^T.
            CUBLAS_CHECK(cublasDgemmStridedBatched(
                handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                static_cast<int>(n),
                static_cast<int>(m),
                static_cast<int>(k),
                &alpha,
                device_right,
                static_cast<int>(n),
                stride_b,
                device_left,
                static_cast<int>(k),
                stride_a,
                &beta,
                device_result,
                static_cast<int>(n),
                stride_c,
                static_cast<int>(current_batch)));

            CUDA_CHECK(cudaMemcpyAsync(
                host_result.data(),
                device_result,
                current_batch * m * n * sizeof(double),
                cudaMemcpyDeviceToHost,
                stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));

            for (std::size_t b = 0; b < current_batch; ++b) {
                const std::size_t slice = dense_slices[base + b];
                const double* slice_ptr = host_result.data() + b * m * n;
                std::vector<double> slice_result(slice_ptr, slice_ptr + m * n);
                load_dense_slice_from_buffer(result, slice, slice_result, m, n);
                result.slice(slice).preprocess();
            }
        }

        CUBLAS_CHECK(cublasDestroy(handle));
        CUDA_CHECK(cudaStreamDestroy(stream));
        CUDA_CHECK(cudaFree(device_left));
        CUDA_CHECK(cudaFree(device_right));
        CUDA_CHECK(cudaFree(device_result));

        local_metrics.slices_total += static_cast<std::uint64_t>(batch_count);
        local_metrics.slices_dense += static_cast<std::uint64_t>(batch_count);
        local_metrics.iterations += static_cast<std::uint64_t>(batch_count * m * k * n);
        local_metrics.flops += 2.0 * static_cast<double>(batch_count) * static_cast<double>(m) * static_cast<double>(k) * static_cast<double>(n);
        local_metrics.dense_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - dense_begin).count();
    }

    result.preprocess();
    local_metrics.total_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    if (metrics != nullptr) {
        *metrics = local_metrics;
    }
    return result;
}

template <typename T>
std::vector<QRResult<T>> tensor_qr_cuda(const Tensor3D<T>& tensor) {
    return tensor_qr_serial(tensor);
}

template <typename T>
std::vector<SVDResult<T>> tensor_svd_cuda(const Tensor3D<T>& tensor) {
    return tensor_svd_serial(tensor);
}

} // namespace tensor3d

template tensor3d::Tensor3D<double> tensor3d::hadamard_product_cuda<double>(const tensor3d::Tensor3D<double>&, const tensor3d::Tensor3D<double>&, tensor3d::RunMetrics*);
template std::vector<tensor3d::QRResult<double>> tensor3d::tensor_qr_cuda<double>(const tensor3d::Tensor3D<double>&);
template std::vector<tensor3d::SVDResult<double>> tensor3d::tensor_svd_cuda<double>(const tensor3d::Tensor3D<double>&);

#endif
