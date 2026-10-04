#ifndef TENSOR3D_SPARSE_OPS_HPP
#define TENSOR3D_SPARSE_OPS_HPP

#include "dense.hpp"
#include "tensor.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <map>
#include <stdexcept>
#include <vector>

namespace tensor3d {

inline bool should_use_dense_path(std::size_t left_rows,
                                  std::size_t left_cols,
                                  std::size_t right_cols,
                                  std::size_t left_nnz,
                                  std::size_t right_nnz,
                                  double density_threshold = 0.18) {
    const double left_total = static_cast<double>(left_rows) * static_cast<double>(left_cols);
    const double right_total = static_cast<double>(left_cols) * static_cast<double>(right_cols);
    const double left_density = left_total > 0.0 ? static_cast<double>(left_nnz) / left_total : 0.0;
    const double right_density = right_total > 0.0 ? static_cast<double>(right_nnz) / right_total : 0.0;
    const double avg_density = 0.5 * (left_density + right_density);
    return avg_density >= density_threshold;
}

template <typename T>
double estimate_hadamard_slice_work(const SparseSlice<T>& left, const SparseSlice<T>& right) {
    const double rows = static_cast<double>(left.rows);
    const double cols = static_cast<double>(left.cols);
    const double right_cols = static_cast<double>(right.cols);
    const double left_total = rows * cols;
    const double right_total = cols * right_cols;
    const double left_density = left_total > 0.0 ? static_cast<double>(left.nnz()) / left_total : 0.0;
    const double right_density = right_total > 0.0 ? static_cast<double>(right.nnz()) / right_total : 0.0;

    const bool dense_path = should_use_dense_path(
        left.rows,
        left.cols,
        right.cols,
        left.nnz(),
        right.nnz());

    double format_factor = 1.0;
    if (left.preferred_format == StorageFormat::ELL || right.preferred_format == StorageFormat::ELL) {
        format_factor = 1.25;
    }

    if (dense_path) {
        return std::max(1.0, format_factor * rows * cols * right_cols);
    }

    const double sparse_weight =
        static_cast<double>(left.nnz() + right.nnz()) *
        (1.0 + 0.5 * (left_density + right_density));
    return std::max(1.0, format_factor * sparse_weight);
}

template <typename T>
CooSlice<T> dense_to_coo(const DenseMatrix<T>& matrix) {
    CooSlice<T> result(matrix.rows(), matrix.cols());
    for (std::size_t row = 0; row < matrix.rows(); ++row) {
        for (std::size_t col = 0; col < matrix.cols(); ++col) {
            const T value = matrix(row, col);
            if (is_near_zero(value)) {
                continue;
            }
            result.add_element(row, col, value);
        }
    }
    return result;
}

template <typename T>
CooSlice<T> compact_coo(const CooSlice<T>& input) {
    std::map<std::pair<std::size_t, std::size_t>, T> accumulator;
    for (std::size_t index = 0; index < input.nnz(); ++index) {
        accumulator[{input.row_indices[index], input.col_indices[index]}] += input.values[index];
    }

    CooSlice<T> result(input.rows, input.cols);
    for (const auto& [position, value] : accumulator) {
        if (is_near_zero(value)) {
            continue;
        }
        result.add_element(position.first, position.second, value);
    }
    return result;
}

template <typename T>
CooSlice<T> multiply_sparse_slices_serial(const CooSlice<T>& left, const CooSlice<T>& right, std::uint64_t* iterations = nullptr, double* flops = nullptr) {
    if (left.cols != right.rows) {
        throw std::invalid_argument("Slice multiplication dimension mismatch");
    }

    std::vector<std::vector<std::size_t>> right_rows(right.rows);
    for (std::size_t index = 0; index < right.nnz(); ++index) {
        right_rows[right.row_indices[index]].push_back(index);
    }

    std::vector<std::map<std::size_t, T>> accumulator(left.rows);
    std::uint64_t op_iterations = 0;
    for (std::size_t left_index = 0; left_index < left.nnz(); ++left_index) {
        const std::size_t row = left.row_indices[left_index];
        const std::size_t pivot = left.col_indices[left_index];
        const T left_value = left.values[left_index];
        const auto& matches = right_rows[pivot];
        for (std::size_t right_index : matches) {
            ++op_iterations;
            const std::size_t col = right.col_indices[right_index];
            accumulator[row][col] += left_value * right.values[right_index];
        }
    }

    CooSlice<T> result(left.rows, right.cols);
    for (std::size_t row = 0; row < accumulator.size(); ++row) {
        for (const auto& [col, value] : accumulator[row]) {
            if (is_near_zero(value)) {
                continue;
            }
            result.add_element(row, col, value);
        }
    }

    if (iterations != nullptr) {
        *iterations += op_iterations;
    }
    if (flops != nullptr) {
        *flops += static_cast<double>(op_iterations) * 2.0;
    }

    return result;
}

template <typename T>
Tensor3D<T> hadamard_product_serial(const Tensor3D<T>& left, const Tensor3D<T>& right, RunMetrics* metrics = nullptr) {
    if (left.depth() != right.depth() || left.second_dim() != right.first_dim()) {
        throw std::invalid_argument("Tensor dimensions are incompatible for slice-wise multiplication");
    }

    const auto start = std::chrono::steady_clock::now();
    RunMetrics local_metrics;
    Tensor3D<T> result(left.first_dim(), right.second_dim(), left.depth());
    for (std::size_t slice = 0; slice < left.depth(); ++slice) {
        const auto slice_begin = std::chrono::steady_clock::now();
        const double slice_work = estimate_hadamard_slice_work(left.slice(slice), right.slice(slice));
        const CooSlice<T> left_coo = left.slice(slice).to_coo();
        const CooSlice<T> right_coo = right.slice(slice).to_coo();
        const bool use_dense = should_use_dense_path(left.first_dim(), left.second_dim(), right.second_dim(), left_coo.nnz(), right_coo.nnz());

        if (use_dense) {
            const DenseMatrix<T> dense_left = left.slice(slice).to_dense();
            const DenseMatrix<T> dense_right = right.slice(slice).to_dense();
            const DenseMatrix<T> dense_result = multiply(dense_left, dense_right);
            result.slice(slice).coo = compact_coo(dense_to_coo(dense_result));
            ++local_metrics.slices_dense;
            local_metrics.iterations += left.first_dim() * left.second_dim() * right.second_dim();
            local_metrics.flops += 2.0 * static_cast<double>(left.first_dim()) * static_cast<double>(left.second_dim()) * static_cast<double>(right.second_dim());
            local_metrics.work_units_dense += slice_work;
        } else {
            result.slice(slice).coo = compact_coo(multiply_sparse_slices_serial(left_coo, right_coo, &local_metrics.iterations, &local_metrics.flops));
            ++local_metrics.slices_sparse;
            local_metrics.work_units_sparse += slice_work;
        }

        result.slice(slice).preprocess();
        const std::chrono::duration<double> slice_elapsed = std::chrono::steady_clock::now() - slice_begin;
        if (use_dense) {
            local_metrics.dense_seconds += slice_elapsed.count();
        } else {
            local_metrics.sparse_seconds += slice_elapsed.count();
        }
        ++local_metrics.slices_total;
        local_metrics.work_units_total += slice_work;
    }

    local_metrics.total_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    local_metrics.cpu_assigned_slices = local_metrics.slices_total;
    local_metrics.cpu_assigned_work = local_metrics.work_units_total;
    if (metrics != nullptr) {
        *metrics = local_metrics;
    }
    return result;
}

template <typename T>
Tensor3D<T> hadamard_product_openmp(const Tensor3D<T>& left, const Tensor3D<T>& right, RunMetrics* metrics = nullptr) {
    if (left.depth() != right.depth() || left.second_dim() != right.first_dim()) {
        throw std::invalid_argument("Tensor dimensions are incompatible for slice-wise multiplication");
    }

    const auto start = std::chrono::steady_clock::now();
    Tensor3D<T> result(left.first_dim(), right.second_dim(), left.depth());
    RunMetrics local_metrics;

    std::vector<double> slice_work(left.depth(), 1.0);
    std::vector<std::size_t> ordered_indices(left.depth());
    for (std::size_t slice = 0; slice < left.depth(); ++slice) {
        slice_work[slice] = estimate_hadamard_slice_work(left.slice(slice), right.slice(slice));
        ordered_indices[slice] = slice;
    }
    std::sort(ordered_indices.begin(), ordered_indices.end(), [&](std::size_t lhs, std::size_t rhs) {
        return slice_work[lhs] > slice_work[rhs];
    });

    int thread_capacity = 1;
#ifdef _OPENMP
    thread_capacity = std::max(1, omp_get_max_threads());
#endif

    std::vector<RunMetrics> thread_metrics(static_cast<std::size_t>(thread_capacity));
    std::vector<double> thread_work(static_cast<std::size_t>(thread_capacity), 0.0);
    std::vector<double> thread_time(static_cast<std::size_t>(thread_capacity), 0.0);

#ifdef _OPENMP
#pragma omp parallel for schedule(runtime)
#endif
    for (std::ptrdiff_t work_index = 0; work_index < static_cast<std::ptrdiff_t>(ordered_indices.size()); ++work_index) {
        int thread_id = 0;
#ifdef _OPENMP
        thread_id = omp_get_thread_num();
#endif

        const auto slice_begin = std::chrono::steady_clock::now();
        const std::size_t k = ordered_indices[static_cast<std::size_t>(work_index)];
        const double assigned_work = slice_work[k];

        const CooSlice<T> left_coo = left.slice(k).to_coo();
        const CooSlice<T> right_coo = right.slice(k).to_coo();
        const bool use_dense = should_use_dense_path(left.first_dim(), left.second_dim(), right.second_dim(), left_coo.nnz(), right_coo.nnz());

        std::uint64_t iterations_local = 0;
        double flops_local = 0.0;
        double dense_seconds_local = 0.0;
        double sparse_seconds_local = 0.0;
        std::uint64_t dense_count = 0;
        std::uint64_t sparse_count = 0;

        if (use_dense) {
            const DenseMatrix<T> dense_left = left.slice(k).to_dense();
            const DenseMatrix<T> dense_right = right.slice(k).to_dense();
            const DenseMatrix<T> dense_result = multiply(dense_left, dense_right);
            result.slice(k).coo = compact_coo(dense_to_coo(dense_result));
            dense_count = 1;
            iterations_local = left.first_dim() * left.second_dim() * right.second_dim();
            flops_local = 2.0 * static_cast<double>(left.first_dim()) * static_cast<double>(left.second_dim()) * static_cast<double>(right.second_dim());
        } else {
            result.slice(k).coo = compact_coo(multiply_sparse_slices_serial(left_coo, right_coo, &iterations_local, &flops_local));
            sparse_count = 1;
        }

        result.slice(k).preprocess();
        const std::chrono::duration<double> slice_elapsed = std::chrono::steady_clock::now() - slice_begin;
        if (use_dense) {
            dense_seconds_local = slice_elapsed.count();
        } else {
            sparse_seconds_local = slice_elapsed.count();
        }

        RunMetrics& per_thread_metrics = thread_metrics[static_cast<std::size_t>(thread_id)];
        per_thread_metrics.slices_total += 1;
        per_thread_metrics.slices_dense += dense_count;
        per_thread_metrics.slices_sparse += sparse_count;
        per_thread_metrics.iterations += iterations_local;
        per_thread_metrics.flops += flops_local;
        per_thread_metrics.dense_seconds += dense_seconds_local;
        per_thread_metrics.sparse_seconds += sparse_seconds_local;
        per_thread_metrics.work_units_total += assigned_work;
        if (use_dense) {
            per_thread_metrics.work_units_dense += assigned_work;
        } else {
            per_thread_metrics.work_units_sparse += assigned_work;
        }
        thread_work[static_cast<std::size_t>(thread_id)] += assigned_work;
        thread_time[static_cast<std::size_t>(thread_id)] += slice_elapsed.count();
    }

    for (const RunMetrics& thread_metric : thread_metrics) {
        local_metrics = add_metrics(local_metrics, thread_metric);
    }

    for (double workload : thread_work) {
        if (workload <= 0.0) {
            continue;
        }
        local_metrics.max_thread_work = std::max(local_metrics.max_thread_work, workload);
        if (local_metrics.min_thread_work <= 0.0) {
            local_metrics.min_thread_work = workload;
        } else {
            local_metrics.min_thread_work = std::min(local_metrics.min_thread_work, workload);
        }
    }

    for (double elapsed : thread_time) {
        if (elapsed <= 0.0) {
            continue;
        }
        local_metrics.max_thread_time = std::max(local_metrics.max_thread_time, elapsed);
        if (local_metrics.min_thread_time <= 0.0) {
            local_metrics.min_thread_time = elapsed;
        } else {
            local_metrics.min_thread_time = std::min(local_metrics.min_thread_time, elapsed);
        }
    }

    if (local_metrics.min_thread_work > 0.0) {
        local_metrics.thread_work_imbalance =
            (local_metrics.max_thread_work - local_metrics.min_thread_work) / local_metrics.min_thread_work;
    }
    if (local_metrics.min_thread_time > 0.0) {
        local_metrics.thread_time_imbalance =
            (local_metrics.max_thread_time - local_metrics.min_thread_time) / local_metrics.min_thread_time;
    }

    local_metrics.total_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    local_metrics.cpu_assigned_slices = local_metrics.slices_total;
    local_metrics.cpu_assigned_work = local_metrics.work_units_total;
    if (metrics != nullptr) {
        *metrics = local_metrics;
    }
    return result;
}

template <typename T>
std::vector<QRResult<T>> tensor_qr_serial(const Tensor3D<T>& tensor) {
    std::vector<QRResult<T>> results;
    results.reserve(tensor.depth());
    for (std::size_t slice = 0; slice < tensor.depth(); ++slice) {
        results.push_back(qr_gram_schmidt(tensor.slice(slice).to_dense()));
    }
    return results;
}

template <typename T>
std::vector<SVDResult<T>> tensor_svd_serial(const Tensor3D<T>& tensor) {
    std::vector<SVDResult<T>> results;
    results.reserve(tensor.depth());
    for (std::size_t slice = 0; slice < tensor.depth(); ++slice) {
        results.push_back(svd_via_normal_equation(tensor.slice(slice).to_dense()));
    }
    return results;
}

template <typename T>
std::vector<QRResult<T>> tensor_qr_openmp(const Tensor3D<T>& tensor) {
    std::vector<QRResult<T>> results(tensor.depth());
#ifdef _OPENMP
#pragma omp parallel for schedule(runtime)
#endif
    for (std::ptrdiff_t slice = 0; slice < static_cast<std::ptrdiff_t>(tensor.depth()); ++slice) {
        results[static_cast<std::size_t>(slice)] = qr_gram_schmidt(tensor.slice(static_cast<std::size_t>(slice)).to_dense());
    }
    return results;
}

template <typename T>
std::vector<SVDResult<T>> tensor_svd_openmp(const Tensor3D<T>& tensor) {
    std::vector<SVDResult<T>> results(tensor.depth());
#ifdef _OPENMP
#pragma omp parallel for schedule(runtime)
#endif
    for (std::ptrdiff_t slice = 0; slice < static_cast<std::ptrdiff_t>(tensor.depth()); ++slice) {
        results[static_cast<std::size_t>(slice)] = svd_via_normal_equation(tensor.slice(static_cast<std::size_t>(slice)).to_dense());
    }
    return results;
}

} // namespace tensor3d

#endif
