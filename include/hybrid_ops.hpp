#ifndef TENSOR3D_HYBRID_OPS_HPP
#define TENSOR3D_HYBRID_OPS_HPP

#include "cuda_bridge.hpp"
#include "sparse_ops.hpp"

#include <algorithm>
#include <cmath>
#include <future>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

namespace tensor3d {

template <typename T>
Tensor3D<T> extract_tensor_indices(const Tensor3D<T>& tensor, const std::vector<std::size_t>& indices) {
    Tensor3D<T> result(tensor.first_dim(), tensor.second_dim(), indices.size());
    for (std::size_t index = 0; index < indices.size(); ++index) {
        result.slice(index) = tensor.slice(indices[index]);
    }
    return result;
}

template <typename T>
void write_tensor_indices(Tensor3D<T>& destination, const Tensor3D<T>& source, const std::vector<std::size_t>& indices) {
    if (source.depth() != indices.size()) {
        throw std::invalid_argument("Source depth does not match index mapping size");
    }

    for (std::size_t index = 0; index < indices.size(); ++index) {
        destination.slice(indices[index]) = source.slice(index);
    }
}

template <typename T>
double estimate_decomposition_slice_work(const SparseSlice<T>& slice) {
    const double rows = static_cast<double>(slice.rows);
    const double cols = static_cast<double>(slice.cols);
    const double total = rows * cols;
    const double density = total > 0.0 ? static_cast<double>(slice.nnz()) / total : 0.0;

    double format_factor = 1.0;
    if (slice.preferred_format == StorageFormat::ELL) {
        format_factor = 1.20;
    }

    if (density >= 0.20) {
        return std::max(1.0, format_factor * rows * cols * std::max(1.0, cols));
    }

    return std::max(
        1.0,
        format_factor * static_cast<double>(slice.nnz()) *
            std::max(1.0, std::log2(static_cast<double>(slice.cols) + 1.0)));
}

inline std::vector<std::size_t> sorted_indices_by_weight_desc(const std::vector<double>& weights) {
    std::vector<std::size_t> order(weights.size());
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::sort(order.begin(), order.end(), [&](std::size_t lhs, std::size_t rhs) {
        return weights[lhs] > weights[rhs];
    });
    return order;
}

inline std::pair<std::vector<std::size_t>, std::vector<std::size_t>>
partition_indices_by_weight(const std::vector<double>& weights, double gpu_ratio) {
    const std::size_t depth = weights.size();
    if (depth == 0) {
        return {std::vector<std::size_t>{}, std::vector<std::size_t>{}};
    }

    const double clamped_ratio = std::clamp(gpu_ratio, 0.0, 1.0);
    const double total_weight = std::accumulate(weights.begin(), weights.end(), 0.0);
    const double target_gpu = total_weight * clamped_ratio;

    std::vector<std::size_t> gpu_indices;
    std::vector<std::size_t> cpu_indices;
    gpu_indices.reserve(depth);
    cpu_indices.reserve(depth);

    double gpu_weight = 0.0;
    double cpu_weight = 0.0;
    for (const std::size_t index : sorted_indices_by_weight_desc(weights)) {
        const double weight = std::max(1e-9, weights[index]);
        const double gpu_distance = std::abs((gpu_weight + weight) - target_gpu);
        const double cpu_distance = std::abs(gpu_weight - target_gpu);

        if (gpu_distance <= cpu_distance) {
            gpu_indices.push_back(index);
            gpu_weight += weight;
        } else {
            cpu_indices.push_back(index);
            cpu_weight += weight;
        }
    }

    if (gpu_indices.empty() && !cpu_indices.empty()) {
        gpu_indices.push_back(cpu_indices.back());
        cpu_indices.pop_back();
    }
    if (cpu_indices.empty() && !gpu_indices.empty()) {
        cpu_indices.push_back(gpu_indices.back());
        gpu_indices.pop_back();
    }

    std::sort(gpu_indices.begin(), gpu_indices.end());
    std::sort(cpu_indices.begin(), cpu_indices.end());
    return {gpu_indices, cpu_indices};
}

template <typename T>
Tensor3D<T> hadamard_product_hybrid(const Tensor3D<T>& left, const Tensor3D<T>& right, double gpu_ratio = 0.5, RunMetrics* metrics = nullptr) {
    if (left.depth() != right.depth() || left.second_dim() != right.first_dim()) {
        throw std::invalid_argument("Hybrid multiplication requires tensors with the same depth");
    }

    std::vector<double> weights(left.depth(), 1.0);
    std::vector<std::size_t> dense_candidates;
    std::vector<std::size_t> cpu_indices;
    dense_candidates.reserve(left.depth());
    cpu_indices.reserve(left.depth());
    double total_estimated_work = 0.0;

    for (std::size_t slice = 0; slice < left.depth(); ++slice) {
        weights[slice] = estimate_hadamard_slice_work(left.slice(slice), right.slice(slice));
        total_estimated_work += weights[slice];
        const bool use_dense = should_use_dense_path(
            left.first_dim(),
            left.second_dim(),
            right.second_dim(),
            left.slice(slice).nnz(),
            right.slice(slice).nnz());

        if (use_dense) {
            dense_candidates.push_back(slice);
        } else {
            cpu_indices.push_back(slice);
        }
    }

    if (dense_candidates.empty()) {
        return hadamard_product_openmp(left, right, metrics);
    }

    std::vector<double> dense_weights;
    dense_weights.reserve(dense_candidates.size());
    for (std::size_t index : dense_candidates) {
        dense_weights.push_back(weights[index]);
    }

    const auto [gpu_dense_local, cpu_dense_local] = partition_indices_by_weight(dense_weights, gpu_ratio);

    std::vector<std::size_t> gpu_indices;
    gpu_indices.reserve(gpu_dense_local.size());
    for (std::size_t local_index : gpu_dense_local) {
        gpu_indices.push_back(dense_candidates[local_index]);
    }
    for (std::size_t local_index : cpu_dense_local) {
        cpu_indices.push_back(dense_candidates[local_index]);
    }

    if (gpu_indices.empty() && !dense_candidates.empty()) {
        const std::size_t forced_gpu = dense_candidates.front();
        gpu_indices.push_back(forced_gpu);
        auto it = std::find(cpu_indices.begin(), cpu_indices.end(), forced_gpu);
        if (it != cpu_indices.end()) {
            cpu_indices.erase(it);
        }
    }

    std::sort(gpu_indices.begin(), gpu_indices.end());
    std::sort(cpu_indices.begin(), cpu_indices.end());

    std::future<std::pair<Tensor3D<T>, RunMetrics>> gpu_task;
    if (!gpu_indices.empty()) {
        gpu_task = std::async(std::launch::async, [&]() {
            RunMetrics gpu_metrics;
            const Tensor3D<T> left_gpu = extract_tensor_indices(left, gpu_indices);
            const Tensor3D<T> right_gpu = extract_tensor_indices(right, gpu_indices);
            Tensor3D<T> gpu_result = hadamard_product_cuda(left_gpu, right_gpu, &gpu_metrics);
            return std::make_pair(std::move(gpu_result), gpu_metrics);
        });
    }

    std::future<std::pair<Tensor3D<T>, RunMetrics>> cpu_task;
    if (!cpu_indices.empty()) {
        cpu_task = std::async(std::launch::async, [&]() {
            RunMetrics cpu_metrics;
            const Tensor3D<T> left_cpu = extract_tensor_indices(left, cpu_indices);
            const Tensor3D<T> right_cpu = extract_tensor_indices(right, cpu_indices);
            Tensor3D<T> cpu_result = hadamard_product_openmp(left_cpu, right_cpu, &cpu_metrics);
            return std::make_pair(std::move(cpu_result), cpu_metrics);
        });
    }

    Tensor3D<T> result(left.first_dim(), right.second_dim(), left.depth());
    RunMetrics merged_metrics;
    if (!gpu_indices.empty()) {
        auto gpu_payload = gpu_task.get();
        write_tensor_indices(result, gpu_payload.first, gpu_indices);
        merged_metrics = add_metrics(merged_metrics, gpu_payload.second);
    }

    if (!cpu_indices.empty()) {
        auto cpu_payload = cpu_task.get();
        write_tensor_indices(result, cpu_payload.first, cpu_indices);
        merged_metrics = add_metrics(merged_metrics, cpu_payload.second);
    }

    const auto accumulate_work = [&](const std::vector<std::size_t>& indices) {
        double work = 0.0;
        for (std::size_t index : indices) {
            work += weights[index];
        }
        return work;
    };

    const double gpu_assigned_work = accumulate_work(gpu_indices);
    const double cpu_assigned_work = accumulate_work(cpu_indices);
    const double partition_total = gpu_assigned_work + cpu_assigned_work;

    merged_metrics.gpu_assigned_slices = static_cast<std::uint64_t>(gpu_indices.size());
    merged_metrics.cpu_assigned_slices = static_cast<std::uint64_t>(cpu_indices.size());
    merged_metrics.gpu_assigned_work = gpu_assigned_work;
    merged_metrics.cpu_assigned_work = cpu_assigned_work;
    if (partition_total > 0.0) {
        merged_metrics.gpu_assigned_ratio = gpu_assigned_work / partition_total;
        const double min_work = std::min(gpu_assigned_work, cpu_assigned_work);
        const double max_work = std::max(gpu_assigned_work, cpu_assigned_work);
        merged_metrics.partition_load_imbalance = min_work > 0.0 ? (max_work - min_work) / min_work : 0.0;
    }
    if (merged_metrics.work_units_total <= 0.0) {
        merged_metrics.work_units_total = total_estimated_work;
    }

    result.preprocess();
    if (metrics != nullptr) {
        *metrics = merged_metrics;
    }
    return result;
}

template <typename T>
std::vector<QRResult<T>> tensor_qr_hybrid(const Tensor3D<T>& tensor, double gpu_ratio = 0.5) {
    std::vector<double> weights(tensor.depth(), 1.0);
    for (std::size_t slice = 0; slice < tensor.depth(); ++slice) {
        weights[slice] = estimate_decomposition_slice_work(tensor.slice(slice));
    }
    const auto [gpu_indices, cpu_indices] = partition_indices_by_weight(weights, gpu_ratio);

    auto gpu_task = std::async(std::launch::async, [&]() {
        const Tensor3D<T> subset = extract_tensor_indices(tensor, gpu_indices);
        return tensor_qr_cuda(subset);
    });

    auto cpu_task = std::async(std::launch::async, [&]() {
        const Tensor3D<T> subset = extract_tensor_indices(tensor, cpu_indices);
        return tensor_qr_openmp(subset);
    });

    std::vector<QRResult<T>> results(tensor.depth());
    const std::vector<QRResult<T>> gpu_results = gpu_task.get();
    const std::vector<QRResult<T>> cpu_results = cpu_task.get();

    for (std::size_t index = 0; index < gpu_results.size(); ++index) {
        results[gpu_indices[index]] = gpu_results[index];
    }
    for (std::size_t index = 0; index < cpu_results.size(); ++index) {
        results[cpu_indices[index]] = cpu_results[index];
    }
    return results;
}

template <typename T>
std::vector<SVDResult<T>> tensor_svd_hybrid(const Tensor3D<T>& tensor, double gpu_ratio = 0.5) {
    std::vector<double> weights(tensor.depth(), 1.0);
    for (std::size_t slice = 0; slice < tensor.depth(); ++slice) {
        weights[slice] = estimate_decomposition_slice_work(tensor.slice(slice));
    }
    const auto [gpu_indices, cpu_indices] = partition_indices_by_weight(weights, gpu_ratio);

    auto gpu_task = std::async(std::launch::async, [&]() {
        const Tensor3D<T> subset = extract_tensor_indices(tensor, gpu_indices);
        return tensor_svd_cuda(subset);
    });

    auto cpu_task = std::async(std::launch::async, [&]() {
        const Tensor3D<T> subset = extract_tensor_indices(tensor, cpu_indices);
        return tensor_svd_openmp(subset);
    });

    std::vector<SVDResult<T>> results(tensor.depth());
    const std::vector<SVDResult<T>> gpu_results = gpu_task.get();
    const std::vector<SVDResult<T>> cpu_results = cpu_task.get();

    for (std::size_t index = 0; index < gpu_results.size(); ++index) {
        results[gpu_indices[index]] = gpu_results[index];
    }
    for (std::size_t index = 0; index < cpu_results.size(); ++index) {
        results[cpu_indices[index]] = cpu_results[index];
    }
    return results;
}

} // namespace tensor3d

#endif
