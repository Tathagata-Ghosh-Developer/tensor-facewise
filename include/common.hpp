#ifndef TENSOR3D_COMMON_HPP
#define TENSOR3D_COMMON_HPP

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace tensor3d {

enum class StorageFormat {
    COO,
    ELL
};

enum class ComputeMode {
    Serial,
    OpenMP,
    CUDA,
    Hybrid
};

enum class OperationKind {
    Hadamard,
    QR,
    SVD
};

struct RunMetrics {
    std::uint64_t slices_total = 0;
    std::uint64_t slices_sparse = 0;
    std::uint64_t slices_dense = 0;
    std::uint64_t iterations = 0;
    double flops = 0.0;
    double total_seconds = 0.0;
    double sparse_seconds = 0.0;
    double dense_seconds = 0.0;
    double work_units_total = 0.0;
    double work_units_sparse = 0.0;
    double work_units_dense = 0.0;
    double max_thread_work = 0.0;
    double min_thread_work = 0.0;
    double thread_work_imbalance = 0.0;
    double max_thread_time = 0.0;
    double min_thread_time = 0.0;
    double thread_time_imbalance = 0.0;
    std::uint64_t gpu_assigned_slices = 0;
    std::uint64_t cpu_assigned_slices = 0;
    double gpu_assigned_work = 0.0;
    double cpu_assigned_work = 0.0;
    double gpu_assigned_ratio = 0.0;
    double partition_load_imbalance = 0.0;
};

inline RunMetrics add_metrics(const RunMetrics& lhs, const RunMetrics& rhs) {
    const auto merge_min_positive = [](double a, double b) {
        if (a <= 0.0) {
            return b;
        }
        if (b <= 0.0) {
            return a;
        }
        return std::min(a, b);
    };

    RunMetrics result;
    result.slices_total = lhs.slices_total + rhs.slices_total;
    result.slices_sparse = lhs.slices_sparse + rhs.slices_sparse;
    result.slices_dense = lhs.slices_dense + rhs.slices_dense;
    result.iterations = lhs.iterations + rhs.iterations;
    result.flops = lhs.flops + rhs.flops;
    result.total_seconds = lhs.total_seconds + rhs.total_seconds;
    result.sparse_seconds = lhs.sparse_seconds + rhs.sparse_seconds;
    result.dense_seconds = lhs.dense_seconds + rhs.dense_seconds;
    result.work_units_total = lhs.work_units_total + rhs.work_units_total;
    result.work_units_sparse = lhs.work_units_sparse + rhs.work_units_sparse;
    result.work_units_dense = lhs.work_units_dense + rhs.work_units_dense;
    result.max_thread_work = std::max(lhs.max_thread_work, rhs.max_thread_work);
    result.min_thread_work = merge_min_positive(lhs.min_thread_work, rhs.min_thread_work);
    if (result.min_thread_work > 0.0) {
        result.thread_work_imbalance =
            (result.max_thread_work - result.min_thread_work) / result.min_thread_work;
    }
    result.max_thread_time = std::max(lhs.max_thread_time, rhs.max_thread_time);
    result.min_thread_time = merge_min_positive(lhs.min_thread_time, rhs.min_thread_time);
    if (result.min_thread_time > 0.0) {
        result.thread_time_imbalance =
            (result.max_thread_time - result.min_thread_time) / result.min_thread_time;
    }
    result.gpu_assigned_slices = lhs.gpu_assigned_slices + rhs.gpu_assigned_slices;
    result.cpu_assigned_slices = lhs.cpu_assigned_slices + rhs.cpu_assigned_slices;
    result.gpu_assigned_work = lhs.gpu_assigned_work + rhs.gpu_assigned_work;
    result.cpu_assigned_work = lhs.cpu_assigned_work + rhs.cpu_assigned_work;
    const double total_partition_work = result.gpu_assigned_work + result.cpu_assigned_work;
    if (total_partition_work > 0.0) {
        result.gpu_assigned_ratio = result.gpu_assigned_work / total_partition_work;
        const double min_work = std::min(result.gpu_assigned_work, result.cpu_assigned_work);
        const double max_work = std::max(result.gpu_assigned_work, result.cpu_assigned_work);
        result.partition_load_imbalance = min_work > 0.0 ? (max_work - min_work) / min_work : 0.0;
    }
    return result;
}

inline std::string trim_copy(const std::string& value) {
    std::size_t begin = 0;
    std::size_t end = value.size();
    while (begin < end && std::isspace(static_cast<unsigned char>(value[begin])) != 0) {
        ++begin;
    }
    while (end > begin && std::isspace(static_cast<unsigned char>(value[end - 1])) != 0) {
        --end;
    }
    return value.substr(begin, end - begin);
}

inline std::vector<std::string> split_csv_line(const std::string& line) {
    std::vector<std::string> tokens;
    std::string token;
    std::stringstream stream(line);
    while (std::getline(stream, token, ',')) {
        token = trim_copy(token);
        if (!token.empty()) {
            tokens.push_back(token);
        }
    }
    return tokens;
}

template <typename T>
inline T zero_threshold() {
    return static_cast<T>(1e-12);
}

template <typename T>
inline bool is_near_zero(const T& value) {
    return std::abs(value) <= zero_threshold<T>();
}

} // namespace tensor3d

#endif
