#include "cuda_bridge.hpp"
#include "hybrid_ops.hpp"
#include "simtensor.hpp"
#include "sparse_ops.hpp"

#include <algorithm>
#include <chrono>
#include <exception>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

using tensor3d::ComputeMode;
using tensor3d::OperationKind;

struct CliOptions {
    ComputeMode mode = ComputeMode::Serial;
    OperationKind operation = OperationKind::Hadamard;
    std::size_t rows_a = 64;
    std::size_t cols_a = 64;
    std::size_t rows_b = 64;
    std::size_t cols_b = 64;
    std::size_t depth = 8;
    double density_a = 0.08;
    double density_b = 0.08;
    std::uint64_t seed = 12345;
    double gpu_ratio = 0.5;
    double preprocess_threshold = 0.20;
    int omp_chunk = 0;
    int warmup_runs = 1;
    bool timing_only = false;
    std::string input_a;
    std::string input_b;
    std::string output_path = "output.csv";
};

ComputeMode parse_mode(const std::string& value) {
    if (value == "serial") {
        return ComputeMode::Serial;
    }
    if (value == "omp") {
        return ComputeMode::OpenMP;
    }
    if (value == "cuda") {
        return ComputeMode::CUDA;
    }
    if (value == "hybrid") {
        return ComputeMode::Hybrid;
    }
    throw std::invalid_argument("Unknown compute mode: " + value);
}

OperationKind parse_operation(const std::string& value) {
    if (value == "hadamard" || value == "product") {
        return OperationKind::Hadamard;
    }
    if (value == "qr") {
        return OperationKind::QR;
    }
    if (value == "svd") {
        return OperationKind::SVD;
    }
    throw std::invalid_argument("Unknown operation: " + value);
}

int resolve_omp_chunk(std::size_t depth, int requested_chunk) {
    if (requested_chunk > 0) {
        return requested_chunk;
    }
#ifdef _OPENMP
    int threads = omp_get_max_threads();
    if (threads <= 0) {
        threads = 1;
    }
    const std::size_t suggested = std::max<std::size_t>(
        1,
        depth / static_cast<std::size_t>(threads * 2));
    return static_cast<int>(std::min<std::size_t>(suggested, 64));
#else
    (void)depth;
    return 1;
#endif
}

CliOptions parse_arguments(int argc, char** argv) {
    CliOptions options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        const auto next = [&]() -> std::string {
            if (index + 1 >= argc) {
                throw std::invalid_argument("Missing value after " + argument);
            }
            return argv[++index];
        };

        if (argument == "--mode") {
            options.mode = parse_mode(next());
        } else if (argument == "--operation") {
            options.operation = parse_operation(next());
        } else if (argument == "--rows-a") {
            options.rows_a = static_cast<std::size_t>(std::stoull(next()));
        } else if (argument == "--cols-a") {
            options.cols_a = static_cast<std::size_t>(std::stoull(next()));
        } else if (argument == "--rows-b") {
            options.rows_b = static_cast<std::size_t>(std::stoull(next()));
        } else if (argument == "--cols-b") {
            options.cols_b = static_cast<std::size_t>(std::stoull(next()));
        } else if (argument == "--depth") {
            options.depth = static_cast<std::size_t>(std::stoull(next()));
        } else if (argument == "--density-a") {
            options.density_a = std::stod(next());
        } else if (argument == "--density-b") {
            options.density_b = std::stod(next());
        } else if (argument == "--seed") {
            options.seed = static_cast<std::uint64_t>(std::stoull(next()));
        } else if (argument == "--gpu-ratio") {
            options.gpu_ratio = std::stod(next());
        } else if (argument == "--preprocess-threshold") {
            options.preprocess_threshold = std::stod(next());
        } else if (argument == "--omp-chunk") {
            options.omp_chunk = std::max(0, std::stoi(next()));
        } else if (argument == "--warmup") {
            options.warmup_runs = std::max(0, std::stoi(next()));
        } else if (argument == "--input-a") {
            options.input_a = next();
        } else if (argument == "--input-b") {
            options.input_b = next();
        } else if (argument == "--out") {
            options.output_path = next();
        } else if (argument == "--timing-only") {
            options.timing_only = true;
        } else if (argument == "--help") {
            throw std::runtime_error("help");
        } else {
            throw std::invalid_argument("Unknown argument: " + argument);
        }
    }
    return options;
}

template <typename T>
tensor3d::Tensor3D<T> load_or_generate_tensor(const std::string& path,
                                             std::size_t rows,
                                             std::size_t cols,
                                             std::size_t depth,
                                             double density,
                                             std::uint64_t seed) {
    if (!path.empty()) {
        return tensor3d::load_tensor_from_csv<T>(path, rows, cols, depth);
    }
    return tensor3d::generate_synthetic_tensor<T>(rows, cols, depth, density, seed);
}

template <typename T>
void print_qr_summary(const std::vector<tensor3d::QRResult<T>>& results) {
    std::cout << "QR slices: " << results.size() << '\n';
}

template <typename T>
void print_svd_summary(const std::vector<tensor3d::SVDResult<T>>& results) {
    std::cout << "SVD slices: " << results.size() << '\n';
}

struct TensorStorageStats {
    std::uint64_t slices_total = 0;
    std::uint64_t slices_ell = 0;
    std::uint64_t slices_coo = 0;
    std::uint64_t nnz_total = 0;
    std::uint64_t ell_slots_total = 0;
    double dense_bytes = 0.0;
    double coo_bytes = 0.0;
    double ell_bytes = 0.0;
    double active_bytes = 0.0;
    double compression_dense_to_coo = 0.0;
    double compression_dense_to_ell = 0.0;
    double compression_dense_to_active = 0.0;
    double ell_slot_utilization = 0.0;
};

template <typename T>
std::size_t estimate_ell_slots(const tensor3d::SparseSlice<T>& slice) {
    if (slice.rows == 0 || slice.cols == 0 || slice.coo.nnz() == 0) {
        return 0;
    }

    std::vector<std::size_t> counts_per_row(slice.rows, 0);
    for (std::size_t row : slice.coo.row_indices) {
        if (row < counts_per_row.size()) {
            ++counts_per_row[row];
        }
    }
    const std::size_t max_width = *std::max_element(counts_per_row.begin(), counts_per_row.end());
    return slice.rows * max_width;
}

template <typename T>
TensorStorageStats compute_tensor_storage_stats(const tensor3d::Tensor3D<T>& tensor) {
    TensorStorageStats stats;
    stats.slices_total = static_cast<std::uint64_t>(tensor.depth());
    stats.dense_bytes =
        static_cast<double>(tensor.first_dim()) *
        static_cast<double>(tensor.second_dim()) *
        static_cast<double>(tensor.depth()) *
        static_cast<double>(sizeof(T));

    constexpr double kIndexBytes = static_cast<double>(2 * sizeof(std::size_t));
    constexpr double kValueBytes = static_cast<double>(sizeof(T));
    constexpr double kEllColumnBytes = static_cast<double>(sizeof(int));

    for (std::size_t slice_index = 0; slice_index < tensor.depth(); ++slice_index) {
        const auto& slice = tensor.slice(slice_index);
        const std::size_t nnz = slice.nnz();
        const double coo_slice_bytes = static_cast<double>(nnz) * (kIndexBytes + kValueBytes);

        std::size_t ell_slots = 0;
        if (slice.ell_ready) {
            ell_slots = slice.rows * slice.ell.width;
        } else {
            ell_slots = estimate_ell_slots(slice);
        }
        const double ell_slice_bytes = static_cast<double>(ell_slots) * (kEllColumnBytes + kValueBytes);

        stats.nnz_total += static_cast<std::uint64_t>(nnz);
        stats.ell_slots_total += static_cast<std::uint64_t>(ell_slots);
        stats.coo_bytes += coo_slice_bytes;
        stats.ell_bytes += ell_slice_bytes;

        const bool active_ell = slice.preferred_format == tensor3d::StorageFormat::ELL && slice.ell_ready;
        if (active_ell) {
            ++stats.slices_ell;
            stats.active_bytes += ell_slice_bytes;
        } else {
            ++stats.slices_coo;
            stats.active_bytes += coo_slice_bytes;
        }
    }

    if (stats.coo_bytes > 0.0) {
        stats.compression_dense_to_coo = stats.dense_bytes / stats.coo_bytes;
    }
    if (stats.ell_bytes > 0.0) {
        stats.compression_dense_to_ell = stats.dense_bytes / stats.ell_bytes;
    }
    if (stats.active_bytes > 0.0) {
        stats.compression_dense_to_active = stats.dense_bytes / stats.active_bytes;
    }
    if (stats.ell_slots_total > 0) {
        stats.ell_slot_utilization =
            static_cast<double>(stats.nnz_total) / static_cast<double>(stats.ell_slots_total);
    }

    return stats;
}

double bytes_to_mib(double bytes) {
    return bytes / (1024.0 * 1024.0);
}

void print_storage_metrics(const TensorStorageStats& stats) {
    std::cout << "storage slices_total=" << stats.slices_total
              << " slices_ell=" << stats.slices_ell
              << " slices_coo=" << stats.slices_coo
              << " nnz_total=" << stats.nnz_total
              << " ell_slots_total=" << stats.ell_slots_total
              << " dense_mem_mib=" << bytes_to_mib(stats.dense_bytes)
              << " coo_mem_mib=" << bytes_to_mib(stats.coo_bytes)
              << " ell_mem_mib=" << bytes_to_mib(stats.ell_bytes)
              << " active_mem_mib=" << bytes_to_mib(stats.active_bytes)
              << " compression_dense_to_coo=" << stats.compression_dense_to_coo
              << " compression_dense_to_ell=" << stats.compression_dense_to_ell
              << " compression_dense_to_active=" << stats.compression_dense_to_active
              << " ell_slot_utilization=" << stats.ell_slot_utilization
              << '\n';
}

void print_run_metrics(const tensor3d::RunMetrics& metrics) {
    std::cout << "metrics slices_total=" << metrics.slices_total
              << " dense_slices=" << metrics.slices_dense
              << " sparse_slices=" << metrics.slices_sparse
              << " iterations=" << metrics.iterations
              << " flops=" << metrics.flops
              << " total_sec=" << metrics.total_seconds
              << " dense_sec=" << metrics.dense_seconds
              << " sparse_sec=" << metrics.sparse_seconds
              << " work_units_total=" << metrics.work_units_total
              << " work_units_dense=" << metrics.work_units_dense
              << " work_units_sparse=" << metrics.work_units_sparse
              << " max_thread_work=" << metrics.max_thread_work
              << " min_thread_work=" << metrics.min_thread_work
              << " thread_work_imbalance=" << metrics.thread_work_imbalance
              << " max_thread_time=" << metrics.max_thread_time
              << " min_thread_time=" << metrics.min_thread_time
              << " thread_time_imbalance=" << metrics.thread_time_imbalance
              << " gpu_assigned_slices=" << metrics.gpu_assigned_slices
              << " cpu_assigned_slices=" << metrics.cpu_assigned_slices
              << " gpu_assigned_work=" << metrics.gpu_assigned_work
              << " cpu_assigned_work=" << metrics.cpu_assigned_work
              << " gpu_assigned_ratio=" << metrics.gpu_assigned_ratio
              << " partition_load_imbalance=" << metrics.partition_load_imbalance
              << '\n';
}

double estimate_qr_flops(std::size_t rows, std::size_t cols, std::size_t slices) {
    const double m = static_cast<double>(rows);
    const double n = static_cast<double>(cols);
    const double per_slice = 2.0 * m * n * n - (2.0 / 3.0) * n * n * n;
    return std::max(0.0, per_slice) * static_cast<double>(slices);
}

double estimate_svd_flops(std::size_t rows, std::size_t cols, std::size_t slices) {
    const double m = static_cast<double>(rows);
    const double n = static_cast<double>(cols);
    const double per_slice = 4.0 * m * n * n + 8.0 * n * n * n;
    return per_slice * static_cast<double>(slices);
}

void print_help() {
    std::cout << "Usage: tensor_app [options]\n"
              << "  --mode serial|omp|cuda|hybrid\n"
              << "  --operation hadamard|qr|svd\n"
              << "  --rows-a N --cols-a N --rows-b N --cols-b N --depth N\n"
              << "  --density-a X --density-b X --seed N\n"
              << "  --input-a file.csv --input-b file.csv\n"
              << "  --gpu-ratio X --preprocess-threshold X\n"
              << "  --omp-chunk N (0 = auto) --warmup N\n"
              << "  --out output.csv --timing-only\n";
}

} // namespace

int main(int argc, char** argv) {
    try {
        const CliOptions options = parse_arguments(argc, argv);
        using value_type = double;

#ifdef _OPENMP
    const int omp_chunk = resolve_omp_chunk(options.depth, options.omp_chunk);
    omp_set_schedule(omp_sched_dynamic, omp_chunk);
#endif

        tensor3d::Tensor3D<value_type> tensor_a = load_or_generate_tensor<value_type>(
            options.input_a, options.rows_a, options.cols_a, options.depth, options.density_a, options.seed);
        tensor_a.preprocess(options.preprocess_threshold);

        tensor3d::Tensor3D<value_type> tensor_b = load_or_generate_tensor<value_type>(
            options.input_b, options.rows_b, options.cols_b, options.depth, options.density_b, options.seed + 1);
        tensor_b.preprocess(options.preprocess_threshold);

        const TensorStorageStats storage_stats_a = compute_tensor_storage_stats(tensor_a);

        const auto start = std::chrono::high_resolution_clock::now();

        if (options.operation == OperationKind::Hadamard) {
            tensor3d::Tensor3D<value_type> result;
            tensor3d::RunMetrics metrics;

            for (int warm = 0; warm < options.warmup_runs; ++warm) {
                switch (options.mode) {
                    case ComputeMode::Serial:
                        (void)tensor3d::hadamard_product_serial(tensor_a, tensor_b, nullptr);
                        break;
                    case ComputeMode::OpenMP:
                        (void)tensor3d::hadamard_product_openmp(tensor_a, tensor_b, nullptr);
                        break;
                    case ComputeMode::CUDA:
                        (void)tensor3d::hadamard_product_cuda(tensor_a, tensor_b, nullptr);
                        break;
                    case ComputeMode::Hybrid:
                        (void)tensor3d::hadamard_product_hybrid(tensor_a, tensor_b, options.gpu_ratio, nullptr);
                        break;
                }
            }

            switch (options.mode) {
                case ComputeMode::Serial:
                    result = tensor3d::hadamard_product_serial(tensor_a, tensor_b, &metrics);
                    break;
                case ComputeMode::OpenMP:
                    result = tensor3d::hadamard_product_openmp(tensor_a, tensor_b, &metrics);
                    break;
                case ComputeMode::CUDA:
                    result = tensor3d::hadamard_product_cuda(tensor_a, tensor_b, &metrics);
                    break;
                case ComputeMode::Hybrid:
                    result = tensor3d::hadamard_product_hybrid(tensor_a, tensor_b, options.gpu_ratio, &metrics);
                    break;
            }

            const auto stop = std::chrono::high_resolution_clock::now();
            const std::chrono::duration<double> elapsed = stop - start;
            std::cout << "mode=" << static_cast<int>(options.mode)
                      << " operation=hadamard time_sec=" << elapsed.count() << '\n';
            print_run_metrics(metrics);
            print_storage_metrics(storage_stats_a);
            if (!options.timing_only) {
                tensor3d::save_tensor_to_csv(result, options.output_path);
            }
        } else if (options.operation == OperationKind::QR) {
            std::vector<tensor3d::QRResult<value_type>> results;

            for (int warm = 0; warm < options.warmup_runs; ++warm) {
                switch (options.mode) {
                    case ComputeMode::Serial:
                        (void)tensor3d::tensor_qr_serial(tensor_a);
                        break;
                    case ComputeMode::OpenMP:
                        (void)tensor3d::tensor_qr_openmp(tensor_a);
                        break;
                    case ComputeMode::CUDA:
                        (void)tensor3d::tensor_qr_cuda(tensor_a);
                        break;
                    case ComputeMode::Hybrid:
                        (void)tensor3d::tensor_qr_hybrid(tensor_a, options.gpu_ratio);
                        break;
                }
            }

            switch (options.mode) {
                case ComputeMode::Serial:
                    results = tensor3d::tensor_qr_serial(tensor_a);
                    break;
                case ComputeMode::OpenMP:
                    results = tensor3d::tensor_qr_openmp(tensor_a);
                    break;
                case ComputeMode::CUDA:
                    results = tensor3d::tensor_qr_cuda(tensor_a);
                    break;
                case ComputeMode::Hybrid:
                    results = tensor3d::tensor_qr_hybrid(tensor_a, options.gpu_ratio);
                    break;
            }
            const auto stop = std::chrono::high_resolution_clock::now();
            const std::chrono::duration<double> elapsed = stop - start;
            std::cout << "mode=" << static_cast<int>(options.mode)
                      << " operation=qr time_sec=" << elapsed.count() << '\n';
            std::cout << "metrics slices_total=" << tensor_a.depth()
                      << " iterations=" << tensor_a.depth()
                      << " flops=" << estimate_qr_flops(tensor_a.first_dim(), tensor_a.second_dim(), tensor_a.depth())
                      << " total_sec=" << elapsed.count() << '\n';
            print_storage_metrics(storage_stats_a);
            if (!options.timing_only) {
                print_qr_summary(results);
            }
        } else {
            std::vector<tensor3d::SVDResult<value_type>> results;

            for (int warm = 0; warm < options.warmup_runs; ++warm) {
                switch (options.mode) {
                    case ComputeMode::Serial:
                        (void)tensor3d::tensor_svd_serial(tensor_a);
                        break;
                    case ComputeMode::OpenMP:
                        (void)tensor3d::tensor_svd_openmp(tensor_a);
                        break;
                    case ComputeMode::CUDA:
                        (void)tensor3d::tensor_svd_cuda(tensor_a);
                        break;
                    case ComputeMode::Hybrid:
                        (void)tensor3d::tensor_svd_hybrid(tensor_a, options.gpu_ratio);
                        break;
                }
            }

            switch (options.mode) {
                case ComputeMode::Serial:
                    results = tensor3d::tensor_svd_serial(tensor_a);
                    break;
                case ComputeMode::OpenMP:
                    results = tensor3d::tensor_svd_openmp(tensor_a);
                    break;
                case ComputeMode::CUDA:
                    results = tensor3d::tensor_svd_cuda(tensor_a);
                    break;
                case ComputeMode::Hybrid:
                    results = tensor3d::tensor_svd_hybrid(tensor_a, options.gpu_ratio);
                    break;
            }
            const auto stop = std::chrono::high_resolution_clock::now();
            const std::chrono::duration<double> elapsed = stop - start;
            std::cout << "mode=" << static_cast<int>(options.mode)
                      << " operation=svd time_sec=" << elapsed.count() << '\n';
            std::cout << "metrics slices_total=" << tensor_a.depth()
                      << " iterations=" << tensor_a.depth()
                      << " flops=" << estimate_svd_flops(tensor_a.first_dim(), tensor_a.second_dim(), tensor_a.depth())
                      << " total_sec=" << elapsed.count() << '\n';
            print_storage_metrics(storage_stats_a);
            if (!options.timing_only) {
                print_svd_summary(results);
            }
        }

        return 0;
    } catch (const std::runtime_error& error) {
        if (std::string(error.what()) == "help") {
            print_help();
            return 0;
        }
        std::cerr << "Runtime error: " << error.what() << '\n';
        return 1;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
