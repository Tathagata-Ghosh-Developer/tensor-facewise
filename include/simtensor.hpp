#ifndef TENSOR3D_SIMTENSOR_HPP
#define TENSOR3D_SIMTENSOR_HPP

#include "common.hpp"
#include "tensor.hpp"

#include <cstdint>
#include <fstream>
#include <iomanip>
#include <ios>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_set>

namespace tensor3d {

template <typename T>
Tensor3D<T> generate_synthetic_tensor(std::size_t rows,
                                      std::size_t cols,
                                      std::size_t depth,
                                      double density,
                                      std::uint64_t seed,
                                      T min_value = static_cast<T>(-1),
                                      T max_value = static_cast<T>(1)) {
    if (density < 0.0 || density > 1.0) {
        throw std::invalid_argument("Density must be between 0 and 1");
    }

    Tensor3D<T> tensor(rows, cols, depth);
    std::mt19937_64 generator(seed);
    std::uniform_real_distribution<double> probability(0.0, 1.0);
    std::uniform_real_distribution<double> values(static_cast<double>(min_value), static_cast<double>(max_value));
    std::uniform_int_distribution<std::size_t> row_picker(0, rows == 0 ? 0 : rows - 1);
    std::uniform_int_distribution<std::size_t> col_picker(0, cols == 0 ? 0 : cols - 1);

    for (std::size_t slice_index = 0; slice_index < depth; ++slice_index) {
        const std::size_t target_nnz = static_cast<std::size_t>(static_cast<double>(rows) * static_cast<double>(cols) * density);
        std::unordered_set<std::uint64_t> occupied;
        while (occupied.size() < target_nnz) {
            const std::size_t row = row_picker(generator);
            const std::size_t col = col_picker(generator);
            const std::uint64_t key = static_cast<std::uint64_t>(row) * static_cast<std::uint64_t>(cols) + static_cast<std::uint64_t>(col);
            if (!occupied.insert(key).second) {
                continue;
            }
            tensor.slice(slice_index).add_element(row, col, static_cast<T>(values(generator)));
        }

        if (target_nnz == 0 && rows > 0 && cols > 0 && probability(generator) < 0.05) {
            tensor.slice(slice_index).add_element(0, 0, static_cast<T>(values(generator)));
        }
    }

    tensor.preprocess();
    return tensor;
}

template <typename T>
Tensor3D<T> load_tensor_from_csv(const std::string& path, std::size_t rows, std::size_t cols, std::size_t depth) {
    Tensor3D<T> tensor(rows, cols, depth);
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Unable to open tensor CSV: " + path);
    }

    std::string line;
    while (std::getline(input, line)) {
        line = trim_copy(line);
        if (line.empty() || line[0] == '#') {
            continue;
        }

        const std::vector<std::string> tokens = split_csv_line(line);
        if (tokens.size() < 4) {
            continue;
        }

        const std::size_t slice = static_cast<std::size_t>(std::stoull(tokens[0]));
        const std::size_t row = static_cast<std::size_t>(std::stoull(tokens[1]));
        const std::size_t col = static_cast<std::size_t>(std::stoull(tokens[2]));
        const T value = static_cast<T>(std::stod(tokens[3]));

        if (slice >= depth || row >= rows || col >= cols) {
            throw std::runtime_error("CSV entry is out of bounds");
        }

        tensor.slice(slice).add_element(row, col, value);
    }

    tensor.preprocess();
    return tensor;
}

template <typename T>
void save_tensor_to_csv(const Tensor3D<T>& tensor, const std::string& path) {
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("Unable to open output CSV: " + path);
    }

    output << std::scientific << std::setprecision(15);
    for (std::size_t slice = 0; slice < tensor.depth(); ++slice) {
        const CooSlice<T> coo = tensor.slice(slice).to_coo();
        for (std::size_t index = 0; index < coo.nnz(); ++index) {
            output << slice << ',' << coo.row_indices[index] << ',' << coo.col_indices[index] << ',' << coo.values[index] << '\n';
        }
    }
}

template <typename T>
DenseMatrix<T> slice_to_dense(const SparseSlice<T>& slice) {
    return slice.to_dense();
}

} // namespace tensor3d

#endif
