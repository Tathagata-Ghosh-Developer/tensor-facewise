#ifndef TENSOR3D_TENSOR_HPP
#define TENSOR3D_TENSOR_HPP

#include "common.hpp"
#include "dense.hpp"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tensor3d {

template <typename T>
struct CooSlice {
    std::size_t rows = 0;
    std::size_t cols = 0;
    std::vector<std::size_t> row_indices;
    std::vector<std::size_t> col_indices;
    std::vector<T> values;

    CooSlice() = default;

    CooSlice(std::size_t row_count, std::size_t col_count)
        : rows(row_count), cols(col_count) {}

    std::size_t nnz() const {
        return values.size();
    }

    void add_element(std::size_t row, std::size_t col, const T& value) {
        if (is_near_zero(value)) {
            return;
        }
        row_indices.push_back(row);
        col_indices.push_back(col);
        values.push_back(value);
    }
};

template <typename T>
struct EllSlice {
    std::size_t rows = 0;
    std::size_t cols = 0;
    std::size_t width = 0;
    std::vector<int> column_indices;
    std::vector<T> values;

    EllSlice() = default;

    EllSlice(std::size_t row_count, std::size_t col_count, std::size_t max_width)
        : rows(row_count), cols(col_count), width(max_width), column_indices(row_count * max_width, -1), values(row_count * max_width, T{}) {}

    static EllSlice from_coo(const CooSlice<T>& coo) {
        std::vector<std::vector<std::pair<std::size_t, T>>> per_row(coo.rows);
        for (std::size_t index = 0; index < coo.nnz(); ++index) {
            per_row[coo.row_indices[index]].push_back({coo.col_indices[index], coo.values[index]});
        }

        std::size_t max_width = 0;
        for (const auto& row_entries : per_row) {
            max_width = std::max(max_width, row_entries.size());
        }

        EllSlice result(coo.rows, coo.cols, max_width);
        for (std::size_t row = 0; row < per_row.size(); ++row) {
            auto& row_entries = per_row[row];
            std::sort(row_entries.begin(), row_entries.end(), [](const auto& lhs, const auto& rhs) {
                return lhs.first < rhs.first;
            });
            for (std::size_t offset = 0; offset < row_entries.size(); ++offset) {
                result.column_indices[row * max_width + offset] = static_cast<int>(row_entries[offset].first);
                result.values[row * max_width + offset] = row_entries[offset].second;
            }
        }

        return result;
    }

    CooSlice<T> to_coo() const {
        CooSlice<T> coo(rows, cols);
        for (std::size_t row = 0; row < rows; ++row) {
            for (std::size_t offset = 0; offset < width; ++offset) {
                const int col = column_indices[row * width + offset];
                if (col < 0) {
                    continue;
                }
                const T value = values[row * width + offset];
                if (is_near_zero(value)) {
                    continue;
                }
                coo.add_element(row, static_cast<std::size_t>(col), value);
            }
        }
        return coo;
    }
};

template <typename T>
struct SparseSlice {
    std::size_t rows = 0;
    std::size_t cols = 0;
    StorageFormat preferred_format = StorageFormat::COO;
    CooSlice<T> coo;
    EllSlice<T> ell;
    bool ell_ready = false;

    SparseSlice() = default;

    SparseSlice(std::size_t row_count, std::size_t col_count)
        : rows(row_count), cols(col_count), coo(row_count, col_count) {}

    void add_element(std::size_t row, std::size_t col, const T& value) {
        coo.add_element(row, col, value);
    }

    std::size_t nnz() const {
        return coo.nnz();
    }

    void preprocess(double ell_density_threshold = 0.20) {
        const double total_entries = static_cast<double>(rows) * static_cast<double>(cols);
        const double density = total_entries > 0.0 ? static_cast<double>(nnz()) / total_entries : 0.0;
        if (density >= ell_density_threshold && nnz() > 0) {
            ell = EllSlice<T>::from_coo(coo);
            ell_ready = true;
            preferred_format = StorageFormat::ELL;
        } else {
            ell_ready = false;
            preferred_format = StorageFormat::COO;
        }
    }

    CooSlice<T> to_coo() const {
        if (preferred_format == StorageFormat::ELL && ell_ready) {
            return ell.to_coo();
        }
        return coo;
    }

    DenseMatrix<T> to_dense() const {
        DenseMatrix<T> dense(rows, cols, T{});
        const CooSlice<T> source = to_coo();
        for (std::size_t index = 0; index < source.nnz(); ++index) {
            dense(source.row_indices[index], source.col_indices[index]) = source.values[index];
        }
        return dense;
    }
};

template <typename T>
class Tensor3D {
public:
    Tensor3D() = default;

    Tensor3D(std::size_t first_dim, std::size_t second_dim, std::size_t depth)
        : n1_(first_dim), n2_(second_dim), n3_(depth), slices_(depth, SparseSlice<T>(first_dim, second_dim)) {}

    std::size_t first_dim() const {
        return n1_;
    }

    std::size_t second_dim() const {
        return n2_;
    }

    std::size_t depth() const {
        return n3_;
    }

    SparseSlice<T>& slice(std::size_t index) {
        return slices_.at(index);
    }

    const SparseSlice<T>& slice(std::size_t index) const {
        return slices_.at(index);
    }

    std::vector<SparseSlice<T>>& slices() {
        return slices_;
    }

    const std::vector<SparseSlice<T>>& slices() const {
        return slices_;
    }

    void preprocess(double ell_density_threshold = 0.20) {
        for (auto& slice : slices_) {
            slice.preprocess(ell_density_threshold);
        }
    }

private:
    std::size_t n1_ = 0;
    std::size_t n2_ = 0;
    std::size_t n3_ = 0;
    std::vector<SparseSlice<T>> slices_;
};

} // namespace tensor3d

#endif
