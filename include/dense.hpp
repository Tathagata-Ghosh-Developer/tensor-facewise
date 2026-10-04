#ifndef TENSOR3D_DENSE_HPP
#define TENSOR3D_DENSE_HPP

#include "common.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace tensor3d {

template <typename T>
class DenseMatrix {
public:
    DenseMatrix() = default;

    DenseMatrix(std::size_t rows, std::size_t cols, const T& value = T{}) {
        assign(rows, cols, value);
    }

    void assign(std::size_t rows, std::size_t cols, const T& value = T{}) {
        rows_ = rows;
        cols_ = cols;
        data_.assign(rows * cols, value);
    }

    std::size_t rows() const {
        return rows_;
    }

    std::size_t cols() const {
        return cols_;
    }

    bool empty() const {
        return rows_ == 0 || cols_ == 0;
    }

    T& operator()(std::size_t row, std::size_t col) {
        return data_.at(row * cols_ + col);
    }

    const T& operator()(std::size_t row, std::size_t col) const {
        return data_.at(row * cols_ + col);
    }

    const std::vector<T>& raw() const {
        return data_; 
    }

    std::vector<T>& raw() {
        return data_;
    }

    static DenseMatrix identity(std::size_t size) {
        DenseMatrix result(size, size, T{});
        for (std::size_t index = 0; index < size; ++index) {
            result(index, index) = static_cast<T>(1);
        }
        return result;
    }

private:
    std::size_t rows_ = 0;
    std::size_t cols_ = 0;
    std::vector<T> data_;
};

template <typename T>
DenseMatrix<T> transpose(const DenseMatrix<T>& matrix) {
    DenseMatrix<T> result(matrix.cols(), matrix.rows(), T{});
    for (std::size_t row = 0; row < matrix.rows(); ++row) {
        for (std::size_t col = 0; col < matrix.cols(); ++col) {
            result(col, row) = matrix(row, col);
        }
    }
    return result;
}

template <typename T>
DenseMatrix<T> multiply(const DenseMatrix<T>& left, const DenseMatrix<T>& right) {
    if (left.cols() != right.rows()) {
        throw std::invalid_argument("Dense multiply dimension mismatch");
    }

    DenseMatrix<T> result(left.rows(), right.cols(), T{});
    for (std::size_t i = 0; i < left.rows(); ++i) {
        for (std::size_t k = 0; k < left.cols(); ++k) {
            const T left_value = left(i, k);
            if (is_near_zero(left_value)) {
                continue;
            }
            for (std::size_t j = 0; j < right.cols(); ++j) {
                result(i, j) += left_value * right(k, j);
            }
        }
    }
    return result;
}

template <typename T>
struct QRResult {
    DenseMatrix<T> q;
    DenseMatrix<T> r;
};

template <typename T>
struct SVDResult {
    DenseMatrix<T> u;
    std::vector<T> singular_values;
    DenseMatrix<T> vt;
};

template <typename T>
struct SymmetricEigenResult {
    DenseMatrix<T> eigenvectors;
    std::vector<T> eigenvalues;
};

template <typename T>
QRResult<T> qr_gram_schmidt(const DenseMatrix<T>& matrix) {
    const std::size_t m = matrix.rows();
    const std::size_t n = matrix.cols();

    DenseMatrix<T> q(m, n, T{});
    DenseMatrix<T> r(n, n, T{});

    for (std::size_t col = 0; col < n; ++col) {
        std::vector<T> v(m, T{});
        for (std::size_t row = 0; row < m; ++row) {
            v[row] = matrix(row, col);
        }

        for (std::size_t prev = 0; prev < col; ++prev) {
            T dot_product = T{};
            for (std::size_t row = 0; row < m; ++row) {
                dot_product += q(row, prev) * matrix(row, col);
            }
            r(prev, col) = dot_product;
            for (std::size_t row = 0; row < m; ++row) {
                v[row] -= dot_product * q(row, prev);
            }
        }

        T norm_sq = T{};
        for (std::size_t row = 0; row < m; ++row) {
            norm_sq += v[row] * v[row];
        }

        const T norm = static_cast<T>(std::sqrt(static_cast<long double>(norm_sq)));
        r(col, col) = norm;
        if (!is_near_zero(norm)) {
            for (std::size_t row = 0; row < m; ++row) {
                q(row, col) = v[row] / norm;
            }
        }
    }

    return {q, r};
}

template <typename T>
SymmetricEigenResult<T> jacobi_eigen_symmetric(DenseMatrix<T> matrix, std::size_t max_iterations = 128, T tolerance = static_cast<T>(1e-10)) {
    if (matrix.rows() != matrix.cols()) {
        throw std::invalid_argument("Jacobi eigen decomposition requires a square matrix");
    }

    const std::size_t n = matrix.rows();
    DenseMatrix<T> eigenvectors = DenseMatrix<T>::identity(n);

    for (std::size_t iteration = 0; iteration < max_iterations; ++iteration) {
        std::size_t pivot_row = 0;
        std::size_t pivot_col = 1;
        T pivot_value = T{};

        for (std::size_t row = 0; row < n; ++row) {
            for (std::size_t col = row + 1; col < n; ++col) {
                const T current = std::abs(matrix(row, col));
                if (current > pivot_value) {
                    pivot_value = current;
                    pivot_row = row;
                    pivot_col = col;
                }
            }
        }

        if (pivot_value <= tolerance) {
            break;
        }

        const T app = matrix(pivot_row, pivot_row);
        const T aqq = matrix(pivot_col, pivot_col);
        const T apq = matrix(pivot_row, pivot_col);
        const T theta = static_cast<T>(0.5) * std::atan2(static_cast<long double>(2 * apq), static_cast<long double>(aqq - app));
        const T cosine = static_cast<T>(std::cos(static_cast<long double>(theta)));
        const T sine = static_cast<T>(std::sin(static_cast<long double>(theta)));

        for (std::size_t k = 0; k < n; ++k) {
            if (k == pivot_row || k == pivot_col) {
                continue;
            }

            const T aik = matrix(pivot_row, k);
            const T aqk = matrix(pivot_col, k);
            matrix(pivot_row, k) = cosine * aik - sine * aqk;
            matrix(k, pivot_row) = matrix(pivot_row, k);
            matrix(pivot_col, k) = sine * aik + cosine * aqk;
            matrix(k, pivot_col) = matrix(pivot_col, k);
        }

        const T app_new = cosine * cosine * app - static_cast<T>(2) * sine * cosine * apq + sine * sine * aqq;
        const T aqq_new = sine * sine * app + static_cast<T>(2) * sine * cosine * apq + cosine * cosine * aqq;
        matrix(pivot_row, pivot_row) = app_new;
        matrix(pivot_col, pivot_col) = aqq_new;
        matrix(pivot_row, pivot_col) = T{};
        matrix(pivot_col, pivot_row) = T{};

        for (std::size_t k = 0; k < n; ++k) {
            const T vip = eigenvectors(k, pivot_row);
            const T viq = eigenvectors(k, pivot_col);
            eigenvectors(k, pivot_row) = cosine * vip - sine * viq;
            eigenvectors(k, pivot_col) = sine * vip + cosine * viq;
        }
    }

    std::vector<T> eigenvalues(n, T{});
    for (std::size_t index = 0; index < n; ++index) {
        eigenvalues[index] = matrix(index, index);
    }

    return {eigenvectors, eigenvalues};
}

template <typename T>
SVDResult<T> svd_via_normal_equation(const DenseMatrix<T>& matrix) {
    const std::size_t rows = matrix.rows();
    const std::size_t cols = matrix.cols();

    const DenseMatrix<T> at = transpose(matrix);
    DenseMatrix<T> ata = multiply(at, matrix);
    SymmetricEigenResult<T> eigen = jacobi_eigen_symmetric(ata);

    std::vector<std::size_t> order(eigen.eigenvalues.size());
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::sort(order.begin(), order.end(), [&](std::size_t lhs, std::size_t rhs) {
        return eigen.eigenvalues[lhs] > eigen.eigenvalues[rhs];
    });

    DenseMatrix<T> v(cols, cols, T{});
    std::vector<T> singular_values(cols, T{});
    for (std::size_t new_col = 0; new_col < cols; ++new_col) {
        const std::size_t source_col = order[new_col];
        const T eigenvalue = eigen.eigenvalues[source_col];
        singular_values[new_col] = static_cast<T>(std::sqrt(std::max<T>(eigenvalue, T{})));
        for (std::size_t row = 0; row < cols; ++row) {
            v(row, new_col) = eigen.eigenvectors(row, source_col);
        }
    }

    DenseMatrix<T> u(rows, cols, T{});
    for (std::size_t col = 0; col < cols; ++col) {
        const T sigma = singular_values[col];
        if (is_near_zero(sigma)) {
            continue;
        }

        for (std::size_t row = 0; row < rows; ++row) {
            T accumulator = T{};
            for (std::size_t k = 0; k < cols; ++k) {
                accumulator += matrix(row, k) * v(k, col);
            }
            u(row, col) = accumulator / sigma;
        }
    }

    return {u, singular_values, transpose(v)};
}

} // namespace tensor3d

#endif
