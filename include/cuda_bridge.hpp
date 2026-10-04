#ifndef TENSOR3D_CUDA_BRIDGE_HPP
#define TENSOR3D_CUDA_BRIDGE_HPP

#include "sparse_ops.hpp"

namespace tensor3d {

template <typename T>
Tensor3D<T> hadamard_product_cuda(const Tensor3D<T>& left, const Tensor3D<T>& right, RunMetrics* metrics = nullptr);

template <typename T>
std::vector<QRResult<T>> tensor_qr_cuda(const Tensor3D<T>& tensor);

template <typename T>
std::vector<SVDResult<T>> tensor_svd_cuda(const Tensor3D<T>& tensor);

#ifndef TENSOR3D_USE_CUDA

template <typename T>
Tensor3D<T> hadamard_product_cuda(const Tensor3D<T>& left, const Tensor3D<T>& right, RunMetrics* metrics) {
    return hadamard_product_serial(left, right, metrics);
}

template <typename T>
std::vector<QRResult<T>> tensor_qr_cuda(const Tensor3D<T>& tensor) {
    return tensor_qr_serial(tensor);
}

template <typename T>
std::vector<SVDResult<T>> tensor_svd_cuda(const Tensor3D<T>& tensor) {
    return tensor_svd_serial(tensor);
}

#endif

} // namespace tensor3d

#endif
