#include "storage/GTensorTorchWrapper.h"
#include <torch/torch.h>
#include <stdexcept>

namespace RSG_SIM {

namespace internal {
class TorchTensorImpl {
public:
    torch::Tensor tensor;
    
    static torch::Dtype getTorchDtype(TensorDataType dtype) {
        switch (dtype) {
            case TensorDataType::kFloat32: return torch::kFloat32;
            case TensorDataType::kFloat64: return torch::kFloat64;
            case TensorDataType::kInt32: return torch::kInt32;
            case TensorDataType::kInt64: return torch::kInt64;
            default: throw std::runtime_error("Unsupported data type");
        }
    }
    
    static TensorDataType getDataType(torch::Dtype dtype) {
        if (dtype == torch::kFloat32) return TensorDataType::kFloat32;
        if (dtype == torch::kFloat64) return TensorDataType::kFloat64;
        if (dtype == torch::kInt32) return TensorDataType::kInt32;
        if (dtype == torch::kInt64) return TensorDataType::kInt64;
        throw std::runtime_error("Unsupported torch dtype");
    }
};
}

GTensorTorchWrapper::GTensorTorchWrapper(const std::vector<int64_t> &shape, TensorDataType dtype)
    : impl_(std::make_unique<internal::TorchTensorImpl>())
{
    auto options = torch::TensorOptions()
        .dtype(internal::TorchTensorImpl::getTorchDtype(dtype))
        .device(torch::kCUDA);
    impl_->tensor = torch::empty(shape, options);
}

GTensorTorchWrapper::~GTensorTorchWrapper() = default;

// 数据访问实现
void* GTensorTorchWrapper::ptr() {
    return impl_->tensor.data_ptr();
}

const void* GTensorTorchWrapper::ptr() const {
    return impl_->tensor.data_ptr();
}

std::vector<int64_t> GTensorTorchWrapper::shape() const {
    std::vector<int64_t> ret_shape(impl_->tensor.sizes().begin(), impl_->tensor.sizes().end());
    return ret_shape;
}

size_t GTensorTorchWrapper::elemCount() const {
    return impl_->tensor.numel();
}

size_t GTensorTorchWrapper::elemSize() const {
    return impl_->tensor.element_size();
}

size_t GTensorTorchWrapper::dim() const {
    return impl_->tensor.dim();
}

TensorDataType GTensorTorchWrapper::getTensorDataType() const
{
    return internal::TorchTensorImpl::getDataType(impl_->tensor.scalar_type());
}

void GTensorTorchWrapper::zero() {
    impl_->tensor.zero_();
}

void GTensorTorchWrapper::fill(const void* value, TensorDataType dtype) {
    switch (dtype) {
        case TensorDataType::kFloat32:
            impl_->tensor.fill_(*static_cast<const float*>(value));
            break;
        case TensorDataType::kFloat64:
            impl_->tensor.fill_(*static_cast<const double*>(value));
            break;
        case TensorDataType::kInt32:
            impl_->tensor.fill_(*static_cast<const int32_t*>(value));
            break;
        case TensorDataType::kInt64:
            impl_->tensor.fill_(*static_cast<const int64_t*>(value));
            break;
        default:
            throw std::runtime_error("Unsupported data type in fill");
    }
}

} // namespace RSG_SIM