#include "storage/GTensorBase.h"
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

GTensorBase::GTensorBase(const TensorMeta& meta)
    : meta_(meta)
    , impl_(std::make_unique<internal::TorchTensorImpl>()) {
    auto options = torch::TensorOptions()
        .dtype(internal::TorchTensorImpl::getTorchDtype(meta.dtype))
        .device(torch::kCUDA);
    impl_->tensor = torch::empty(meta.shape, options);
}

GTensorBase::~GTensorBase() = default;

// 数据访问实现
void* GTensorBase::ptr() {
    return impl_->tensor.data_ptr();
}

const void* GTensorBase::ptr() const {
    return impl_->tensor.data_ptr();
}

size_t GTensorBase::elemCount() const {
    return impl_->tensor.numel();
}

// 设备操作实现
bool GTensorBase::isOnCPU() const {
    return impl_->tensor.device().is_cpu();
}

bool GTensorBase::isOnGPU() const {
    return impl_->tensor.device().is_cuda();
}

void GTensorBase::toCPU() {
    impl_->tensor = impl_->tensor.to(torch::kCPU);
}

void GTensorBase::toGPU() {
    impl_->tensor = impl_->tensor.to(torch::kCUDA);
}

// 数据操作实现
void GTensorBase::zero() {
    impl_->tensor.zero_();
}

void GTensorBase::fill(const void* value) {
    switch (meta_.dtype) {
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

// 形状操作实现
void GTensorBase::reshape(const std::vector<int64_t>& shape) {
    impl_->tensor = impl_->tensor.reshape(shape);
    meta_.shape = shape;
}

void GTensorBase::resize(const std::vector<int64_t>& shape) {
    impl_->tensor.resize_(shape);
    meta_.shape = shape;
}

void GTensorBase::squeeze(int dim) {
    if (dim == -1) {
        impl_->tensor = impl_->tensor.squeeze();
    } else {
        impl_->tensor = impl_->tensor.squeeze(dim);
    }
    auto sizes = impl_->tensor.sizes();
    meta_.shape.assign(sizes.begin(), sizes.end());
}

void GTensorBase::unsqueeze(int dim) {
    impl_->tensor = impl_->tensor.unsqueeze(dim);
    auto sizes = impl_->tensor.sizes();
    meta_.shape.assign(sizes.begin(), sizes.end());
}

void GTensorBase::flatten(int start_dim, int end_dim) {
    impl_->tensor = impl_->tensor.flatten(start_dim, end_dim);
    auto sizes = impl_->tensor.sizes();
    meta_.shape.assign(sizes.begin(), sizes.end());
}

void GTensorBase::transpose(int dim0, int dim1) {
    impl_->tensor = impl_->tensor.transpose(dim0, dim1);
    auto sizes = impl_->tensor.sizes();
    meta_.shape.assign(sizes.begin(), sizes.end());
}

void GTensorBase::permute(const std::vector<int64_t>& dims) {
    impl_->tensor = impl_->tensor.permute(dims);
    auto sizes = impl_->tensor.sizes();
    meta_.shape.assign(sizes.begin(), sizes.end());
}

// 切片和索引实现
std::unique_ptr<ITensor> GTensorBase::slice(
    int dim, 
    int64_t start, 
    int64_t end, 
    int64_t step) const {
    auto sliced = impl_->tensor.slice(dim, start, end, step);
    auto result = std::make_unique<GTensorBase>(
        TensorMeta{
            std::vector<int64_t>(sliced.sizes().begin(), sliced.sizes().end()),
            meta_.dtype,
            meta_.type_size,
            meta_.type_info
        }
    );
    result->impl_->tensor = sliced;
    return result;
}

void GTensorBase::select(int dim, int64_t index) {
    impl_->tensor = impl_->tensor.select(dim, index);
    auto sizes = impl_->tensor.sizes();
    meta_.shape.assign(sizes.begin(), sizes.end());
}

void GTensorBase::index(const std::vector<ITensor*>& indices) {
    std::vector<torch::indexing::TensorIndex> torch_indices;
    torch_indices.reserve(indices.size());
    
    for (auto* idx : indices) {
        if (auto* base = dynamic_cast<const GTensorBase*>(idx)) {
            torch_indices.push_back(base->impl_->tensor);
        } else {
            throw std::runtime_error("Invalid index tensor type");
        }
    }
    
    impl_->tensor = impl_->tensor.index(torch_indices);
    auto sizes = impl_->tensor.sizes();
    meta_.shape.assign(sizes.begin(), sizes.end());
}

// 数学操作实现
void GTensorBase::abs() {
    impl_->tensor = torch::abs(impl_->tensor);
}

void GTensorBase::clip(const void* min_val, const void* max_val) {
    switch (meta_.dtype) {
        case TensorDataType::kFloat32:
            impl_->tensor = torch::clamp(impl_->tensor,
                *static_cast<const float*>(min_val),
                *static_cast<const float*>(max_val));
            break;
        case TensorDataType::kFloat64:
            impl_->tensor = torch::clamp(impl_->tensor,
                *static_cast<const double*>(min_val),
                *static_cast<const double*>(max_val));
            break;
        case TensorDataType::kInt32:
            impl_->tensor = torch::clamp(impl_->tensor,
                *static_cast<const int32_t*>(min_val),
                *static_cast<const int32_t*>(max_val));
            break;
        case TensorDataType::kInt64:
            impl_->tensor = torch::clamp(impl_->tensor,
                *static_cast<const int64_t*>(min_val),
                *static_cast<const int64_t*>(max_val));
            break;
        default:
            throw std::runtime_error("Unsupported data type in clip");
    }
}

void GTensorBase::sqrt() {
    impl_->tensor = torch::sqrt(impl_->tensor);
}

void GTensorBase::pow(double exponent) {
    impl_->tensor = torch::pow(impl_->tensor, exponent);
}

void GTensorBase::exp() {
    impl_->tensor = torch::exp(impl_->tensor);
}

void GTensorBase::log() {
    impl_->tensor = torch::log(impl_->tensor);
}

// 统计操作实现
std::unique_ptr<ITensor> GTensorBase::sum(int dim, bool keepdim) const {
    auto result_tensor = dim == -1 ? 
        torch::sum(impl_->tensor) : 
        torch::sum(impl_->tensor, dim, keepdim);
    
    auto result = std::make_unique<GTensorBase>(
        TensorMeta{
            std::vector<int64_t>(result_tensor.sizes().begin(), result_tensor.sizes().end()),
            meta_.dtype,
            meta_.type_size,
            meta_.type_info
        }
    );
    result->impl_->tensor = result_tensor;
    return result;
}

std::unique_ptr<ITensor> GTensorBase::mean(int dim, bool keepdim) const {
    auto result_tensor = dim == -1 ? 
        torch::mean(impl_->tensor) : 
        torch::mean(impl_->tensor, dim, keepdim);
    
    auto result = std::make_unique<GTensorBase>(
        TensorMeta{
            std::vector<int64_t>(result_tensor.sizes().begin(), result_tensor.sizes().end()),
            meta_.dtype,
            meta_.type_size,
            meta_.type_info
        }
    );
    result->impl_->tensor = result_tensor;
    return result;
}

std::unique_ptr<ITensor> GTensorBase::std(int dim, bool keepdim) const {
    auto result_tensor = dim == -1 ? 
        torch::std(impl_->tensor) : 
        torch::std(impl_->tensor, dim, keepdim);
    
    auto result = std::make_unique<GTensorBase>(
        TensorMeta{
            std::vector<int64_t>(result_tensor.sizes().begin(), result_tensor.sizes().end()),
            meta_.dtype,
            meta_.type_size,
            meta_.type_info
        }
    );
    result->impl_->tensor = result_tensor;
    return result;
}

std::unique_ptr<ITensor> GTensorBase::var(int dim, bool keepdim) const {
    auto result_tensor = dim == -1 ? 
        torch::var(impl_->tensor) : 
        torch::var(impl_->tensor, dim, keepdim);
    
    auto result = std::make_unique<GTensorBase>(
        TensorMeta{
            std::vector<int64_t>(result_tensor.sizes().begin(), result_tensor.sizes().end()),
            meta_.dtype,
            meta_.type_size,
            meta_.type_info
        }
    );
    result->impl_->tensor = result_tensor;
    return result;
}

std::unique_ptr<ITensor> GTensorBase::max(int dim, bool keepdim) const {
    torch::Tensor result_tensor;
    
    if (dim == -1) {
        // 全局最小值
        result_tensor = torch::max(impl_->tensor);
    } else {
        // 按维度求最小值
        auto result = torch::max(impl_->tensor, dim, keepdim);
        result_tensor = std::get<0>(result);  // 获取最小值，忽略索引
    }
    
    auto result = std::make_unique<GTensorBase>(
        TensorMeta{
            std::vector<int64_t>(result_tensor.sizes().begin(), result_tensor.sizes().end()),
            meta_.dtype,
            meta_.type_size,
            meta_.type_info
        }
    );
    result->impl_->tensor = result_tensor;
    return result;
}

std::unique_ptr<ITensor> GTensorBase::min(int dim, bool keepdim) const {
    torch::Tensor result_tensor;
    
    if (dim == -1) {
        // 全局最小值
        result_tensor = torch::min(impl_->tensor);
    } else {
        // 按维度求最小值
        auto result = torch::min(impl_->tensor, dim, keepdim);
        result_tensor = std::get<0>(result);  // 获取最小值，忽略索引
    }
    
    auto result = std::make_unique<GTensorBase>(
        TensorMeta{
            std::vector<int64_t>(result_tensor.sizes().begin(), result_tensor.sizes().end()),
            meta_.dtype,
            meta_.type_size,
            meta_.type_info
        }
    );
    result->impl_->tensor = result_tensor;
    return result;
}

std::unique_ptr<ITensor> GTensorBase::argmax(int dim, bool keepdim) const {
    auto result_tensor = dim == -1 ? 
        torch::argmax(impl_->tensor) : 
        torch::argmax(impl_->tensor, dim, keepdim);
    
    auto result = std::make_unique<GTensorBase>(
        TensorMeta{
            std::vector<int64_t>(result_tensor.sizes().begin(), result_tensor.sizes().end()),
            TensorDataType::kInt64,
            sizeof(int64_t),
            std::type_index(typeid(int64_t))
        }
    );
    result->impl_->tensor = result_tensor;
    return result;
}

std::unique_ptr<ITensor> GTensorBase::argmin(int dim, bool keepdim) const {
    auto result_tensor = dim == -1 ? 
        torch::argmin(impl_->tensor) : 
        torch::argmin(impl_->tensor, dim, keepdim);
    
    auto result = std::make_unique<GTensorBase>(
        TensorMeta{
            std::vector<int64_t>(result_tensor.sizes().begin(), result_tensor.sizes().end()),
            TensorDataType::kInt64,
            sizeof(int64_t),
            std::type_index(typeid(int64_t))
        }
    );
    result->impl_->tensor = result_tensor;
    return result;
}

// 比较操作实现
void GTensorBase::eq(const ITensor& other) {
    if (auto* base = dynamic_cast<const GTensorBase*>(&other)) {
        impl_->tensor = impl_->tensor.eq(base->impl_->tensor);
    } else {
        throw std::runtime_error("Invalid tensor type in eq");
    }
}

void GTensorBase::ne(const ITensor& other) {
    if (auto* base = dynamic_cast<const GTensorBase*>(&other)) {
        impl_->tensor = impl_->tensor.ne(base->impl_->tensor);
    } else {
        throw std::runtime_error("Invalid tensor type in ne");
    }
}

void GTensorBase::gt(const ITensor& other) {
    if (auto* base = dynamic_cast<const GTensorBase*>(&other)) {
        impl_->tensor = impl_->tensor.gt(base->impl_->tensor);
    } else {
        throw std::runtime_error("Invalid tensor type in gt");
    }
}

void GTensorBase::lt(const ITensor& other) {
    if (auto* base = dynamic_cast<const GTensorBase*>(&other)) {
        impl_->tensor = impl_->tensor.lt(base->impl_->tensor);
    } else {
        throw std::runtime_error("Invalid tensor type in lt");
    }
}

void GTensorBase::ge(const ITensor& other) {
    if (auto* base = dynamic_cast<const GTensorBase*>(&other)) {
        impl_->tensor = impl_->tensor.ge(base->impl_->tensor);
    } else {
        throw std::runtime_error("Invalid tensor type in ge");
    }
}

void GTensorBase::le(const ITensor& other) {
    if (auto* base = dynamic_cast<const GTensorBase*>(&other)) {
        impl_->tensor = impl_->tensor.le(base->impl_->tensor);
    } else {
        throw std::runtime_error("Invalid tensor type in le");
    }
}

// 辅助函数实现
GTensorBase* GTensorBase::createTensorFromImpl(
    const std::string& name,
    internal::TorchTensorImpl* impl) const {
    auto tensor = std::make_unique<GTensorBase>(
        TensorMeta{
            std::vector<int64_t>(impl->tensor.sizes().begin(), impl->tensor.sizes().end()),
            meta_.dtype,
            meta_.type_size,
            meta_.type_info
        }
    );
    tensor->impl_->tensor = impl->tensor;
    return tensor.release();
}

const internal::TorchTensorImpl* GTensorBase::getImpl(const ITensor& tensor) const {
    if (auto* base = dynamic_cast<const GTensorBase*>(&tensor)) {
        return base->impl_.get();
    }
    throw std::runtime_error("Invalid tensor type");
}

} // namespace RSG_SIM