#include <cstdint>
#include <memory>
#include <torch/torch.h>
#include <stdexcept>
#include "core/storage/GTensorTorchWrapper.hh"

namespace cuda_simulator
{
namespace core
{
namespace internal
{
    class TorchTensorImpl : public std::enable_shared_from_this<TorchTensorImpl>
    {
    public:
        torch::Tensor tensor;

        static torch::Dtype getTorchDtype(TensorDataType dtype)
        {
            switch (dtype)
            {
            case TensorDataType::kFloat32:
                return torch::kFloat32;
            case TensorDataType::kFloat64:
                return torch::kFloat64;
            case TensorDataType::kInt32:
                return torch::kInt32;
            case TensorDataType::kInt64:
                return torch::kInt64;
            default:
                throw std::runtime_error("Unsupported data type");
            }
        }

        static TensorDataType getDataType(torch::Dtype dtype)
        {
            if (dtype == torch::kFloat32)
                return TensorDataType::kFloat32;
            if (dtype == torch::kFloat64)
                return TensorDataType::kFloat64;
            if (dtype == torch::kInt32)
                return TensorDataType::kInt32;
            if (dtype == torch::kInt64)
                return TensorDataType::kInt64;
            throw std::runtime_error("Unsupported torch dtype");
        }
    };

    std::shared_ptr<TorchTensorImpl> shareTorchTensorImpl(const std::shared_ptr<TorchTensorImpl> &impl)
    {
        return impl->shared_from_this();
    }
}

// GTensorTorchWrapper::GTensorTorchWrapper()
//     : impl_(std::make_shared<internal::TorchTensorImpl>()), dtype_(TensorDataType::kFloat32)
// {
// }

GTensorTorchWrapper::GTensorTorchWrapper(const std::vector<int64_t> &shape, TensorDataType dtype)
    : impl_(std::make_shared<internal::TorchTensorImpl>()), dtype_(dtype)
{
    auto options = torch::TensorOptions()
                        .dtype(internal::TorchTensorImpl::getTorchDtype(dtype))
                        .device(torch::kCUDA);
    impl_->tensor = torch::empty(shape, options);
}


GTensorTorchWrapper::~GTensorTorchWrapper() = default;

// 数据访问实现
// 打印实现
void GTensorTorchWrapper::print(std::ostream &out) const
{
    out << impl_->tensor;
}

std::string GTensorTorchWrapper::toString() const
{
    std::ostringstream oss;
    print(oss);
    return oss.str();
}

std::vector<int64_t> GTensorTorchWrapper::shape() const
{
    std::vector<int64_t> ret_shape(impl_->tensor.sizes().begin(), impl_->tensor.sizes().end());
    return ret_shape;
}

size_t GTensorTorchWrapper::elemCount() const
{
    return impl_->tensor.numel();
}

size_t GTensorTorchWrapper::elemSize() const
{
    return impl_->tensor.element_size();
}

size_t GTensorTorchWrapper::dim() const
{
    std::cout << impl_->tensor;
    return impl_->tensor.dim();
}

void GTensorTorchWrapper::zero()
{
    impl_->tensor.zero_();
}

GTensorTorchWrapper GTensorTorchWrapper::add_impl(const GTensorTorchWrapper& other) const {
    GTensorTorchWrapper result(shape(), dtype());
    result.impl_->tensor = impl_->tensor + other.impl_->tensor;
    return result;
}

GTensorTorchWrapper GTensorTorchWrapper::sub_impl(const GTensorTorchWrapper& other) const {
    GTensorTorchWrapper result(shape(), dtype());
    result.impl_->tensor = impl_->tensor - other.impl_->tensor;
    return result;
}

GTensorTorchWrapper GTensorTorchWrapper::mul_impl(const GTensorTorchWrapper& other) const {
    GTensorTorchWrapper result(shape(), dtype());
    result.impl_->tensor = impl_->tensor * other.impl_->tensor;
    return result;
}

GTensorTorchWrapper GTensorTorchWrapper::div_impl(const GTensorTorchWrapper& other) const {
    GTensorTorchWrapper result(shape(), dtype());
    result.impl_->tensor = impl_->tensor / other.impl_->tensor;
    return result;
}

GTensorTorchWrapper GTensorTorchWrapper::slice_impl(int64_t dim, int64_t start, int64_t end) const {
    GTensorTorchWrapper result(shape(), dtype());
    result.impl_->tensor = impl_->tensor.slice(dim, start, end);
    return result;
}

GTensorTorchWrapper GTensorTorchWrapper::index_impl(const std::vector<int64_t>& indices) const {
    torch::Tensor indexed = impl_->tensor;
    // 依次对每个维度进行索引
    for (size_t i = 0; i < indices.size(); ++i) {
        indexed = indexed.select(i, indices[i]);
    }

    GTensorTorchWrapper result(indexed.sizes().vec(), dtype_);
    result.impl_->tensor = indexed;
    return result;
}

GTensorTorchWrapper GTensorTorchWrapper::index_impl(int64_t index) const {
    GTensorTorchWrapper result(shape(), dtype());
    result.impl_->tensor = impl_->tensor.select(0, index);
    return result;
}

GTensorTorchWrapper GTensorTorchWrapper::bitwise_not_impl() const {
    GTensorTorchWrapper result(shape(), dtype());
    result.impl_->tensor = torch::bitwise_not(impl_->tensor);
    return result;
}

GTensorTorchWrapper GTensorTorchWrapper::neg_impl() const {
    GTensorTorchWrapper result(shape(), dtype());
    result.impl_->tensor = impl_->tensor.neg();
    return result;
}

GTensorTorchWrapper& GTensorTorchWrapper::add_inplace_impl(const GTensorTorchWrapper& other) {
    impl_->tensor.add_(other.impl_->tensor);
    return *this;
}

GTensorTorchWrapper& GTensorTorchWrapper::sub_inplace_impl(const GTensorTorchWrapper& other) {
    impl_->tensor.sub_(other.impl_->tensor);
    return *this;
}

GTensorTorchWrapper& GTensorTorchWrapper::mul_inplace_impl(const GTensorTorchWrapper& other) {
    impl_->tensor.mul_(other.impl_->tensor);
    return *this;
}

GTensorTorchWrapper& GTensorTorchWrapper::div_inplace_impl(const GTensorTorchWrapper& other) {
    impl_->tensor.div_(other.impl_->tensor);
    return *this;
}

GTensorTorchWrapper& GTensorTorchWrapper::bitwise_and_inplace_impl(const GTensorTorchWrapper& other) {
    impl_->tensor.bitwise_and_(other.impl_->tensor);
    return *this;
}

GTensorTorchWrapper& GTensorTorchWrapper::bitwise_or_inplace_impl(const GTensorTorchWrapper& other) {
    impl_->tensor.bitwise_or_(other.impl_->tensor);
    return *this;
}

GTensorTorchWrapper GTensorTorchWrapper::clone() const {
    return GTensorTorchWrapper(*this);
}

GTensorTorchWrapper GTensorTorchWrapper::move() {
    return GTensorTorchWrapper(std::move(*this));
}

float GTensorTorchWrapper::item_float_impl() const {
    return impl_->tensor.item<float>();
}

double GTensorTorchWrapper::item_double_impl() const {
    return impl_->tensor.item<double>();
}

int64_t GTensorTorchWrapper::item_int64_impl() const {
    return impl_->tensor.item<int64_t>();
}

int32_t GTensorTorchWrapper::item_int32_impl() const {
    return impl_->tensor.item<int32_t>();
}

TensorDataType GTensorTorchWrapper::dtype() const {
    return dtype_;
}

void* GTensorTorchWrapper::data() {
    return impl_->tensor.data_ptr();
}

const void* GTensorTorchWrapper::data() const {
    return impl_->tensor.data_ptr();
}

} // namespace core
} // namespace cuda_simulator