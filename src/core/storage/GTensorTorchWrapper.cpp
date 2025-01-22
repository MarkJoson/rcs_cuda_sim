#include "core/storage/GTensorTorchWrapper.hh"
#include "core/storage/ITensor.hh"
#include "core/storage/Scalar.hh"
#include <ATen/core/TensorBody.h>
#include <c10/core/TensorOptions.h>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <torch/torch.h>

namespace cuda_simulator {
namespace core {
namespace internal {
class TorchTensorImpl {
public:
  torch::Tensor tensor;
  // NumericalDataType dtype;
  // DeviceType device_type;

  TorchTensorImpl() = default;

  TorchTensorImpl(torch::Tensor t) : tensor(t) { }

  NumericalDataType dtype() const {
    return TorchTensorImpl::getDataType(tensor.dtype());
  }

  DeviceType device() const {
    return TorchTensorImpl::getDeviceType(tensor.device());
  }

  static caffe2::TypeMeta getTorchDtype(NumericalDataType dtype) {
    switch (dtype) {
    case NumericalDataType::kFloat32:
      return c10::scalarTypeToTypeMeta(torch::kFloat32);
    case NumericalDataType::kFloat64:
      return c10::scalarTypeToTypeMeta(torch::kFloat64);
    case NumericalDataType::kInt32:
      return c10::scalarTypeToTypeMeta(torch::kInt32);
    case NumericalDataType::kInt64:
      return c10::scalarTypeToTypeMeta(torch::kInt64);
    case NumericalDataType::kUInt8:
      return c10::scalarTypeToTypeMeta(torch::kUInt8);
    case NumericalDataType::kUInt32:
      return c10::scalarTypeToTypeMeta(torch::kUInt32);
    default:
      throw std::runtime_error("Unsupported data type");
    }
  }

  static NumericalDataType getDataType(caffe2::TypeMeta dtype) {
    if (c10::typeMetaToScalarType(dtype) == torch::kFloat32)
      return NumericalDataType::kFloat32;
    if (c10::typeMetaToScalarType(dtype) == torch::kFloat64)
      return NumericalDataType::kFloat64;
    if (c10::typeMetaToScalarType(dtype) == torch::kInt32)
      return NumericalDataType::kInt32;
    if (c10::typeMetaToScalarType(dtype) == torch::kInt64)
      return NumericalDataType::kInt64;
    if (c10::typeMetaToScalarType(dtype) == torch::kUInt8)
      return NumericalDataType::kUInt8;
    if (c10::typeMetaToScalarType(dtype) == torch::kUInt32)
      return NumericalDataType::kUInt32;
    throw std::runtime_error("Unsupported torch dtype");
  }

  static DeviceType getDeviceType(const torch::Device &device) {
    if (device.is_cuda())
      return DeviceType::kCUDA;
    else if (device.is_cpu())
      return DeviceType::kCPU;
    else
      throw std::runtime_error("Unsupported device type");
  }

  static torch::DeviceType getTorchDevice(DeviceType device_type) {
    if (device_type == DeviceType::kCPU)
      return torch::kCPU;
    else if (device_type == DeviceType::kCUDA)
      return torch::kCUDA;
    else
      throw std::runtime_error("Unsupported device type");
  }
};

torch::Scalar toTorchScalar(const Scalar &s) {
  switch (s.type()) {
  case Scalar::Type::kFloat32:
    return torch::Scalar(s.toFloat());
  case Scalar::Type::kFloat64:
    return torch::Scalar(s.toDouble());
  case Scalar::Type::kInt32:
    return torch::Scalar(s.toInt32());
  case Scalar::Type::kInt64:
    return torch::Scalar(s.toInt64());
  case Scalar::Type::kUInt8:
    return torch::Scalar(s.toUInt8());
  default:
    throw std::runtime_error("Unsupported scalar type");
  }
}

Scalar toScalar(const torch::Scalar &s) {
  if (s.isFloatingPoint()) {
    return Scalar(s.toFloat());
  } else {
    return Scalar(s.toInt());
  }
  throw std::runtime_error("Unsupported torch scalar type");
}
} // namespace internal

static int8_t g_default_cuda_id = -1; // -1 means current cuda device

// &---------------------  创建  ---------------------
GTensorTorchWrapper::GTensorTorchWrapper(const TensorShape &shape, NumericalDataType dtype,
                                         DeviceType device_type)
    : impl_(std::make_unique<internal::TorchTensorImpl>()) {
  if (shape.size() != 0) {
    int8_t device_id = (device_type == DeviceType::kCPU)
                           ? 0
                           : g_default_cuda_id; // CPU设备ID为0，CUDA设备ID为g_default_cuda_id

    auto options = torch::TensorOptions()
                       .dtype(internal::TorchTensorImpl::getTorchDtype(dtype))
                       .device(internal::TorchTensorImpl::getTorchDevice(device_type), device_id);
    impl_->tensor = torch::zeros(shape, options);
  }
}

GTensorTorchWrapper::GTensorTorchWrapper(const Scalar &scalar, DeviceType device_type)
    : impl_(std::make_unique<internal::TorchTensorImpl>()) {
  int8_t device_id = (device_type == DeviceType::kCPU)
                         ? 0
                         : g_default_cuda_id; // CPU设备ID为0，CUDA设备ID为g_default_cuda_id

  auto options = torch::TensorOptions()
                     .dtype(internal::TorchTensorImpl::getTorchDtype(scalar.type()))
                     .device(internal::TorchTensorImpl::getTorchDevice(device_type), device_id);
  impl_->tensor = torch::full({}, internal::toTorchScalar(scalar), options);
}

GTensorTorchWrapper::~GTensorTorchWrapper() {
}

// &---------------------  拷贝构造函数  ---------------------
GTensorTorchWrapper::GTensorTorchWrapper(const GTensorTorchWrapper &other) noexcept {
  impl_ = std::make_unique<internal::TorchTensorImpl>();
  impl_->tensor = other.impl_->tensor;
}

// &---------------------  移动构造函数  ---------------------
GTensorTorchWrapper::GTensorTorchWrapper(GTensorTorchWrapper &&other) noexcept {
  impl_ = std::move(other.impl_);
}

// &---------------------  赋值, 替换tensor  ---------------------
GTensorTorchWrapper &GTensorTorchWrapper::operator=(const GTensorTorchWrapper &other) noexcept {
  if (this != &other) {
    impl_->tensor = other.impl_->tensor;
  }
  return *this;
}

// &---------------------  移动赋值, 替换impl  ---------------------
GTensorTorchWrapper &GTensorTorchWrapper::operator=(GTensorTorchWrapper &&other) noexcept {
  if (this != &other) {
    impl_ = std::move(other.impl_);
  }
  return *this;
}

// &---------------------  Scalar方法实现  ---------------------
GTensorTorchWrapper &GTensorTorchWrapper::from_scalar_impl(const Scalar &scalar) {
  int8_t device_id = (impl_->device() == DeviceType::kCPU) ? 0 : g_default_cuda_id;
  impl_->tensor = torch::scalar_tensor(
      internal::toTorchScalar(scalar),
      torch::TensorOptions()
          .dtype(internal::TorchTensorImpl::getTorchDtype(impl_->dtype()))
          .device(internal::TorchTensorImpl::getTorchDevice(impl_->device()), device_id));
  return *this;
}

Scalar GTensorTorchWrapper::to_scalar_impl() const {
  Scalar scalar(0);
  switch (impl_->dtype()) {
  case NumericalDataType::kFloat32:
    scalar = Scalar(impl_->tensor.item<float>());
    break;
  case NumericalDataType::kFloat64:
    scalar = Scalar(impl_->tensor.item<double>());
    break;
  case NumericalDataType::kInt32:
    scalar = Scalar(impl_->tensor.item<int32_t>());
    break;
  case NumericalDataType::kInt64:
    scalar = Scalar(impl_->tensor.item<int64_t>());
    break;
  case NumericalDataType::kUInt8:
    scalar = Scalar(impl_->tensor.item<uint8_t>());
    break;
  // case NumericalDataType::kUInt32:
  //     scalar = Scalar(impl_->tensor.item<uint32_t>());
  //     break;
  default:
    throw std::runtime_error("Unsupported data type");
  }
  return scalar;
}

// &---------------------  从主机数组创建张量  ---------------------
GTensorTorchWrapper GTensorTorchWrapper::from_host_array_impl(const void *data, int64_t numel, NumericalDataType type, DeviceType device) {
  auto new_tensor = torch::from_blob(const_cast<void *>(data), {numel},
                                     internal::TorchTensorImpl::getTorchDtype(type));

  new_tensor = new_tensor.to(
      torch::TensorOptions()
          .device(internal::TorchTensorImpl::getTorchDevice(device), g_default_cuda_id)
          .dtype(internal::TorchTensorImpl::getTorchDtype(type)));

  GTensorTorchWrapper result;
  result.impl_->tensor = new_tensor.clone();
  return result;
}

void GTensorTorchWrapper::to_host_array_impl(void *data, NumericalDataType type, int64_t numel) const {
  auto new_tensor = impl_->tensor.to(torch::TensorOptions()
                                         .device(torch::kCPU)
                                         .dtype(internal::TorchTensorImpl::getTorchDtype(type)));

  new_tensor = new_tensor.reshape({numel});
  std::memcpy(data, new_tensor.data_ptr(), numel * new_tensor.element_size());
}

// &---------------------  模板创建  ---------------------
GTensorTorchWrapper GTensorTorchWrapper::zeros_impl(const TensorShape &shape,
                                                   NumericalDataType dtype,
                                                   DeviceType device_type) {
  GTensorTorchWrapper result;
  result.impl_->tensor =
      torch::zeros(shape, torch::TensorOptions()
                              .dtype(internal::TorchTensorImpl::getTorchDtype(dtype))
                              .device(internal::TorchTensorImpl::getTorchDevice(device_type),
                                      g_default_cuda_id));
  return result;
}

GTensorTorchWrapper GTensorTorchWrapper::rands_impl(const TensorShape &shape,
                                                   NumericalDataType dtype,
                                                   DeviceType device_type) {
  GTensorTorchWrapper result;
  result.impl_->tensor =
      torch::rand(shape, torch::TensorOptions()
                             .dtype(internal::TorchTensorImpl::getTorchDtype(dtype))
                             .device(internal::TorchTensorImpl::getTorchDevice(device_type),
                                     g_default_cuda_id));
  return result;
}


// &---------------------  属性  ---------------------
TensorShape GTensorTorchWrapper::shape_impl() const {
  TensorShape ret_shape(impl_->tensor.sizes().begin(), impl_->tensor.sizes().end());
  return ret_shape;
}

DeviceType GTensorTorchWrapper::device_impl() const {
  return impl_->device();
}

size_t GTensorTorchWrapper::elem_count_impl() const {
  return impl_->tensor.numel();
}

size_t GTensorTorchWrapper::elem_size_impl() const {
  return impl_->tensor.element_size();
}

size_t GTensorTorchWrapper::dim_impl() const {
  std::cout << impl_->tensor;
  return impl_->tensor.dim();
}

NumericalDataType GTensorTorchWrapper::dtype_impl() const {
  return impl_->dtype();
}

bool GTensorTorchWrapper::is_contiguous_impl() const {
  return impl_->tensor.is_contiguous();
}

void *GTensorTorchWrapper::data_impl() {
  return impl_->tensor.data_ptr();
}

const void *GTensorTorchWrapper::data_impl() const {
  return impl_->tensor.data_ptr();
}

// &---------------------  内部tensor操作  ---------------------
void GTensorTorchWrapper::copy_from_impl(const GTensorTorchWrapper &other) {
  impl_->tensor.copy_(other.impl_->tensor);
}

void GTensorTorchWrapper::copy_to_impl(GTensorTorchWrapper &other) const {
  other.impl_->tensor.copy_(impl_->tensor);
}

GTensorTorchWrapper GTensorTorchWrapper::clone_impl() const {
  GTensorTorchWrapper result;
  result.impl_->tensor = impl_->tensor.clone();
  return result;
}

void GTensorTorchWrapper::move_from_impl(GTensorTorchWrapper &&other) {
  impl_->tensor = std::move(other.impl_->tensor);
}

// &---------------------  打印调试  ---------------------
void GTensorTorchWrapper::print_impl(std::ostream &out) const {
  out << impl_->tensor;
}

std::string GTensorTorchWrapper::to_string_impl() const {
  std::ostringstream oss;
  print(oss);
  return oss.str();
}

// &---------------------  填充  ---------------------
void GTensorTorchWrapper::zero_impl() {
  impl_->tensor.zero_();
}

void GTensorTorchWrapper::fill_impl(Scalar value) {
  impl_->tensor.fill_(internal::toTorchScalar(value));
}

// &---------------------  batch操作  ---------------------
void GTensorTorchWrapper::gather_sum_impl(const std::vector<const GTensorTorchWrapper*> src) {
  std::vector<torch::Tensor> tensors;
  tensors.reserve(src.size());

  // 收集所有源张量
  for (auto &tensor : src) {
    tensors.push_back(tensor->impl_->tensor);
  }

  // 使用stack在新维度上堆叠所有张量，然后在该维度上求和
  auto stacked = torch::stack(tensors, 0);
  impl_->tensor = torch::sum(stacked, 0);
}

void GTensorTorchWrapper::gather_mean_impl(const std::vector<const GTensorTorchWrapper*> src) {
  std::vector<torch::Tensor> tensors;
  tensors.reserve(src.size());

  for (auto &tensor : src) {
    tensors.push_back(tensor->impl_->tensor);
  }

  auto stacked = torch::stack(tensors, 0);
  impl_->tensor = torch::mean(stacked, 0);
}

void GTensorTorchWrapper::gather_max_impl(const std::vector<const GTensorTorchWrapper*> src) {
  std::vector<torch::Tensor> tensors;
  tensors.reserve(src.size());

  for (auto &tensor : src) {
    tensors.push_back(tensor->impl_->tensor);
  }

  auto stacked = torch::stack(tensors, 0);
  // max 返回两个值：最大值和最大值的索引，我们只需要最大值
  auto [values, indices] = torch::max(stacked, 0);
  impl_->tensor = values;
}

void GTensorTorchWrapper::gather_min_impl(const std::vector<const GTensorTorchWrapper*> src) {
  std::vector<torch::Tensor> tensors;
  tensors.reserve(src.size());

  for (auto &tensor : src) {
    tensors.push_back(tensor->impl_->tensor);
  }

  auto stacked = torch::stack(tensors, 0);
  // min 返回两个值：最小值和最小值的索引，我们只需要最小值
  auto [values, indices] = torch::min(stacked, 0);
  impl_->tensor = values;
}

// &---------------------  四则运算/逻辑运算  ---------------------
GTensorTorchWrapper GTensorTorchWrapper::add_impl(const GTensorTorchWrapper &other) const {
  GTensorTorchWrapper result;
  result.impl_->tensor = impl_->tensor + other.impl_->tensor;
  return result;
}

GTensorTorchWrapper GTensorTorchWrapper::sub_impl(const GTensorTorchWrapper &other) const {
  GTensorTorchWrapper result;
  result.impl_->tensor = impl_->tensor - other.impl_->tensor;
  return result;
}

GTensorTorchWrapper GTensorTorchWrapper::mul_impl(const GTensorTorchWrapper &other) const {
  GTensorTorchWrapper result;
  result.impl_->tensor = impl_->tensor * other.impl_->tensor;
  return result;
}

GTensorTorchWrapper GTensorTorchWrapper::div_impl(const GTensorTorchWrapper &other) const {
  GTensorTorchWrapper result;
  result.impl_->tensor = impl_->tensor / other.impl_->tensor;
  return result;
}

GTensorTorchWrapper GTensorTorchWrapper::matmul_impl(const GTensorTorchWrapper &a,
                                                     const GTensorTorchWrapper &b) {
  GTensorTorchWrapper result;
  result.impl_->tensor = torch::matmul(a.impl_->tensor, b.impl_->tensor);
  return result;
}

GTensorTorchWrapper GTensorTorchWrapper::bitwise_not_impl() const {
  GTensorTorchWrapper result;
  result.impl_->tensor = torch::bitwise_not(impl_->tensor);
  return result;
}

GTensorTorchWrapper GTensorTorchWrapper::neg_impl() const {
  GTensorTorchWrapper result;
  result.impl_->tensor = impl_->tensor.neg();
  return result;
}

GTensorTorchWrapper &GTensorTorchWrapper::add_inplace_impl(const GTensorTorchWrapper &other) {
  impl_->tensor.add_(other.impl_->tensor);
  return *this;
}

GTensorTorchWrapper &GTensorTorchWrapper::sub_inplace_impl(const GTensorTorchWrapper &other) {
  impl_->tensor.sub_(other.impl_->tensor);
  return *this;
}

GTensorTorchWrapper &GTensorTorchWrapper::mul_inplace_impl(const GTensorTorchWrapper &other) {
  impl_->tensor.mul_(other.impl_->tensor);
  return *this;
}

GTensorTorchWrapper &GTensorTorchWrapper::div_inplace_impl(const GTensorTorchWrapper &other) {
  impl_->tensor.div_(other.impl_->tensor);
  return *this;
}

GTensorTorchWrapper &GTensorTorchWrapper::bitwise_and_inplace_impl(
    const GTensorTorchWrapper &other) {
  impl_->tensor.bitwise_and_(other.impl_->tensor);
  return *this;
}

GTensorTorchWrapper &GTensorTorchWrapper::bitwise_or_inplace_impl(
    const GTensorTorchWrapper &other) {
  impl_->tensor.bitwise_or_(other.impl_->tensor);
  return *this;
}

// &---------------------  Scalar运算  ---------------------
GTensorTorchWrapper GTensorTorchWrapper::add_scalar_impl(const Scalar &scalar) const {
  GTensorTorchWrapper result;
  result.impl_->tensor = impl_->tensor + (internal::toTorchScalar(scalar));
  return result;
}

GTensorTorchWrapper GTensorTorchWrapper::sub_scalar_impl(const Scalar &scalar) const {
  GTensorTorchWrapper result;
  result.impl_->tensor = impl_->tensor - (internal::toTorchScalar(scalar));
  return result;
}

GTensorTorchWrapper GTensorTorchWrapper::mul_scalar_impl(const Scalar &scalar) const {
  GTensorTorchWrapper result;
  result.impl_->tensor = impl_->tensor * (internal::toTorchScalar(scalar));
  return result;
}

GTensorTorchWrapper GTensorTorchWrapper::div_scalar_impl(const Scalar &scalar) const {
  GTensorTorchWrapper result;
  result.impl_->tensor = impl_->tensor / (internal::toTorchScalar(scalar));
  ;
  return result;
}

GTensorTorchWrapper &GTensorTorchWrapper::add_inplace_scalar_impl(const Scalar &scalar) {
  impl_->tensor.add_(internal::toTorchScalar(scalar));
  return *this;
}

GTensorTorchWrapper &GTensorTorchWrapper::sub_inplace_scalar_impl(const Scalar &scalar) {
  impl_->tensor.sub_(internal::toTorchScalar(scalar));
  return *this;
}

GTensorTorchWrapper &GTensorTorchWrapper::mul_inplace_scalar_impl(const Scalar &scalar) {
  impl_->tensor.mul_(internal::toTorchScalar(scalar));
  return *this;
}

GTensorTorchWrapper &GTensorTorchWrapper::div_inplace_scalar_impl(const Scalar &scalar) {
  impl_->tensor.div_(internal::toTorchScalar(scalar));
  return *this;
}

// &---------------------  索引操作  ---------------------
GTensorTorchWrapper GTensorTorchWrapper::slice_impl(int64_t dim, int64_t start, int64_t end) const {
  GTensorTorchWrapper result;
  result.impl_->tensor = impl_->tensor.slice(dim, start, end);
  return result;
}

GTensorTorchWrapper GTensorTorchWrapper::index_impl(const TensorShape &indices) const {
  GTensorTorchWrapper result;
  std::vector<torch::indexing::TensorIndex> indices_torch(indices.begin(), indices.end());
  result.impl_->tensor = impl_->tensor.index(indices_torch);
  return result;
}

GTensorTorchWrapper GTensorTorchWrapper::index_impl(int64_t index) const {
  GTensorTorchWrapper result;
  result.impl_->tensor = impl_->tensor.select(0, index);
  return result;
}

// &---------------------  沿axis的操作  ---------------------
GTensorTorchWrapper GTensorTorchWrapper::sum_impl(int64_t axis) const {
  GTensorTorchWrapper result;
  result.impl_->tensor = impl_->tensor.sum(axis);
  return result;
}

GTensorTorchWrapper GTensorTorchWrapper::mean_impl(int64_t axis) const {
  GTensorTorchWrapper result;
  result.impl_->tensor = impl_->tensor.mean(axis);
  return result;
}

GTensorTorchWrapper GTensorTorchWrapper::max_impl(int64_t axis) const {
  GTensorTorchWrapper result;
  auto [values, indices] = torch::max(impl_->tensor, axis);
  result.impl_->tensor = values;
  return result;
}

GTensorTorchWrapper GTensorTorchWrapper::min_impl(int64_t axis) const {
  GTensorTorchWrapper result;
  auto [values, indices] = torch::min(impl_->tensor, axis);
  result.impl_->tensor = values;
  return result;
}

GTensorTorchWrapper GTensorTorchWrapper::clamp_impl(const GTensorTorchWrapper &min,
                                                    const GTensorTorchWrapper &max) const {
  GTensorTorchWrapper result;
  result.impl_->tensor = torch::clamp(impl_->tensor, min.impl_->tensor, max.impl_->tensor);
  return result;
}

// &---------------------  变形  ---------------------
GTensorTorchWrapper GTensorTorchWrapper::reshape_impl(const TensorShape &shape) const {
  GTensorTorchWrapper result;
  result.impl_->tensor = impl_->tensor.reshape(shape);
  return result;
}

GTensorTorchWrapper GTensorTorchWrapper::expand_impl(const TensorShape &new_shape) const {
  GTensorTorchWrapper result;
  result.impl_->tensor = impl_->tensor.expand(new_shape);
  return result;
}

GTensorTorchWrapper GTensorTorchWrapper::squeeze_impl(int64_t dim) const {
  GTensorTorchWrapper result;
  result.impl_->tensor = impl_->tensor.squeeze(dim);
  return result;
}

GTensorTorchWrapper GTensorTorchWrapper::unsqueeze_impl(int64_t dim) const {
  GTensorTorchWrapper result;
  result.impl_->tensor = impl_->tensor.unsqueeze(dim);
  return result;
}

// &---------------------  取scalar的item方法  ---------------------
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

void GTensorTorchWrapper::set_tensor_default_device_id_impl(int device_id) {
  g_default_cuda_id = device_id;
}

void GTensorTorchWrapper::set_seed_impl(uint64_t seed) {
  torch::manual_seed(seed);
  if (torch::cuda::is_available()) {
    torch::cuda::manual_seed_all(seed);
  }
}

at::Tensor *GTensorTorchWrapper::get_torch_tensor_impl() {
  return &impl_->tensor;
}

} // namespace core
} // namespace cuda_simulator