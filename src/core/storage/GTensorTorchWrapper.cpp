#include "core/storage/GTensorTorchWrapper.hh"
#include "core/storage/ITensor.hh"
#include "core/storage/Scalar.hh"
#include <c10/core/TensorOptions.h>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <torch/torch.h>

namespace cuda_simulator {
namespace core {
namespace internal {
class TorchTensorImpl : public std::enable_shared_from_this<TorchTensorImpl> {
public:
  torch::Tensor tensor;
  NumericalDataType dtype;
  DeviceType device_type;

  TorchTensorImpl() = default;
  TorchTensorImpl(NumericalDataType dt, DeviceType devt) : dtype(dt), device_type(devt) {}

  TorchTensorImpl(torch::Tensor t) : tensor(t) {
    device_type = getDeviceType(t.device());
    dtype = getDataType(t.dtype());
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

std::shared_ptr<TorchTensorImpl> shareTorchTensorImpl(const std::shared_ptr<TorchTensorImpl> &impl) {
  return impl->shared_from_this();
}

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

GTensorTorchWrapper::GTensorTorchWrapper(NumericalDataType dtype, DeviceType device_type)
    : impl_(std::make_shared<internal::TorchTensorImpl>(dtype, device_type)) {}

GTensorTorchWrapper::GTensorTorchWrapper(const TensorShape &shape, NumericalDataType dtype,
                                         DeviceType device_type)
    : impl_(std::make_shared<internal::TorchTensorImpl>(dtype, device_type)) {
  int8_t device_id =
      (device_type == DeviceType::kCPU) ? 0 : g_default_cuda_id; // CPU设备ID为0，CUDA设备ID为g_default_cuda_id

  auto options = torch::TensorOptions()
                     .dtype(internal::TorchTensorImpl::getTorchDtype(dtype))
                     .device(internal::TorchTensorImpl::getTorchDevice(device_type), device_id);
  impl_->tensor = torch::zeros(shape, options);
}

GTensorTorchWrapper::GTensorTorchWrapper(const Scalar &scalar, DeviceType device_type)
    : impl_(std::make_shared<internal::TorchTensorImpl>(scalar.type(), device_type)) {
  int8_t device_id =
      (device_type == DeviceType::kCPU) ? 0 : g_default_cuda_id; // CPU设备ID为0，CUDA设备ID为g_default_cuda_id

  auto options = torch::TensorOptions()
                     .dtype(internal::TorchTensorImpl::getTorchDtype(scalar.type()))
                     .device(internal::TorchTensorImpl::getTorchDevice(device_type), device_id);
  impl_->tensor = torch::full({}, internal::toTorchScalar(scalar), options);
}

GTensorTorchWrapper &GTensorTorchWrapper::fromScalar(const Scalar &scalar) {
  int8_t device_id = (impl_->device_type == DeviceType::kCPU) ? 0 : g_default_cuda_id;
  impl_->tensor =
      torch::scalar_tensor(internal::toTorchScalar(scalar),
                           torch::TensorOptions()
                               .dtype(internal::TorchTensorImpl::getTorchDtype(impl_->dtype))
                               .device(internal::TorchTensorImpl::getTorchDevice(impl_->device_type), device_id));
  return *this;
}

// GTensorTorchWrapper::~GTensorTorchWrapper() {
//     std::cout << "GTensorTorchWrapper::~GTensorTorchWrapper()" << std::endl;
// }

// 数据访问实现
// 打印实现
void GTensorTorchWrapper::print(std::ostream &out) const { out << impl_->tensor; }

std::string GTensorTorchWrapper::toString() const {
  std::ostringstream oss;
  print(oss);
  return oss.str();
}

TensorShape GTensorTorchWrapper::shape() const {
  TensorShape ret_shape(impl_->tensor.sizes().begin(), impl_->tensor.sizes().end());
  return ret_shape;
}

DeviceType GTensorTorchWrapper::device() const { return impl_->device_type; }

size_t GTensorTorchWrapper::elemCount() const { return impl_->tensor.numel(); }

size_t GTensorTorchWrapper::elemSize() const { return impl_->tensor.element_size(); }

size_t GTensorTorchWrapper::dim() const {
  std::cout << impl_->tensor;
  return impl_->tensor.dim();
}

void GTensorTorchWrapper::zero() { impl_->tensor.zero_(); }

void GTensorTorchWrapper::fill(Scalar value) { impl_->tensor.fill_(internal::toTorchScalar(value)); }

void GTensorTorchWrapper::copyFrom(const GTensorTorchWrapper &other) { impl_->tensor.copy_(other.impl_->tensor); }

void GTensorTorchWrapper::copyTo(GTensorTorchWrapper &other) const { other.impl_->tensor.copy_(impl_->tensor); }

GTensorTorchWrapper GTensorTorchWrapper::reshapeImpl(const TensorShape &shape) {
  GTensorTorchWrapper result(impl_->dtype, impl_->device_type);
  result.impl_->tensor = torch::zeros(shape, impl_->tensor.options());
  return result;
}

void GTensorTorchWrapper::replaceTensor(const GTensorTorchWrapper &other) {
  impl_->device_type = other.impl_->device_type;
  impl_->dtype = other.impl_->dtype;
  impl_->tensor = other.impl_->tensor;
}

void GTensorTorchWrapper::replaceTensor(GTensorTorchWrapper &&other) {
  impl_->device_type = other.impl_->device_type;
  impl_->dtype = other.impl_->dtype;
  impl_->tensor = std::move(other.impl_->tensor);
}

GTensorTorchWrapper GTensorTorchWrapper::add_impl(const GTensorTorchWrapper &other) const {
  GTensorTorchWrapper result(impl_->dtype, impl_->device_type);
  result.impl_->tensor = impl_->tensor + other.impl_->tensor;
  return result;
}

GTensorTorchWrapper GTensorTorchWrapper::sub_impl(const GTensorTorchWrapper &other) const {
  GTensorTorchWrapper result(impl_->dtype, impl_->device_type);
  result.impl_->tensor = impl_->tensor - other.impl_->tensor;
  return result;
}

GTensorTorchWrapper GTensorTorchWrapper::mul_impl(const GTensorTorchWrapper &other) const {
  GTensorTorchWrapper result(impl_->dtype, impl_->device_type);
  result.impl_->tensor = impl_->tensor * other.impl_->tensor;
  return result;
}

GTensorTorchWrapper GTensorTorchWrapper::div_impl(const GTensorTorchWrapper &other) const {
  GTensorTorchWrapper result(impl_->dtype, impl_->device_type);
  result.impl_->tensor = impl_->tensor / other.impl_->tensor;
  return result;
}

GTensorTorchWrapper GTensorTorchWrapper::slice_impl(int64_t dim, int64_t start, int64_t end) const {
  GTensorTorchWrapper result(impl_->dtype, impl_->device_type);
  result.impl_->tensor = impl_->tensor.slice(dim, start, end);
  return result;
}

GTensorTorchWrapper GTensorTorchWrapper::index_impl(const TensorShape &indices) const {
  GTensorTorchWrapper result(impl_->dtype, impl_->device_type);
  std::vector<torch::indexing::TensorIndex> indices_torch(indices.begin(), indices.end());
  result.impl_->tensor = impl_->tensor.index(indices_torch);
  return result;
}

GTensorTorchWrapper GTensorTorchWrapper::index_impl(int64_t index) const {
  GTensorTorchWrapper result(impl_->dtype, impl_->device_type);
  result.impl_->tensor = impl_->tensor.select(0, index);
  return result;
}

GTensorTorchWrapper GTensorTorchWrapper::bitwise_not_impl() const {
  GTensorTorchWrapper result(impl_->dtype, impl_->device_type);
  result.impl_->tensor = torch::bitwise_not(impl_->tensor);
  return result;
}

GTensorTorchWrapper GTensorTorchWrapper::neg_impl() const {
  GTensorTorchWrapper result(impl_->dtype, impl_->device_type);
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

GTensorTorchWrapper &GTensorTorchWrapper::bitwise_and_inplace_impl(const GTensorTorchWrapper &other) {
  impl_->tensor.bitwise_and_(other.impl_->tensor);
  return *this;
}

GTensorTorchWrapper &GTensorTorchWrapper::bitwise_or_inplace_impl(const GTensorTorchWrapper &other) {
  impl_->tensor.bitwise_or_(other.impl_->tensor);
  return *this;
}

GTensorTorchWrapper GTensorTorchWrapper::clone() const {
  GTensorTorchWrapper result(impl_->dtype, impl_->device_type);
  result.impl_->tensor = impl_->tensor.clone();
  return result;
}

GTensorTorchWrapper GTensorTorchWrapper::move() { return GTensorTorchWrapper(std::move(*this)); }

float GTensorTorchWrapper::item_float_impl() const { return impl_->tensor.item<float>(); }

double GTensorTorchWrapper::item_double_impl() const { return impl_->tensor.item<double>(); }

int64_t GTensorTorchWrapper::item_int64_impl() const { return impl_->tensor.item<int64_t>(); }

int32_t GTensorTorchWrapper::item_int32_impl() const { return impl_->tensor.item<int32_t>(); }

NumericalDataType GTensorTorchWrapper::dtype() const { return impl_->dtype; }

bool GTensorTorchWrapper::is_contiguous() const { return impl_->tensor.is_contiguous(); }

void *GTensorTorchWrapper::data() { return impl_->tensor.data_ptr(); }

const void *GTensorTorchWrapper::data() const { return impl_->tensor.data_ptr(); }

GTensorTorchWrapper GTensorTorchWrapper::zerosImpl(const TensorShape &shape, NumericalDataType dtype,
                                                   DeviceType device_type) {
  GTensorTorchWrapper result(dtype, device_type);
  result.impl_->tensor =
      torch::zeros(shape, torch::TensorOptions()
                              .dtype(internal::TorchTensorImpl::getTorchDtype(dtype))
                              .device(internal::TorchTensorImpl::getTorchDevice(device_type), g_default_cuda_id));
  return result;
}

GTensorTorchWrapper GTensorTorchWrapper::randsImpl(const TensorShape &shape, NumericalDataType dtype,
                                                   DeviceType device_type) {
  GTensorTorchWrapper result(dtype, device_type);
  result.impl_->tensor =
      torch::rand(shape, torch::TensorOptions()
                             .dtype(internal::TorchTensorImpl::getTorchDtype(dtype))
                             .device(internal::TorchTensorImpl::getTorchDevice(device_type), g_default_cuda_id));
  return result;
}

Scalar GTensorTorchWrapper::toScalar() const {
  Scalar scalar(0);
  switch (impl_->dtype) {
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

GTensorTorchWrapper GTensorTorchWrapper::add_scalar_impl(const Scalar &scalar) const {
  GTensorTorchWrapper result(impl_->dtype, impl_->device_type);
  result.impl_->tensor = impl_->tensor + (internal::toTorchScalar(scalar));
  ;
  return result;
}

GTensorTorchWrapper GTensorTorchWrapper::sub_scalar_impl(const Scalar &scalar) const {
  GTensorTorchWrapper result(impl_->dtype, impl_->device_type);
  result.impl_->tensor = impl_->tensor - (internal::toTorchScalar(scalar));
  ;
  return result;
}

GTensorTorchWrapper GTensorTorchWrapper::mul_scalar_impl(const Scalar &scalar) const {
  GTensorTorchWrapper result(impl_->dtype, impl_->device_type);
  result.impl_->tensor = impl_->tensor * (internal::toTorchScalar(scalar));
  ;
  return result;
}

GTensorTorchWrapper GTensorTorchWrapper::div_scalar_impl(const Scalar &scalar) const {
  GTensorTorchWrapper result(impl_->dtype, impl_->device_type);
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

GTensorTorchWrapper GTensorTorchWrapper::expandImpl(const TensorShape &new_shape) const {
  GTensorTorchWrapper result(impl_->dtype, impl_->device_type);
  result.impl_->tensor = impl_->tensor.expand(new_shape);
  return result;
}

void GTensorTorchWrapper::gatherSum(const std::vector<GTensorTorchWrapper> src) {
  std::vector<torch::Tensor> tensors;
  tensors.reserve(src.size());

  // 收集所有源张量
  for (auto &tensor : src) {
    tensors.push_back(tensor.impl_->tensor);
  }

  // 使用stack在新维度上堆叠所有张量，然后在该维度上求和
  auto stacked = torch::stack(tensors, 0);
  impl_->tensor = torch::sum(stacked, 0);
}

void GTensorTorchWrapper::gatherMean(const std::vector<GTensorTorchWrapper> src) {
  std::vector<torch::Tensor> tensors;
  tensors.reserve(src.size());

  for (auto &tensor : src) {
    tensors.push_back(tensor.impl_->tensor);
  }

  auto stacked = torch::stack(tensors, 0);
  impl_->tensor = torch::mean(stacked, 0);
}

void GTensorTorchWrapper::gatherMax(const std::vector<GTensorTorchWrapper> src) {
  std::vector<torch::Tensor> tensors;
  tensors.reserve(src.size());

  for (auto &tensor : src) {
    tensors.push_back(tensor.impl_->tensor);
  }

  auto stacked = torch::stack(tensors, 0);
  // max 返回两个值：最大值和最大值的索引，我们只需要最大值
  auto [values, indices] = torch::max(stacked, 0);
  impl_->tensor = values;
}

void GTensorTorchWrapper::gatherMin(const std::vector<GTensorTorchWrapper> src) {
  std::vector<torch::Tensor> tensors;
  tensors.reserve(src.size());

  for (auto &tensor : src) {
    tensors.push_back(tensor.impl_->tensor);
  }

  auto stacked = torch::stack(tensors, 0);
  // min 返回两个值：最小值和最小值的索引，我们只需要最小值
  auto [values, indices] = torch::min(stacked, 0);
  impl_->tensor = values;
}

void GTensorTorchWrapper::fromHostArray(const void *data, NumericalDataType type, int64_t numel) {
  auto new_tensor = torch::from_blob(const_cast<void *>(data), {numel}, internal::TorchTensorImpl::getTorchDtype(type));

  new_tensor =
      new_tensor.to(torch::TensorOptions()
                        .device(internal::TorchTensorImpl::getTorchDevice(impl_->device_type), g_default_cuda_id)
                        .dtype(internal::TorchTensorImpl::getTorchDtype(type)));

  impl_->tensor = new_tensor.clone();

  // 转换到指定设备
  impl_->dtype = type;
}

void GTensorTorchWrapper::toHostArray(void *data, NumericalDataType type, int64_t numel) const {
  auto new_tensor = impl_->tensor.to(
      torch::TensorOptions().device(torch::kCPU).dtype(internal::TorchTensorImpl::getTorchDtype(type)));

  new_tensor = new_tensor.reshape({numel});
  std::memcpy(data, new_tensor.data_ptr(), numel * new_tensor.element_size());
}

void GTensorTorchWrapper::setTensorDefaultDeviceIdImpl(int device_id) { g_default_cuda_id = device_id; }

void GTensorTorchWrapper::setSeedImpl(uint64_t seed) {
  torch::manual_seed(seed);
  if (torch::cuda::is_available()) {
    torch::cuda::manual_seed_all(seed);
  }
}

} // namespace core
} // namespace cuda_simulator