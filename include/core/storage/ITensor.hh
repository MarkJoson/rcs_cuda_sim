#ifndef __ITENSOR_H__
#define __ITENSOR_H__

#include "Scalar.hh"
#include <cstdint>
#include <iostream>
#include <ostream>
#include <vector>

namespace at {
class Tensor;
};

namespace cuda_simulator {
namespace core {

enum class DeviceType {
  kCUDA,
  kCPU,
  kNumDevices
};

using TensorShape = std::vector<int64_t>;

template <typename Derived> class ITensor {
public:
  // Helper function
  Derived &derived() {
    return static_cast<Derived &>(*this);
  }
  const Derived &derived() const {
    return static_cast<const Derived &>(*this);
  }

  // Scalar conversion methods
  inline Derived &fromScalar(const Scalar &scalar) {
    return derived().from_scalar_impl(scalar);
  };
  inline Scalar toScalar() const {
    return derived().to_scalar_impl();
  };

  // 从主机数组创建张量
  template <typename T> inline static Derived fromHostVector(const std::vector<T> &vec) {
    if constexpr (std::is_same_v<T, float>) {
      return fromHostArray(vec.data(), vec.size(), NumericalDataType::kFloat32);
    } else if constexpr (std::is_same_v<T, double>) {
      return fromHostArray(vec.data(), vec.size(), NumericalDataType::kFloat64);
    } else if constexpr (std::is_same_v<T, int64_t>) {
      return fromHostArray(vec.data(), vec.size(), NumericalDataType::kInt64);
    } else if constexpr (std::is_same_v<T, int32_t>) {
      return fromHostArray(vec.data(), vec.size(), NumericalDataType::kInt32);
    } else if constexpr (std::is_same_v<T, uint8_t>) {
      return fromHostArray(vec.data(), vec.size(), NumericalDataType::kUInt8);
    } else if constexpr (std::is_same_v<T, uint32_t>) {
      return fromHostArray(vec.data(), vec.size(), NumericalDataType::kUInt32);
    } else {
      // print what type is not supported
      static_assert(always_false_v<T>, "Unsupported data type");
    }
  }
  template <typename T> inline void toHostVector(std::vector<T> &vec) const {
    vec.resize(elemCount());
    if constexpr (std::is_same_v<T, float>) {
      toHostArray(vec.data(), vec.size(), NumericalDataType::kFloat32);
    } else if constexpr (std::is_same_v<T, double>) {
      toHostArray(vec.data(), vec.size(), NumericalDataType::kFloat64);
    } else if constexpr (std::is_same_v<T, int64_t>) {
      toHostArray(vec.data(), vec.size(), NumericalDataType::kInt64);
    } else if constexpr (std::is_same_v<T, int32_t>) {
      toHostArray(vec.data(), vec.size(), NumericalDataType::kInt32);
    } else if constexpr (std::is_same_v<T, uint8_t>) {
      toHostArray(vec.data(), vec.size(), NumericalDataType::kUInt8);
    } else if constexpr (std::is_same_v<T, uint32_t>) {
      toHostArray(vec.data(), vec.size(), NumericalDataType::kUInt32);
    } else {
      // print what type is not supported
      static_assert(always_false_v<T>, "Unsupported data type");
    }
  }

  static inline Derived fromHostArray(const void *data, int64_t numel, NumericalDataType type,
                                      DeviceType device=DeviceType::kCUDA) {
    return Derived::from_host_array_impl(data, numel, type, device);
  }
  inline void toHostArray(void *data, int64_t numel, NumericalDataType type) const {
    derived().to_host_array_impl(data, numel, type);
  };

  // 属性
  inline TensorShape shape() const {
    return derived().shape_impl();
  }
  inline size_t elemCount() const {
    return derived().elem_count_impl();
  }
  inline size_t elemSize() const {
    return derived().elem_size_impl();
  }
  inline size_t numel() const {
    return elemCount();
  }
  inline size_t dim() const {
    return derived().dim_impl();
  }
  inline NumericalDataType dtype() const {
    return derived().dtype_impl();
  }
  inline DeviceType device() const {
    return derived().device_impl();
  }
  inline bool isContiguous() const {
    return derived().is_contiguous_impl();
  }
  inline void *data() {
    return derived().data_impl();
  }
  inline const void *data() const {
    return derived().data_impl();
  }

  // 内部tensor操作
  inline void copyFrom(const Derived &other) {
    derived().copy_from_impl(other);
  };
  inline void copyTo(Derived &other) const {
    derived().copy_to_impl(other);
  };
  inline Derived clone() const {
    return derived().clone_impl();
  };
  inline void moveFrom(Derived &&other) {
    derived().move_from_impl(std::move(other));
  };

  // 生成新Tensor
  static inline Derived zeros(const TensorShape &shape,
                              NumericalDataType dtype = NumericalDataType::kFloat32,
                              DeviceType device_type = DeviceType::kCUDA) {
    return Derived::zeros_impl(shape, dtype, device_type);
  }
  static inline Derived rands(const TensorShape &shape,
                              NumericalDataType dtype = NumericalDataType::kFloat32,
                              DeviceType device_type = DeviceType::kCUDA) {
    return Derived::rands_impl(shape, dtype, device_type);
  }

  inline Derived zerosLike() const {
    return zeros(shape(), dtype(), device());
  }
  inline Derived randsLike() const {
    return rands(shape(), dtype(), device());
  }

  //
  inline void print() const {
    derived().print_impl(std::cout);
  };
  inline void print(std::ostream &out) const {
    derived().print_impl(out);
  };
  inline std::string toString() const {
    return derived().to_string_impl();
  };

  // 填充
  inline void zero() {
    derived().zero_impl();
  };
  inline void fill(Scalar value) {
    derived().fill_impl(value);
  };

  // Gather方法
  inline void gatherSum(const std::vector<const Derived*> src) {
    derived().gather_sum_impl(src);
  }
  inline void gatherMean(const std::vector<const Derived*> src) {
    derived().gather_mean_impl(src);
  }
  inline void gatherMax(const std::vector<const Derived*> src) {
    derived().gather_max_impl(src);
  }
  inline void gatherMin(const std::vector<const Derived*> src) {
    derived().gather_min_impl(src);
  }

  template <typename T> inline T item() const {
    return derived().template item_impl<T>();
  }

  // 返回新Tensor的操作
  inline Derived add(const Derived &other) const {
    return derived().add_impl(other);
  }
  inline Derived sub(const Derived &other) const {
    return derived().sub_impl(other);
  }
  inline Derived mul(const Derived &other) const {
    return derived().mul_impl(other);
  }
  inline Derived div(const Derived &other) const {
    return derived().div_impl(other);
  }
  inline static Derived matmul(const Derived &a, const Derived &b) {
    return Derived::matmul_impl(a, b);
  }

  // Scalar operators
  inline Derived add(const Scalar &scalar) const {
    return derived().add_scalar_impl(scalar);
  }
  inline Derived sub(const Scalar &scalar) const {
    return derived().sub_scalar_impl(scalar);
  }
  inline Derived mul(const Scalar &scalar) const {
    return derived().mul_scalar_impl(scalar);
  }
  inline Derived div(const Scalar &scalar) const {
    return derived().div_scalar_impl(scalar);
  }

  // 运算符重载
  inline Derived operator+(const Derived &other) const {
    return add(other);
  }
  inline Derived operator-(const Derived &other) const {
    return sub(other);
  }
  inline Derived operator*(const Derived &other) const {
    return mul(other);
  }
  inline Derived operator/(const Derived &other) const {
    return div(other);
  }

  inline Derived operator+(const Scalar &scalar) const {
    return add(scalar);
  }
  inline Derived operator-(const Scalar &scalar) const {
    return sub(scalar);
  }
  inline Derived operator*(const Scalar &scalar) const {
    return mul(scalar);
  }
  inline Derived operator/(const Scalar &scalar) const {
    return div(scalar);
  }

  // 复合赋值操作符
  inline Derived &operator+=(const Derived &other) {
    return derived().add_inplace_impl(other);
  }
  inline Derived &operator-=(const Derived &other) {
    return derived().sub_inplace_impl(other);
  }
  inline Derived &operator*=(const Derived &other) {
    return derived().mul_inplace_impl(other);
  }
  inline Derived &operator/=(const Derived &other) {
    return derived().div_inplace_impl(other);
  }

  inline Derived &operator+=(const Scalar &scalar) {
    return derived().add_inplace_scalar_impl(scalar);
  }
  inline Derived &operator-=(const Scalar &scalar) {
    return derived().sub_inplace_scalar_impl(scalar);
  }
  inline Derived &operator*=(const Scalar &scalar) {
    return derived().mul_inplace_scalar_impl(scalar);
  }
  inline Derived &operator/=(const Scalar &scalar) {
    return derived().div_inplace_scalar_impl(scalar);
  }

  // 位运算操作符
  inline Derived operator~() const {
    return static_cast<const Derived *>(this)->bitwise_not_impl();
  }
  inline Derived operator-() const {
    return static_cast<const Derived *>(this)->neg_impl();
  }
  inline Derived &operator&=(const Derived &other) {
    return derived().bitwise_and_inplace_impl(other);
  }
  inline Derived &operator|=(const Derived &other) {
    return derived().bitwise_or_inplace_impl(other);
  }

  // 索引操作
  inline Derived slice(int64_t dim, int64_t start, int64_t end) const {
    return derived().slice_impl(dim, start, end);
  }
  inline Derived operator[](int64_t index) const {
    return derived().index_impl(index);
  }
  inline Derived operator[](const TensorShape &indices) const {
    return derived().index_impl(indices);
  }

  // 沿axis的操作
  inline Derived sum(int64_t axis) const {
    return derived().sum_impl(axis);
  }
  inline Derived mean(int64_t axis) const {
    return derived().mean_impl(axis);
  }
  inline Derived max(int64_t axis) const {
    return derived().max_impl(axis);
  }
  inline Derived min(int64_t axis) const {
    return derived().min_impl(axis);
  }
  inline Derived clamp(const Derived &min, const Derived &max) const {
    return derived().clamp(min, max);
  }

  // 变形
  inline Derived expand(const TensorShape &new_shape) const {
    return derived().expand_impl(new_shape);
  }
  inline Derived reshape(const TensorShape &shape) const {
    return derived().reshape_impl(shape);
  };
  inline Derived squeeze(int64_t dim) const {
    return derived().squeeze_impl(dim);
  }
  inline Derived unsqueeze(int64_t dim) const {
    return derived().unsqueeze_impl(dim);
  }

  // 获取原始数据
  template <typename T> inline T *typed_data() {
    return static_cast<T *>(this->data());
  }
  template <typename T> inline const T *typed_data() const {
    return static_cast<const T *>(this->data());
  }

  static inline void setTensorDefaultDeviceId(int device_id) {
    Derived::set_tensor_default_device_id_impl(device_id);
  }
  static inline void setSeed(uint64_t seed) {
    Derived::set_seed_impl(seed);
  }

  at::Tensor *getTorchTensor() {
    return derived().get_torch_tensor_impl();
  }

protected:
  ~ITensor() = default;
};

template <typename Derived>
static std::ostream &operator<<(std::ostream &out, const ITensor<Derived> &t) {
  t.print(out);
  return out;
}

} // namespace core

} // namespace cuda_simulator

#endif