#ifndef __GTENSOR_TORCH_WRAPPER_H__
#define __GTENSOR_TORCH_WRAPPER_H__

#include "ITensor.hh"
#include "Scalar.hh"
#include <cstdint>
#include <memory>

namespace cuda_simulator {
namespace core {

namespace internal {
class TorchTensorImpl;
}

// PyTorch后端的基类实现
class GTensorTorchWrapper final : public ITensor<GTensorTorchWrapper> {
  friend class ITensor<GTensorTorchWrapper>;

public:
  // 创建
  explicit GTensorTorchWrapper(const TensorShape &shape = {},
                               NumericalDataType dtype = NumericalDataType::kFloat32,
                               DeviceType device = DeviceType::kCUDA);
  explicit GTensorTorchWrapper(const Scalar &scalar, DeviceType device_type = DeviceType::kCUDA);

  ~GTensorTorchWrapper();

  // 拷贝
  GTensorTorchWrapper(const GTensorTorchWrapper &other) noexcept;
  // 移动
  GTensorTorchWrapper(GTensorTorchWrapper &&other) noexcept;

  // 赋值, 替换tensor
  GTensorTorchWrapper &operator=(const GTensorTorchWrapper &other) noexcept;
  // 移动赋值, 替换impl
  GTensorTorchWrapper &operator=(GTensorTorchWrapper &&other) noexcept;
  // 被Scalar赋值时，填充整个Tensor
  GTensorTorchWrapper &operator=(const Scalar &scalar) {
    fill(scalar);
    return *this;
  }

protected:
  // Scalar方法实现
  GTensorTorchWrapper &from_scalar_impl(const Scalar &scalar);
  Scalar to_scalar_impl() const;

  // 从主机数组创建张量
  static GTensorTorchWrapper from_host_array_impl(const void *data, int64_t numel, NumericalDataType type, DeviceType device);
  void to_host_array_impl(void *data, NumericalDataType type, int64_t numel) const;

  // 生成创建
  static GTensorTorchWrapper zeros_impl(const TensorShape &shape, NumericalDataType dtype,
                                       DeviceType device_type);
  static GTensorTorchWrapper rands_impl(const TensorShape &shape, NumericalDataType dtype,
                                       DeviceType device_type);

  // 属性
  TensorShape shape_impl() const;
  size_t elem_count_impl() const;
  size_t elem_size_impl() const;
  size_t dim_impl() const;
  NumericalDataType dtype_impl() const;
  DeviceType device_impl() const;
  bool is_contiguous_impl() const;
  void *data_impl();
  const void *data_impl() const;

  // 内部tensor操作
  void copy_from_impl(const GTensorTorchWrapper &other);
  void copy_to_impl(GTensorTorchWrapper &other) const;
  GTensorTorchWrapper clone_impl() const;
  void move_from_impl(GTensorTorchWrapper &&other);

  // 调试
  void print_impl(std::ostream &out) const;
  std::string to_string_impl() const;

  // 填充
  void zero_impl();
  void fill_impl(Scalar value);

  // batch操作
  void gather_sum_impl(const std::vector<const GTensorTorchWrapper*> src);
  void gather_mean_impl(const std::vector<const GTensorTorchWrapper*> src);
  void gather_max_impl(const std::vector<const GTensorTorchWrapper*> src);
  void gather_min_impl(const std::vector<const GTensorTorchWrapper*> src);

  // 四则运算
  GTensorTorchWrapper add_impl(const GTensorTorchWrapper &other) const;
  GTensorTorchWrapper sub_impl(const GTensorTorchWrapper &other) const;
  GTensorTorchWrapper mul_impl(const GTensorTorchWrapper &other) const;
  GTensorTorchWrapper div_impl(const GTensorTorchWrapper &other) const;
  static GTensorTorchWrapper matmul_impl(const GTensorTorchWrapper &a,
                                         const GTensorTorchWrapper &b);

  // 逻辑运算位运算
  GTensorTorchWrapper bitwise_not_impl() const;
  GTensorTorchWrapper neg_impl() const;
  GTensorTorchWrapper &add_inplace_impl(const GTensorTorchWrapper &other);
  GTensorTorchWrapper &sub_inplace_impl(const GTensorTorchWrapper &other);
  GTensorTorchWrapper &mul_inplace_impl(const GTensorTorchWrapper &other);
  GTensorTorchWrapper &div_inplace_impl(const GTensorTorchWrapper &other);
  GTensorTorchWrapper &bitwise_and_inplace_impl(const GTensorTorchWrapper &other);
  GTensorTorchWrapper &bitwise_or_inplace_impl(const GTensorTorchWrapper &other);
  GTensorTorchWrapper &bitwise_xor_inplace_impl(const GTensorTorchWrapper &other);

  // Scalar操作的具体实现
  GTensorTorchWrapper add_scalar_impl(const Scalar &scalar) const;
  GTensorTorchWrapper sub_scalar_impl(const Scalar &scalar) const;
  GTensorTorchWrapper mul_scalar_impl(const Scalar &scalar) const;
  GTensorTorchWrapper div_scalar_impl(const Scalar &scalar) const;
  GTensorTorchWrapper &add_inplace_scalar_impl(const Scalar &scalar);
  GTensorTorchWrapper &sub_inplace_scalar_impl(const Scalar &scalar);
  GTensorTorchWrapper &mul_inplace_scalar_impl(const Scalar &scalar);
  GTensorTorchWrapper &div_inplace_scalar_impl(const Scalar &scalar);

  // 索引操作
  GTensorTorchWrapper slice_impl(int64_t dim, int64_t start, int64_t end) const;
  GTensorTorchWrapper index_impl(int64_t index) const;
  GTensorTorchWrapper index_impl(const TensorShape &indices) const;

  // 沿axis的操作
  GTensorTorchWrapper sum_impl(int64_t axis) const;
  GTensorTorchWrapper mean_impl(int64_t axis) const;
  GTensorTorchWrapper max_impl(int64_t axis) const;
  GTensorTorchWrapper min_impl(int64_t axis) const;
  GTensorTorchWrapper clamp_impl(const GTensorTorchWrapper &min,
                                 const GTensorTorchWrapper &max) const;

  // 变形
  GTensorTorchWrapper expand_impl(const TensorShape &new_shape) const;
  GTensorTorchWrapper reshape_impl(const TensorShape &shape) const;
  GTensorTorchWrapper squeeze_impl(int64_t dim) const;
  GTensorTorchWrapper unsqueeze_impl(int64_t dim) const;

  float item_float_impl() const;
  double item_double_impl() const;
  int64_t item_int64_impl() const;
  int32_t item_int32_impl() const;

  // 取scalar的item方法
  template <typename T> inline T item_impl() const {
    if constexpr (std::is_same_v<T, float>) {
      return item_float_impl();
    } else if constexpr (std::is_same_v<T, double>) {
      return item_double_impl();
    } else if constexpr (std::is_same_v<T, int64_t>) {
      return item_int64_impl();
    } else if constexpr (std::is_same_v<T, int32_t>) {
      return item_int32_impl();
    } else {
      static_assert(always_false_v<T>, "Unsupported item type");
    }
  }

  // 类的Static声明
  static void set_tensor_default_device_id_impl(int device_id);
  static void set_seed_impl(uint64_t seed);

  // 兼容接口方法
  at::Tensor *get_torch_tensor_impl();

private:
  std::unique_ptr<internal::TorchTensorImpl> impl_;
};

} // namespace core
} // namespace cuda_simulator

#endif