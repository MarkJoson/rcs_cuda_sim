#ifndef __GTENSOR_TORCH_WRAPPER_H__
#define __GTENSOR_TORCH_WRAPPER_H__

#include <cstdint>
#include <memory>
#include "ITensor.hh"
#include "Scalar.hh"

namespace cuda_simulator {
namespace core {

namespace internal {
    class TorchTensorImpl;
    std::shared_ptr<TorchTensorImpl> shareTorchTensorImpl(const std::shared_ptr<TorchTensorImpl> &impl);
}



// PyTorch后端的基类实现
class GTensorTorchWrapper final : public ITensor<GTensorTorchWrapper> {
    friend class ITensor<GTensorTorchWrapper>;
public:
    // 创建
    explicit GTensorTorchWrapper(
        NumericalDataType dtype=NumericalDataType::kFloat32,
        DeviceType device_type=DeviceType::kCUDA);

    explicit GTensorTorchWrapper(
        const TensorShape& shape,
        NumericalDataType dtype=NumericalDataType::kFloat32,
        DeviceType device_type=DeviceType::kCUDA);

    explicit GTensorTorchWrapper(
        const Scalar& scalar,
        DeviceType device_type=DeviceType::kCUDA);

    explicit GTensorTorchWrapper(
        const std::shared_ptr<internal::TorchTensorImpl>& impl)
        : impl_(impl) {}

    GTensorTorchWrapper& fromScalar(const Scalar& scalar) override;

    // 拷贝
    GTensorTorchWrapper(const GTensorTorchWrapper& other) : impl_(other.impl_) {}
    // 移动
    GTensorTorchWrapper(GTensorTorchWrapper&& other) noexcept : impl_(std::move(other.impl_)) {}

    virtual ~GTensorTorchWrapper() final = default;

    GTensorTorchWrapper& operator=(const GTensorTorchWrapper& other) {
        // !赋值操作将替换最内层的tensor，
        // TODO. 引入*符号取代当前操作
        if (this != &other) { replaceTensor(other); }
        return *this;
    }

    GTensorTorchWrapper& operator=(GTensorTorchWrapper&& other) noexcept {
        // !赋值操作将替换最内层的tensor
        if (this != &other) { replaceTensor(std::move(other)); }
        return *this;
    }

    // 被Scalar赋值时，填充整个Tensor
    GTensorTorchWrapper& operator=(const Scalar& scalar) {
        fill(scalar);
        return *this;
    }


    TensorShape shape() const override;
    size_t elemCount() const override;
    size_t elemSize() const override;
    size_t dim() const override;
    NumericalDataType dtype() const override;
    DeviceType device() const override;
    bool is_contiguous() const override;

    void* data() override;
    const void* data() const override;

    void print(std::ostream &out) const override;
    std::string toString() const override;

    virtual void zero() override;
    void fill(Scalar value) override;
    void copyFrom(const GTensorTorchWrapper& other) override;
    void copyTo(GTensorTorchWrapper& other) const override;

    // TODO. replaceTensor需要成为虚函数吗？
    // TODO. fixme. bindTensor 和 replaceTensor这种奇怪的东西还是别要了
    // 替换内部的Tensor
    void replaceTensor(const GTensorTorchWrapper& other);
    void replaceTensor(GTensorTorchWrapper&& other);              // 移动替换内部的Tensor
    // 绑定shared_ptr，不拷贝Tensor。使两个对象共享同一个TensorPtr
    void bindTensorRef(const GTensorTorchWrapper& other) { impl_ = other.impl_; }
    void bindTensorRef(GTensorTorchWrapper&& other) { impl_ = std::move(other.impl_); }

    // 从主机数组创建张量
    void fromHostArray(const void* data, NumericalDataType type, int64_t numel) override;
    void toHostArray(void* data, NumericalDataType type, int64_t numel) const override;

    GTensorTorchWrapper clone() const override;
    GTensorTorchWrapper move() override;

    // Scalar方法实现
    Scalar toScalar() const override;

    void gatherSum(const std::vector<GTensorTorchWrapper> src) override;
    void gatherMean(const std::vector<GTensorTorchWrapper> src) override;
    void gatherMax(const std::vector<GTensorTorchWrapper> src) override;
    void gatherMin(const std::vector<GTensorTorchWrapper> src) override;

protected:
    // 具体实现方法
    GTensorTorchWrapper add_impl(const GTensorTorchWrapper& other) const;
    GTensorTorchWrapper sub_impl(const GTensorTorchWrapper& other) const;
    GTensorTorchWrapper mul_impl(const GTensorTorchWrapper& other) const;
    GTensorTorchWrapper div_impl(const GTensorTorchWrapper& other) const;
    static GTensorTorchWrapper matmul_impl(const GTensorTorchWrapper& a, const GTensorTorchWrapper& b);

    // 实现ITensor要求的内部方法
    GTensorTorchWrapper bitwise_not_impl() const;
    GTensorTorchWrapper neg_impl() const;
    GTensorTorchWrapper& add_inplace_impl(const GTensorTorchWrapper& other);
    GTensorTorchWrapper& sub_inplace_impl(const GTensorTorchWrapper& other);
    GTensorTorchWrapper& mul_inplace_impl(const GTensorTorchWrapper& other);
    GTensorTorchWrapper& div_inplace_impl(const GTensorTorchWrapper& other);
    GTensorTorchWrapper& bitwise_and_inplace_impl(const GTensorTorchWrapper& other);
    GTensorTorchWrapper& bitwise_or_inplace_impl(const GTensorTorchWrapper& other);
    GTensorTorchWrapper& bitwise_xor_inplace_impl(const GTensorTorchWrapper& other);

    // Scalar操作的具体实现
    GTensorTorchWrapper add_scalar_impl(const Scalar& scalar) const;
    GTensorTorchWrapper sub_scalar_impl(const Scalar& scalar) const;
    GTensorTorchWrapper mul_scalar_impl(const Scalar& scalar) const;
    GTensorTorchWrapper div_scalar_impl(const Scalar& scalar) const;
    GTensorTorchWrapper& add_inplace_scalar_impl(const Scalar& scalar);
    GTensorTorchWrapper& sub_inplace_scalar_impl(const Scalar& scalar);
    GTensorTorchWrapper& mul_inplace_scalar_impl(const Scalar& scalar);
    GTensorTorchWrapper& div_inplace_scalar_impl(const Scalar& scalar);

    // 索引操作
    GTensorTorchWrapper slice_impl(int64_t dim, int64_t start, int64_t end) const;
    GTensorTorchWrapper index_impl(int64_t index) const;
    GTensorTorchWrapper index_impl(const TensorShape& indices) const;

    // 沿axis的操作
    GTensorTorchWrapper sum_impl(int64_t axis) const;
    GTensorTorchWrapper mean_impl(int64_t axis) const;
    GTensorTorchWrapper max_impl(int64_t axis) const;
    GTensorTorchWrapper min_impl(int64_t axis) const;
    GTensorTorchWrapper clamp_impl(const GTensorTorchWrapper& min, const GTensorTorchWrapper& max) const;

    // 变形
    GTensorTorchWrapper expand_impl(const TensorShape& new_shape) const;
    GTensorTorchWrapper reshape_impl(const TensorShape &shape) const;
    GTensorTorchWrapper squeeze_impl(int64_t dim) const;
    GTensorTorchWrapper unsqueeze_impl(int64_t dim) const;

    static GTensorTorchWrapper zerosImpl(const TensorShape& shape, NumericalDataType dtype, DeviceType device_type);
    static GTensorTorchWrapper randsImpl(const TensorShape& shape, NumericalDataType dtype, DeviceType device_type);

    float item_float_impl() const;
    double item_double_impl() const;
    int64_t item_int64_impl() const;
    int32_t item_int32_impl() const;

    // 取scalar的item方法
    template<typename T>
    inline T item_impl() const {
        if constexpr (std::is_same_v<T, float>) {
            return item_float_impl();
        } else if constexpr (std::is_same_v<T, double>) {
            return item_double_impl();
        } else if constexpr (std::is_same_v<T, int64_t>) {
            return item_int64_impl();
        } else if constexpr (std::is_same_v<T, int32_t>) {
            return item_int32_impl();
        } else {
            // print what type is not supported
            static_assert(always_false_v<T>, "Unsupported item type");
        }
    }

    template<typename T>
    void fromHostVectorImpl(const std::vector<T>& vec) {
        if constexpr (std::is_same_v<T, float>) {
            fromHostArray(vec.data(), NumericalDataType::kFloat32, vec.size());
        } else if constexpr (std::is_same_v<T, double>) {
            fromHostArray(vec.data(), NumericalDataType::kFloat64, vec.size());
        } else if constexpr (std::is_same_v<T, int64_t>) {
            fromHostArray(vec.data(), NumericalDataType::kInt64, vec.size());
        } else if constexpr (std::is_same_v<T, int32_t>) {
            fromHostArray(vec.data(), NumericalDataType::kInt32, vec.size());
        } else if constexpr (std::is_same_v<T, uint8_t>) {
            fromHostArray(vec.data(), NumericalDataType::kUInt8, vec.size());
        } else if constexpr (std::is_same_v<T, uint32_t>) {
            fromHostArray(vec.data(), NumericalDataType::kUInt32, vec.size());
        } else {
            // print what type is not supported
            static_assert(always_false_v<T>, "Unsupported data type");
        }
    }

    template<typename T>
    void toHostVectorImpl(std::vector<T>& vec) const {
        vec.resize(elemCount());
        if constexpr (std::is_same_v<T, float>) {
            toHostArray(vec.data(), NumericalDataType::kFloat32, vec.size());
        } else if constexpr (std::is_same_v<T, double>) {
            toHostArray(vec.data(), NumericalDataType::kFloat64, vec.size());
        } else if constexpr (std::is_same_v<T, int64_t>) {
            toHostArray(vec.data(), NumericalDataType::kInt64, vec.size());
        } else if constexpr (std::is_same_v<T, int32_t>) {
            toHostArray(vec.data(), NumericalDataType::kInt32, vec.size());
        } else if constexpr (std::is_same_v<T, uint8_t>) {
            toHostArray(vec.data(), NumericalDataType::kUInt8, vec.size());
        } else if constexpr (std::is_same_v<T, uint32_t>) {
            toHostArray(vec.data(), NumericalDataType::kUInt32, vec.size());
        } else {
            // print what type is not supported
            static_assert(always_false_v<T>, "Unsupported data type");
        }
    }

    // 类的Static声明
    static void setTensorDefaultDeviceIdImpl(int device_id);
    static void setSeedImpl(uint64_t seed);


private:
    std::shared_ptr<internal::TorchTensorImpl> impl_;
};



} // namespace core
} // namespace cuda_simulator

#endif