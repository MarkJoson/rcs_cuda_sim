#ifndef __GTENSOR_TORCH_WRAPPER_H__
#define __GTENSOR_TORCH_WRAPPER_H__

#include <memory>
#include "ITensor.hh"
#include "core/storage/Scalar.hh"

namespace cuda_simulator
{
namespace core
{


namespace internal
{
    class TorchTensorImpl;
    std::shared_ptr<TorchTensorImpl> shareTorchTensorImpl(const std::shared_ptr<TorchTensorImpl> &impl);
}

template<typename T>
inline constexpr bool always_false_v = false;

// PyTorch后端的基类实现
class GTensorTorchWrapper final : public ITensor<GTensorTorchWrapper> {
    friend class ITensor<GTensorTorchWrapper>;
public:
    // explicit GTensorTorchWrapper();
    explicit GTensorTorchWrapper(const std::vector<int64_t>& shape, NumericalDataType dtype);
    explicit GTensorTorchWrapper(const Scalar& scalar);

    virtual ~GTensorTorchWrapper() final = default;

    GTensorTorchWrapper(const GTensorTorchWrapper& other)
        : impl_(internal::shareTorchTensorImpl(other.impl_)), dtype_(other.dtype_) {}

    GTensorTorchWrapper(GTensorTorchWrapper&& other) noexcept
        : impl_(std::move(other.impl_)), dtype_(other.dtype_) {}

    GTensorTorchWrapper& operator=(const GTensorTorchWrapper& other) {
        if (this != &other) {
            impl_ = internal::shareTorchTensorImpl(other.impl_);
            dtype_ = other.dtype_;
        }
        return *this;
    }

    GTensorTorchWrapper& operator=(GTensorTorchWrapper&& other) noexcept {
        if (this != &other) {
            impl_ = std::move(other.impl_);
            dtype_ = other.dtype_;
        }
        return *this;
    }

    // 被Scalar赋值时，填充整个Tensor
    GTensorTorchWrapper& operator=(const Scalar& scalar) {
        fill(scalar);
        return *this;
    }


    std::vector<int64_t> shape() const override;
    size_t elemCount() const override;
    size_t elemSize() const override;
    size_t dim() const override;
    NumericalDataType dtype() const override;

    void* data() override;
    const void* data() const override;

    void print(std::ostream &out) const override;
    std::string toString() const override;

    virtual void zero() override;
    void fill(Scalar value) override;
    void copyFrom(const GTensorTorchWrapper& other) override;
    void copyTo(GTensorTorchWrapper& other) const override;

    GTensorTorchWrapper clone() const override;
    GTensorTorchWrapper move() override;

    // Scalar方法实现
    Scalar toScalar() const override;
    GTensorTorchWrapper& fromScalar(const Scalar& scalar) override;

    void gather_sum(const std::vector<const GTensorTorchWrapper*> src) override;
    void gather_mean(const std::vector<const GTensorTorchWrapper*> src) override;
    void gather_max(const std::vector<const GTensorTorchWrapper*> src) override;
    void gather_min(const std::vector<const GTensorTorchWrapper*> src) override;

protected:
    // 具体实现方法
    GTensorTorchWrapper add_impl(const GTensorTorchWrapper& other) const;
    GTensorTorchWrapper sub_impl(const GTensorTorchWrapper& other) const;
    GTensorTorchWrapper mul_impl(const GTensorTorchWrapper& other) const;
    GTensorTorchWrapper div_impl(const GTensorTorchWrapper& other) const;
    GTensorTorchWrapper slice_impl(int64_t dim, int64_t start, int64_t end) const;
    GTensorTorchWrapper index_impl(int64_t index) const;
    GTensorTorchWrapper index_impl(const std::vector<int64_t>& indices) const;

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

    template<typename T>
    T item_impl() const {
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

    float item_float_impl() const;
    double item_double_impl() const;
    int64_t item_int64_impl() const;
    int32_t item_int32_impl() const;

    std::shared_ptr<internal::TorchTensorImpl> impl_;
    NumericalDataType dtype_;
};



} // namespace core
} // namespace cuda_simulator

#endif