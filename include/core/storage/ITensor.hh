#ifndef __ITENSOR_H__
#define __ITENSOR_H__

#include <ostream>
#include <vector>
#include <iostream>
#include "Scalar.hh"

namespace cuda_simulator
{
namespace core
{

template<typename T>
inline constexpr bool always_false_v = false;

enum class DeviceType {
    kCUDA,
    kCPU,
    kNumDevices
};

using TensorShape = std::vector<int64_t>;

template<typename Derived>
class ITensor {
public:
    // Helper function
    Derived& derived() { return static_cast<Derived&>(*this); }
    const Derived& derived() const { return static_cast<const Derived&>(*this); }

    // copy and move
    virtual Derived clone() const = 0;
    virtual Derived move() = 0;


    // basic information
    virtual std::vector<int64_t> shape() const = 0;
    virtual size_t elemCount() const = 0;
    virtual size_t elemSize() const = 0;
    virtual size_t dim() const = 0;
    virtual NumericalDataType dtype() const = 0;
    virtual bool is_contiguous() const = 0;

    // raw data access
    virtual void* data() = 0;
    virtual const void* data() const = 0;

    //
    virtual void print(std::ostream &out) const = 0;
    virtual std::string toString() const = 0;

    //
    virtual void zero() = 0;
    virtual void fill(Scalar value) = 0;
    virtual void copyFrom(const Derived& other) = 0;
    virtual void copyTo(Derived& other) const = 0;
    virtual void resize(const std::vector<int64_t>& shape) = 0;

    // Gather方法
    virtual void gatherSum(const std::vector<Derived> src) = 0;
    virtual void gatherMean(const std::vector<Derived> src) = 0;
    virtual void gatherMax(const std::vector<Derived> src) = 0;
    virtual void gatherMin(const std::vector<Derived> src) = 0;

    template<typename T>
    inline T item() const { return derived().template item_impl<T>(); }


    static inline void setTensorDefaultDeviceId(int device_id) {
        Derived::setTensorDefaultDeviceIdImpl(device_id);
    }

    static inline void setSeed(uint64_t seed) {
        Derived::setSeedImpl(seed);
    }


    // 返回新Tensor的操作
    inline Derived add(const Derived& other) const { return derived().add_impl(other); }
    inline Derived sub(const Derived& other) const { return derived().sub_impl(other); }
    inline Derived mul(const Derived& other) const { return derived().mul_impl(other); }
    inline Derived div(const Derived& other) const { return derived().div_impl(other); }

    // Scalar operators
    inline Derived add(const Scalar& scalar) const { return derived().add_scalar_impl(scalar); }
    inline Derived sub(const Scalar& scalar) const { return derived().sub_scalar_impl(scalar); }
    inline Derived mul(const Scalar& scalar) const { return derived().mul_scalar_impl(scalar); }
    inline Derived div(const Scalar& scalar) const { return derived().div_scalar_impl(scalar); }

    // 运算符重载
    inline Derived operator+(const Derived& other) const { return add(other); }
    inline Derived operator-(const Derived& other) const { return sub(other); }
    inline Derived operator*(const Derived& other) const { return mul(other); }
    inline Derived operator/(const Derived& other) const { return div(other); }

    inline Derived operator+(const Scalar& scalar) const { return add(scalar); }
    inline Derived operator-(const Scalar& scalar) const { return sub(scalar); }
    inline Derived operator*(const Scalar& scalar) const { return mul(scalar); }
    inline Derived operator/(const Scalar& scalar) const { return div(scalar); }

    // 复合赋值操作符
    inline Derived& operator+=(const Derived& other) { return derived().add_inplace_impl(other); }
    inline Derived& operator-=(const Derived& other) { return derived().sub_inplace_impl(other); }
    inline Derived& operator*=(const Derived& other) { return derived().mul_inplace_impl(other); }
    inline Derived& operator/=(const Derived& other) { return derived().div_inplace_impl(other); }

    inline Derived& operator+=(const Scalar& scalar) { return derived().add_inplace_scalar_impl(scalar); }
    inline Derived& operator-=(const Scalar& scalar) { return derived().sub_inplace_scalar_impl(scalar); }
    inline Derived& operator*=(const Scalar& scalar) { return derived().mul_inplace_scalar_impl(scalar); }
    inline Derived& operator/=(const Scalar& scalar) { return derived().div_inplace_scalar_impl(scalar); }

    // 位运算操作符
    inline Derived operator~() const { return static_cast<const Derived*>(this)->bitwise_not_impl(); }
    inline Derived operator-() const { return static_cast<const Derived*>(this)->neg_impl(); }
    inline Derived& operator&=(const Derived& other) { return derived().bitwise_and_inplace_impl(other); }
    inline Derived& operator|=(const Derived& other) { return derived().bitwise_or_inplace_impl(other); }

    // 索引操作
    inline Derived slice(int64_t dim, int64_t start, int64_t end) const { return derived().slice_impl(dim, start, end); }
    inline Derived operator[](int64_t index) const { return derived().index_impl(index); }
    inline Derived operator[](const std::vector<int64_t>& indices) const { return derived().index_impl(indices); }


    template<typename T>
    inline T* typed_data() { return static_cast<T*>(this->data()); }

    template<typename T>
    inline const T* typed_data() const { return static_cast<const T*>(this->data()); }

    // 工具方法
    template<typename U>
    static constexpr NumericalDataType convertTypeToTensorType() {
        if constexpr (std::is_same_v<U, float>) {
            return NumericalDataType::kFloat32;
        } else if constexpr (std::is_same_v<U, double>) {
            return NumericalDataType::kFloat64;
        } else if constexpr (std::is_same_v<U, int32_t>) {
            return NumericalDataType::kInt32;
        } else if constexpr (std::is_same_v<U, int64_t>) {
            return NumericalDataType::kInt64;
        } else if constexpr (std::is_same_v<U, uint8_t>) {
            return NumericalDataType::kUInt8;
        } else if constexpr (std::is_same_v<U, uint32_t>) {
            return NumericalDataType::kUInt32;
        } else {
            static_assert(always_false_v<U>, "Unsupported data type");
        }
    }

    // Scalar conversion methods
    virtual Scalar toScalar() const = 0;
    virtual Derived& fromScalar(const Scalar& scalar) = 0;

protected:
    ~ITensor() = default;  // 保护析构函数防止直接删除基类指针
};

template<typename Derived>
static std::ostream& operator<<(std::ostream & out, const ITensor<Derived>& t) {
    t.print(out);
    return out;
}

} // namespace core

} // namespace cuda_simulator


#endif