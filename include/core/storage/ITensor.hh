#ifndef __ITENSOR_H__
#define __ITENSOR_H__

#include <ostream>
#include <vector>
#include <iostream>

namespace cuda_simulator
{
namespace core
{

enum class TensorDataType
{
    kFloat32,
    kFloat64,
    kInt32,
    kInt64,
    kUInt8
};


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
    virtual TensorDataType dtype() const = 0;

    // raw data access
    virtual void* data() = 0;
    virtual const void* data() const = 0;

    //
    virtual void print(std::ostream &out) const = 0;
    virtual std::string toString() const = 0;

    //
    virtual void zero() = 0;

    // template<typename T>
    // T item() const { return derived().template item_impl<T>(); }
    template<typename T>
    float item() const { return derived().item_float_impl(); }


    // 返回新Tensor的操作
    Derived add(const Derived& other) const { return derived().add_impl(other); }
    Derived sub(const Derived& other) const { return derived().sub_impl(other); }
    Derived mul(const Derived& other) const { return derived().mul_impl(other); }
    Derived div(const Derived& other) const { return derived().div_impl(other); }

    // 运算符重载
    Derived operator+(const Derived& other) const { return add(other); }
    Derived operator-(const Derived& other) const { return sub(other); }
    Derived operator*(const Derived& other) const { return mul(other); }
    Derived operator/(const Derived& other) const { return div(other); }

    // 复合赋值操作符
    Derived& operator+=(const Derived& other) { return derived()->add_inplace_impl(other); }
    Derived& operator-=(const Derived& other) { return derived()->sub_inplace_impl(other); }
    Derived& operator*=(const Derived& other) { return derived()->mul_inplace_impl(other); }
    Derived& operator/=(const Derived& other) { return derived()->div_inplace_impl(other); }

    // 位运算操作符
    Derived operator~() const { return static_cast<const Derived*>(this)->bitwise_not_impl(); }
    Derived operator-() const { return static_cast<const Derived*>(this)->neg_impl(); }
    Derived& operator&=(const Derived& other) { return derived()->bitwise_and_inplace_impl(other); }
    Derived& operator|=(const Derived& other) { return derived()->bitwise_or_inplace_impl(other); }

    // 索引操作
    Derived slice(int64_t dim, int64_t start, int64_t end) const { return derived().slice_impl(dim, start, end); }
    Derived operator[](int64_t index) const { return derived().index_impl(index); }
    Derived operator[](const std::vector<int64_t>& indices) const { return derived().index_impl(indices); }


    template<typename T>
    T* typed_data() { return static_cast<T*>(this->data()); }

    template<typename T>
    const T* typed_data() const { return static_cast<const T*>(this->data()); }

    // 工具方法
    template<typename U>
    static constexpr TensorDataType convertTypeToTensorType() {
        if constexpr (std::is_same_v<U, float>) return TensorDataType::kFloat32;
        if constexpr (std::is_same_v<U, double>) return TensorDataType::kFloat64;
        if constexpr (std::is_same_v<U, int32_t>) return TensorDataType::kInt32;
        if constexpr (std::is_same_v<U, int64_t>) return TensorDataType::kInt64;
        return TensorDataType::kUInt8;
    }

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