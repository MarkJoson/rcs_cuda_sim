#ifndef __ITENSOR_H__
#define __ITENSOR_H__

#include <memory>
#include <vector>
#include <string>
#include <typeindex>

namespace RSG_SIM {

enum class TensorDataType {
    kFloat32,
    kFloat64,
    kInt32,
    kInt64,
    kUInt8
    // 可以根据需要添加更多类型
};

template<typename T>
struct always_false : std::false_type {};

struct TensorMeta {
    std::vector<int64_t> shape;
    TensorDataType dtype;
    size_t type_size;
    std::type_index type_info;
    
    template<typename T>
    static TensorMeta create(const std::vector<int64_t>& shape) {
        return {
            shape,
            getDataType<T>(),
            sizeof(T),
            std::type_index(typeid(T))
        };
    }

    static TensorDataType getDataType(const std::type_index &ti) {
        using DT = TensorDataType;
        if (ti == typeid(float))
            return DT::kFloat32;
        else if (ti == typeid(double))
            return DT::kFloat64;
        else if (ti == typeid(int32_t))
            return DT::kInt32;
        else if (ti == typeid(int64_t))
            return DT::kInt64;
        else
            return DT::kUInt8;
    }

    template<typename T>
    static constexpr TensorDataType getDataType() {
        using DT = TensorDataType;
        if constexpr (std::is_same_v<T, float>)
            return DT::kFloat32;
        else if constexpr (std::is_same_v<T, double>)
            return DT::kFloat64;
        else if constexpr (std::is_same_v<T, int32_t>)
            return DT::kInt32;
        else if constexpr (std::is_same_v<T, int64_t>)
            return DT::kInt64;
        else
            return DT::kUInt8;
        
        // static_assert(always_false<T>::value, "Unsupported type");
    }

};

class ITensor {
public:
    virtual ~ITensor() = default;
    
    // 元信息接口
    virtual const TensorMeta& meta() const = 0;
    virtual bool isTypeMatch(const std::type_index& type) const = 0;
    
    // 基础信息
    virtual const std::vector<int64_t>& shape() const = 0;
    virtual size_t elemCount() const = 0;
    virtual size_t elemSize() const = 0;
    virtual size_t dim() const = 0;
    
    // 数据访问
    virtual void* ptr() = 0;
    virtual const void* ptr() const = 0;
    
    // 设备操作
    virtual bool isOnCPU() const = 0;
    virtual bool isOnGPU() const = 0;
    virtual void toCPU() = 0;
    virtual void toGPU() = 0;
    
    // 数据操作
    virtual void zero() = 0;
    virtual void fill(const void* value) = 0;
};

} // namespace RSG_SIM

#endif