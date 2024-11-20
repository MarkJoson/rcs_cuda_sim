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
    
    // 形状操作
    virtual void reshape(const std::vector<int64_t>& shape) = 0;
    virtual void resize(const std::vector<int64_t>& shape) = 0;
    virtual void squeeze(int dim = -1) = 0;
    virtual void unsqueeze(int dim) = 0;
    virtual void flatten(int start_dim = 0, int end_dim = -1) = 0;
    virtual void transpose(int dim0, int dim1) = 0;
    virtual void permute(const std::vector<int64_t>& dims) = 0;
    
    // 切片和索引
    virtual std::unique_ptr<ITensor> slice(
        int dim, 
        int64_t start, 
        int64_t end, 
        int64_t step = 1) const = 0;
    virtual void select(int dim, int64_t index) = 0;
    virtual void index(const std::vector<ITensor*>& indices) = 0;
    
    // 数学操作
    virtual void abs() = 0;
    virtual void clip(const void* min_val, const void* max_val) = 0;
    virtual void sqrt() = 0;
    virtual void pow(double exponent) = 0;
    virtual void exp() = 0;
    virtual void log() = 0;
    
    // 统计操作
    virtual std::unique_ptr<ITensor> sum(int dim = -1, bool keepdim = false) const = 0;
    virtual std::unique_ptr<ITensor> mean(int dim = -1, bool keepdim = false) const = 0;
    virtual std::unique_ptr<ITensor> std(int dim = -1, bool keepdim = false) const = 0;
    virtual std::unique_ptr<ITensor> var(int dim = -1, bool keepdim = false) const = 0;
    virtual std::unique_ptr<ITensor> max(int dim = -1, bool keepdim = false) const = 0;
    virtual std::unique_ptr<ITensor> min(int dim = -1, bool keepdim = false) const = 0;
    virtual std::unique_ptr<ITensor> argmax(int dim = -1, bool keepdim = false) const = 0;
    virtual std::unique_ptr<ITensor> argmin(int dim = -1, bool keepdim = false) const = 0;
    
    // 比较操作
    virtual void eq(const ITensor& other) = 0;
    virtual void ne(const ITensor& other) = 0;
    virtual void gt(const ITensor& other) = 0;
    virtual void lt(const ITensor& other) = 0;
    virtual void ge(const ITensor& other) = 0;
    virtual void le(const ITensor& other) = 0;
};

} // namespace RSG_SIM

#endif