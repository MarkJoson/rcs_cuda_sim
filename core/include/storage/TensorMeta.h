#ifndef __TENSOR_META_H__
#define __TENSOR_META_H__

#include <vector>
#include <string>
#include <typeindex>

namespace RSG_SIM {

enum class TensorDataType {
    kFloat32,
    kFloat64,
    kInt32,
    kInt64,
    // 可以根据需要添加更多类型
};

template<typename T>
struct always_false : std::false_type {};

struct TensorMeta {
    std::string name;
    std::vector<int64_t> shape;
    TensorDataType dtype;
    size_t type_size;
    std::type_index type_info;
    
    template<typename T>
    static TensorMeta create(const std::string& name, const std::vector<int64_t>& shape) {
        return {
            name,
            shape,
            getDataType<T>(),
            sizeof(T),
            std::type_index(typeid(T))
        };
    }
    
private:
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
        return DT::kFloat32;
    
    // static_assert(always_false<T>::value, "Unsupported type");
}
};

} // namespace RSG_SIM

#endif