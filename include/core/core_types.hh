#ifndef CUDASIM_CORE_TYPES_HH
#define CUDASIM_CORE_TYPES_HH

#include <cstdint>
#include <string>
#include <string_view>

namespace cuda_simulator
{

namespace core
{

// 枚举定义
enum class ReduceMethod {
    STACK,      // 堆叠
    REPLACE,    // 替换
    SUM,        // 求和
    MAX,        // 求最大值
    MIN,        // 求最小值
    AVERAGE     // 求平均值
};

using NodeId = std::uint32_t;
using NodeName = std::string;
using NodeNameRef = std::string_view;
using NodeTag = std::string;
using NodeTagRef = std::string_view;

using MessageId = std::uint32_t;
using MessageName = std::string;
using MessageNameRef = std::string_view;


}
}


#endif // CUDASIM_CORE_TYPES_HH