#ifndef CUDASIM_CORE_TYPES_HH
#define CUDASIM_CORE_TYPES_HH

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "storage/GTensorConfig.hh"
#include "MessageShape.hh"

namespace cuda_simulator {
namespace core {

// 枚举定义
enum class ReduceMethod {
    STACK,      // 堆叠
    SUM,        // 求和
    MAX,        // 求最大值
    MIN,        // 求最小值
    AVERAGE     // 求平均值
};

inline std::string reduce_method_to_string(ReduceMethod method) {
    switch (method) {
        case ReduceMethod::STACK: return "STACK";
        case ReduceMethod::SUM: return "SUM";
        case ReduceMethod::MAX: return "MAX";
        case ReduceMethod::MIN: return "MIN";
        case ReduceMethod::AVERAGE: return "AVERAGE";
        default: return "UNKNOWN";
    }
}

using NodeId = std::uint32_t;
using NodeName = std::string;
using NodeNameRef = std::string_view;
using NodeTag = std::string;
using NodeTagRef = std::string_view;

using NodeExecInputType = std::unordered_map<MessageNameRef, const std::vector<const GTensor*>>;
using NodeExecOutputType = std::unordered_map<MessageNameRef, GTensor*>;
using NodeExecStateType = std::unordered_map<MessageNameRef, GTensor*>;

} // namespace core
} // namespace cuda_simulator

#endif // CUDASIM_CORE_TYPES_HH