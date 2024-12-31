#ifndef CUDASIM_MESSAGEBUS_HH
#define CUDASIM_MESSAGEBUS_HH

#include <string>
#include <optional>
#include "Component.hh"
#include "storage/ITensor.h"

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

class MessageBus
{
public:
    MessageBus() {}
    virtual ~MessageBus() {}

    void registerNode(ComponentBase* component);
    void registerInput(
        ComponentBase* component,
        const std::string name,
        const std::vector<int64_t> shape,
        int history_offset = 0,
        ITensor* history_padding_val = nullptr,
        ReduceMethod reduce_method = ReduceMethod::STACK
        ) {}
    void registerOutput() {}
};

} // namespace core
} // namespace cuda_simulator


#endif // CUDASIM_MESSAGEBUS_HH