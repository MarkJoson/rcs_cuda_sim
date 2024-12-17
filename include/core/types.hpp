#pragma once
#include <string>
#include <vector>
#include <memory>
#include <map>
#include <queue>
#include <optional>
#include <Eigen/Dense>  // 使用Eigen替代numpy

namespace core {

// 基础类型定义
using EntityId = std::string;
using MessageId = std::string;
using GraphId = std::string;
using ContextId = std::string;
using Vector2D = Eigen::Vector2d;
using Tensor = Eigen::MatrixXf;  // 可以根据需要调整为其他类型

// 前向声明
class Component;
class Context;
class MessageBus;

// 形状定义
using MessageShape = std::vector<int>;

// 枚举定义
enum class ReduceMethod {
    STACK,      // 堆叠
    REPLACE,    // 替换
    SUM,        // 求和
    MAX,        // 求最大值
    MIN,        // 求最小值
    AVERAGE     // 求平均值
};

enum class ContextType {
    SPACE_MANAGER,    // 空间管理器
    TIME_MANAGER,     // 时间管理器
    MESSAGE_BUS,      // 消息总线
    COMPONENT        // 组件
};

} // namespace core