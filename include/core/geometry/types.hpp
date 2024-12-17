#pragma once
#include <core/types.hpp>
#include <Eigen/Dense>
#include <variant>

namespace physics {

enum class ShapeType {
    Circle,
    Segment,
    Polygon,
    Chain
};

enum class ObjectType {
    Static,
    Dynamic
};

// 2D向量和变换相关的类型定义
using Vector2 = Eigen::Vector2d;
using Matrix2 = Eigen::Matrix2d;

} // namespace physics