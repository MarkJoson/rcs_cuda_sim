#ifndef CUDA_SIMULATOR_GEOMETRY_TYPES_HH
#define CUDA_SIMULATOR_GEOMETRY_TYPES_HH

#pragma once
#include <Eigen/Dense>

namespace cuda_simulator {
namespace geometry {

enum class ShapeType {
    POLYGON,
    CIRCLE,
};

enum class ObjectType {
    Static,
    Dynamic
};

// 2D向量和变换相关的类型定义
using Vector2 = Eigen::Vector2d;
using Matrix2 = Eigen::Matrix2d;

} // namespace geometry
} // namespace cuda_simulator

#endif // CUDA_SIMULATOR_GEOMETRY_TYPES_HH
