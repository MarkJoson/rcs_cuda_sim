#ifndef CUDASIM_GEOMETRY_SHAPES_HH
#define CUDASIM_GEOMETRY_SHAPES_HH

#include <initializer_list>
#pragma once
#include <vector>
#include <memory>
#include "geometry_types.hh"

namespace cuda_simulator {
namespace geometry {

// 基础形状定义
struct ShapeDef {
public:
    ShapeDef(ShapeType type, float restitution = 1.0f, float friction = 0.0f)
        : type(type)
        , restitution(restitution)
        , friction(friction) {}

    ShapeType type;
    float restitution = 1.0f;
    float friction = 0.0f;
};

// 圆形
struct CircleShapeDef : public ShapeDef {
public:
    CircleShapeDef(const Vector2& center, float radius)
        : ShapeDef(ShapeType::CIRCLE)
        , center(center)
        , radius(radius) {}

    Vector2 center;
    float radius;
};

// 多边形
struct PolygonShapeDef : public ShapeDef {
public:
    PolygonShapeDef(const std::vector<Vector2>& vertices,
                    const Vector2& centroid,
                    float radius)
        : ShapeDef(ShapeType::POLYGON)
        , vertices(vertices)
        , centroid(centroid)
        , radius(radius) {}

    std::vector<Vector2> vertices;
    Vector2 centroid;
    float radius;
};


} // namespace geometry
} // namespace cuda_simulator

#endif // CUDASIM_GEOMETRY_SHAPES_HH