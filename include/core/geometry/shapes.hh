#ifndef CUDASIM_GEOMETRY_SHAPES_HH
#define CUDASIM_GEOMETRY_SHAPES_HH

#pragma once
#include <vector>
#include <memory>
#include "types.hh"

namespace cuda_simulator {
namespace geometry {

// 基础形状定义
class ShapeDef {
public:
    ShapeDef(float restitution = 1.0f, float friction = 0.0f)
        : restitution_(restitution)
        , friction_(friction) {}

    virtual ~ShapeDef() = default;

    float getRestitution() const { return restitution_; }
    float getFriction() const { return friction_; }

protected:
    float restitution_;
    float friction_;
};

// 圆形
class CircleShapeDef : public ShapeDef {
public:
    CircleShapeDef(const Vector2& center, float radius)
        : center_(center)
        , radius_(radius) {}

    const Vector2& getCenter() const { return center_; }
    float getRadius() const { return radius_; }

private:
    Vector2 center_;
    float radius_;
};

// 多边形
class PolygonShapeDef : public ShapeDef {
public:
    PolygonShapeDef(const std::vector<Vector2>& vertices,
                    const Vector2& centroid,
                    float radius)
        : vertices_(vertices)
        , centroid_(centroid)
        , radius_(radius) {}

    const std::vector<Vector2>& getVertices() const { return vertices_; }
    const Vector2& getCentroid() const { return centroid_; }
    float getRadius() const { return radius_; }

private:
    std::vector<Vector2> vertices_;
    Vector2 centroid_;
    float radius_;
};


} // namespace geometry
} // namespace cuda_simulator

#endif // CUDASIM_GEOMETRY_SHAPES_HH