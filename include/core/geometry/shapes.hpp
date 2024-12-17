#pragma once
#include "types.hpp"
#include <vector>
#include <memory>

namespace physics {

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

// 线段
class SegmentShapeDef : public ShapeDef {
public:
    SegmentShapeDef(const Vector2& point1, const Vector2& point2)
        : point1_(point1)
        , point2_(point2) {}

    const Vector2& getPoint1() const { return point1_; }
    const Vector2& getPoint2() const { return point2_; }

private:
    Vector2 point1_;
    Vector2 point2_;
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

// 链条段
class ChainSegmentDef : public ShapeDef {
public:
    ChainSegmentDef(int chainId,
                    const Vector2& ghostTail,
                    const std::shared_ptr<SegmentShapeDef>& segment,
                    const Vector2& ghostHead)
        : chainId_(chainId)
        , ghostTail_(ghostTail)
        , segment_(segment)
        , ghostHead_(ghostHead) {}

    int getChainId() const { return chainId_; }
    const Vector2& getGhostTail() const { return ghostTail_; }
    const std::shared_ptr<SegmentShapeDef>& getSegment() const { return segment_; }
    const Vector2& getGhostHead() const { return ghostHead_; }

private:
    int chainId_;
    Vector2 ghostTail_;
    std::shared_ptr<SegmentShapeDef> segment_;
    Vector2 ghostHead_;
};

} // namespace physics