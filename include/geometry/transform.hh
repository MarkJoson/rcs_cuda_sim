#ifndef CUDASIM_GEOMETRY_TRANSFORM_HH
#define CUDASIM_GEOMETRY_TRANSFORM_HH

#pragma once
#include <cmath>
#include "geometry_types.hh"

namespace cuda_simulator {
namespace geometry {

class Rotation2D {
public:
    Rotation2D(float angle = 0.0f)
        : angle_(angle)
        , s_(std::sin(angle))
        , c_(std::cos(angle)) {}

    Rotation2D mulRotation(const Rotation2D& rot) const {
        return Rotation2D(
            angle_ + rot.angle_,
            s_ * rot.c_ + c_ * rot.s_,
            c_ * rot.c_ - s_ * rot.s_
        );
    }

    float angle() const { return angle_; }
    float sin() const { return s_; }
    float cos() const { return c_; }

private:
    Rotation2D(float angle, float s, float c)
        : angle_(angle), s_(s), c_(c) {}

    float angle_;
    float s_;
    float c_;
};

class Transform2D {
public:
    Transform2D(const Vector2& position = Vector2::Zero(),
               const Rotation2D& rotation = Rotation2D())
        : position_(position)
        , rotation_(rotation) {}

    Vector2 localPointTransform(const Vector2& point) const {
        Vector2 result;
        result.x() = point.x() * rotation_.cos() - point.y() * rotation_.sin();
        result.y() = point.x() * rotation_.sin() + point.y() * rotation_.cos();
        result += position_;
        return result;
    }

    Vector2 inverseTransformPoint(const Vector2& point) const {
        Vector2 centered = point - position_;
        Vector2 result;
        result.x() = centered.x() * rotation_.cos() + centered.y() * rotation_.sin();
        result.y() = -centered.x() * rotation_.sin() + centered.y() * rotation_.cos();
        return result;
    }

    Transform2D mulTransform(const Transform2D& transform) const {
        return Transform2D(
            position_ + transform.position_,
            rotation_.mulRotation(transform.rotation_)
        );
    }

    const Vector2& pos() const { return position_; }
    const Rotation2D& rot() const { return rotation_; }

private:
    Vector2 position_;
    Rotation2D rotation_;
};

} // namespace geometry
} // namespace cuda_simulator

#endif // CUDASIM_GEOMETRY_TRANSFORM_HH
