#pragma once
#include "types.hpp"
#include <cmath>

namespace physics {

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

    float getAngle() const { return angle_; }
    float getSin() const { return s_; }
    float getCos() const { return c_; }

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
        result.x() = point.x() * rotation_.getCos() - point.y() * rotation_.getSin();
        result.y() = point.x() * rotation_.getSin() + point.y() * rotation_.getCos();
        result += position_;
        return result;
    }

    Vector2 inverseTransformPoint(const Vector2& point) const {
        Vector2 centered = point - position_;
        Vector2 result;
        result.x() = centered.x() * rotation_.getCos() + centered.y() * rotation_.getSin();
        result.y() = -centered.x() * rotation_.getSin() + centered.y() * rotation_.getCos();
        return result;
    }

    Transform2D mulTransform(const Transform2D& transform) const {
        return Transform2D(
            position_ + transform.position_,
            rotation_.mulRotation(transform.rotation_)
        );
    }

    const Vector2& getPosition() const { return position_; }
    const Rotation2D& getRotation() const { return rotation_; }

private:
    Vector2 position_;
    Rotation2D rotation_;
};

} // namespace physics