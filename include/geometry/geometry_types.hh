#ifndef CUDA_SIMULATOR_GEOMETRY_TYPES_HH
#define CUDA_SIMULATOR_GEOMETRY_TYPES_HH

#pragma once
// #include <Eigen/Eigen>
// #include <Eigen/Dense>
#include <string>
#include <iostream>
#include <cmath>

namespace cuda_simulator {
namespace core {
namespace geometry {

enum class ShapeType {
    SIMPLE_POLYGON,
    COMPOSED_POLYGON,
    CIRCLE,
    LINE,
};

enum class ObjectType {
    Static,
    Dynamic
};

template <typename T>
struct Vector2 {
    T x = 0;
    T y = 0;

    constexpr Vector2() = default;
    constexpr Vector2(T x, T y) : x(x), y(y) {}

    template<typename U>
    explicit constexpr Vector2(const Vector2<U>& other) : x(other.x), y(other.y) {}

    constexpr Vector2 operator+(const Vector2& other) const { return Vector2(x + other.x, y + other.y); }
    constexpr Vector2 operator-(const Vector2& other) const { return Vector2(x - other.x, y - other.y); }
    constexpr Vector2 operator*(T scale) const { return Vector2(x * scale, y * scale); }
    constexpr Vector2 operator/(T scale) const { return Vector2(x / scale, y / scale); }

    constexpr Vector2& operator+=(const Vector2& other) {
        x += other.x;
        y += other.y;
        return *this;
    }

    constexpr Vector2& operator-=(const Vector2& other) {
        x -= other.x;
        y -= other.y;
        return *this;
    }

    constexpr Vector2& operator*=(T scale) {
        x *= scale;
        y *= scale;
        return *this;
    }

    constexpr Vector2& operator/=(T scale) {
        x /= scale;
        y /= scale;
        return *this;
    }

    constexpr bool operator==(const Vector2& other) const { return other.x==x && other.y==y; }

    constexpr double dot(const Vector2& other) const { return x * other.x + y * other.y; }
    constexpr double cross(const Vector2& other) const { return x * other.y - y * other.x; }
    double length() const { return std::sqrt(x * x + y * y); }

    Vector2 normalized() const {
        double len = length();
        return Vector2(x / len, y / len);
    }

    constexpr static Vector2 Zero() { return Vector2(); }
};


template <typename T>
static std::ostream &operator<<(std::ostream &out, const Vector2<T> &d) {
    out << " [" << d.x << "," << d.y << "] ";
    return out;
}

template <typename T>
struct Line {
    Vector2<T> start;
    Vector2<T> end;

    constexpr Line() : start(Vector2<T>::Zero()), end(Vector2<T>::Zero()) {}
    constexpr Line(const Vector2<T>& start, const Vector2<T>& end) : start(start), end(end) {}

    constexpr bool operator==(const Line& other) const {
        return start == other.start && end == other.end;
    }
};

// 2D向量和变换相关的类型定义
using Vector2f = Vector2<float>;
using Vector2i = Vector2<int>;
using Linef = Line<float>;


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
    Transform2D(const Vector2f& position = Vector2f::Zero(),
               const Rotation2D& rotation = Rotation2D())
        : position_(position)
        , rotation_(rotation) {}

    Vector2f localPointTransform(const Vector2f& point) const {
        Vector2f result;
        result.x = point.x * rotation_.cos() - point.y * rotation_.sin();
        result.y = point.x * rotation_.sin() + point.y * rotation_.cos();
        result += position_;
        return result;
    }

    Vector2f inverseTransformPoint(const Vector2f& point) const {
        Vector2f centered = point - position_;
        Vector2f result;
        result.x = centered.x * rotation_.cos() + centered.y * rotation_.sin();
        result.y = -centered.x * rotation_.sin() + centered.y * rotation_.cos();
        return result;
    }

    Transform2D mulTransform(const Transform2D& transform) const {
        return Transform2D(
            position_ + transform.position_,
            rotation_.mulRotation(transform.rotation_)
        );
    }

    const Vector2f& pos() const { return position_; }
    const Rotation2D& rot() const { return rotation_; }

private:
    Vector2f position_;
    Rotation2D rotation_;
};


} // namespace geometry
} // namespace core
} // namespace cuda_simulator

#endif // CUDA_SIMULATOR_GEOMETRY_TYPES_HH
