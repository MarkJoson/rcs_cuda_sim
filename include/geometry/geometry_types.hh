#ifndef CUDA_SIMULATOR_GEOMETRY_TYPES_HH
#define CUDA_SIMULATOR_GEOMETRY_TYPES_HH

#pragma once
// #include <Eigen/Eigen>
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

struct Vector2 {
    double x;
    double y;

    Vector2() : x(0), y(0) {}
    Vector2(double x, double y) : x(x), y(y) {}

    Vector2 operator+(const Vector2& other) const {
        return Vector2(x + other.x, y + other.y);
    }

    Vector2 operator-(const Vector2& other) const {
        return Vector2(x - other.x, y - other.y);
    }

    Vector2 operator*(double scale) const {
        return Vector2(x * scale, y * scale);
    }

    Vector2 operator/(double scale) const {
        return Vector2(x / scale, y / scale);
    }

    Vector2& operator+=(const Vector2& other) {
        x += other.x;
        y += other.y;
        return *this;
    }

    Vector2& operator-=(const Vector2& other) {
        x -= other.x;
        y -= other.y;
        return *this;
    }

    Vector2& operator*=(double scale) {
        x *= scale;
        y *= scale;
        return *this;
    }

    Vector2& operator/=(double scale) {
        x /= scale;
        y /= scale;
        return *this;
    }

    double dot(const Vector2& other) const {
        return x * other.x + y * other.y;
    }

    double cross(const Vector2& other) const {
        return x * other.y - y * other.x;
    }

    double length() const {
        return std::sqrt(x * x + y * y);
    }

    Vector2 normalized() const {
        double len = length();
        return Vector2(x / len, y / len);
    }

    static Vector2 Zero() { return Vector2(0, 0); }
};


// 2D向量和变换相关的类型定义
// using Vector2 = Eigen::Vector2d;
// using Matrix2 = Eigen::Matrix2d;

} // namespace geometry
} // namespace cuda_simulator

#endif // CUDA_SIMULATOR_GEOMETRY_TYPES_HH
