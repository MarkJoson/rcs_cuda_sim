#pragma once

#include <cstdint>


namespace cuda_simulator {
namespace core {

enum class NumericalDataType
{
    kFloat32,
    kFloat64,
    kInt32,
    kInt64,
    kUInt8,
    kUInt32
};

class Scalar {
public:

    using Type = NumericalDataType;

    // Constructors
    Scalar() : type_(Type::kFloat32), float_value_(0.0f) {}
    Scalar(float v) : type_(Type::kFloat32), float_value_(v) {}
    Scalar(double v) : type_(Type::kFloat64), double_value_(v) {}
    Scalar(int32_t v) : type_(Type::kInt32), int32_value_(v) {}
    Scalar(int64_t v) : type_(Type::kInt64), int64_value_(v) {}
    Scalar(uint8_t v) : type_(Type::kUInt8), uint8_value_(v) {}
    Scalar(uint32_t v) : type_(Type::kUInt32), uint32_value_(v) {}

    // NumericalDataType checking
    bool isFloatingPoint() const {
        return type_ == Type::kFloat32 || type_ == Type::kFloat64;
    }

    bool isIntegral() const {
        return type_ == Type::kInt32 || type_ == Type::kInt64;
    }

    // Value getters
    float toFloat() const {
        switch (type_) {
            case Type::kFloat32: return float_value_;
            case Type::kFloat64: return static_cast<float>(double_value_);
            case Type::kInt32: return static_cast<float>(int32_value_);
            case Type::kInt64: return static_cast<float>(int64_value_);
            case Type::kUInt8: return static_cast<uint8_t>(uint8_value_);
            case Type::kUInt32: return static_cast<uint32_t>(uint32_value_);
        }
        return 0.0f;
    }

    double toDouble() const {
        switch (type_) {
            case Type::kFloat32: return static_cast<double>(float_value_);
            case Type::kFloat64: return double_value_;
            case Type::kInt32: return static_cast<double>(int32_value_);
            case Type::kInt64: return static_cast<double>(int64_value_);
            case Type::kUInt8: return static_cast<uint8_t>(uint8_value_);
            case Type::kUInt32: return static_cast<uint32_t>(uint32_value_);
        }
        return 0.0;
    }

    int32_t toInt32() const {
        switch (type_) {
            case Type::kFloat32: return static_cast<int32_t>(float_value_);
            case Type::kFloat64: return static_cast<int32_t>(double_value_);
            case Type::kInt32: return int32_value_;
            case Type::kInt64: return static_cast<int32_t>(int64_value_);
            case Type::kUInt8: return static_cast<uint8_t>(uint8_value_);
            case Type::kUInt32: return static_cast<uint32_t>(uint32_value_);
        }
        return 0;
    }

    int64_t toInt64() const {
        switch (type_) {
            case Type::kFloat32: return static_cast<int64_t>(float_value_);
            case Type::kFloat64: return static_cast<int64_t>(double_value_);
            case Type::kInt32: return static_cast<int64_t>(int32_value_);
            case Type::kInt64: return int64_value_;
            case Type::kUInt8: return static_cast<uint8_t>(uint8_value_);
            case Type::kUInt32: return static_cast<uint32_t>(uint32_value_);
        }
        return 0;
    }

    uint8_t toUInt8() const {
        switch (type_) {
            case Type::kFloat32: return static_cast<uint8_t>(float_value_);
            case Type::kFloat64: return static_cast<uint8_t>(double_value_);
            case Type::kInt32: return static_cast<uint8_t>(int32_value_);
            case Type::kInt64: return static_cast<uint8_t>(int64_value_);
            case Type::kUInt8: return uint8_value_;
            case Type::kUInt32: return static_cast<uint8_t>(uint32_value_);
        }
        return 0;
    }

    uint32_t toUInt32() const {
        switch (type_) {
            case Type::kFloat32: return static_cast<uint32_t>(float_value_);
            case Type::kFloat64: return static_cast<uint32_t>(double_value_);
            case Type::kInt32: return static_cast<uint32_t>(int32_value_);
            case Type::kInt64: return static_cast<uint32_t>(int64_value_);
            case Type::kUInt8: return static_cast<uint32_t>(uint8_value_);
            case Type::kUInt32: return uint32_value_;
        }
        return 0;
    }

    Type type() const {
        return type_;
    }

private:
    Type type_;
    union {
        float float_value_;
        double double_value_;
        int32_t int32_value_;
        int64_t int64_value_;
        uint8_t uint8_value_;
        uint32_t uint32_value_;
    };
};

} // namespace core
} // namespace cuda_simulator
