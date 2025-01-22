#pragma once

#include <cstdint>
#include <type_traits>

namespace cuda_simulator {
namespace core {

enum class NumericalDataType {
  kFloat32,
  kFloat64,
  kInt32,
  kInt64,
  kUInt8,
  kUInt32
};

template <typename T> inline constexpr bool always_false_v = false;

template <NumericalDataType T> struct dtype_converter {

  static auto impl_cast() {
    if constexpr (T == NumericalDataType::kFloat32) {
      return float();
    } else if constexpr (T == NumericalDataType::kFloat64) {
      return double();
    } else if constexpr (T == NumericalDataType::kInt32) {
      return int32_t();
    } else if constexpr (T == NumericalDataType::kInt64) {
      return int64_t();
    } else if constexpr (T == NumericalDataType::kUInt8) {
      return uint8_t();
    } else if constexpr (T == NumericalDataType::kUInt32) {
      return uint32_t();
    }
  }

  using value_type = decltype(impl_cast());
};

template <typename U> struct dtype_rev_converter {
  static constexpr auto cast() {
    if constexpr (std::is_same_v<U, float>) {
      return NumericalDataType::kFloat32;
    } else if constexpr (std::is_same_v<U, double>) {
      return NumericalDataType::kFloat64;
    } else if constexpr (std::is_same_v<U, int32_t>) {
      return NumericalDataType::kInt32;
    } else if constexpr (std::is_same_v<U, int64_t>) {
      return NumericalDataType::kInt64;
    } else if constexpr (std::is_same_v<U, uint8_t>) {
      return NumericalDataType::kUInt8;
    } else if constexpr (std::is_same_v<U, uint32_t>) {
      return NumericalDataType::kUInt32;
    } else {
      static_assert(always_false_v<U>, "Unsupported data type");
    }
  }
};

class Scalar {
public:
  using Type = NumericalDataType;

  // Constructors
  Scalar() : type_(Type::kFloat32), float_value_(0.0f) {
  }
  Scalar(float v) : type_(Type::kFloat32), float_value_(v) {
  }
  Scalar(double v) : type_(Type::kFloat64), double_value_(v) {
  }
  Scalar(int32_t v) : type_(Type::kInt32), int32_value_(v) {
  }
  Scalar(int64_t v) : type_(Type::kInt64), int64_value_(v) {
  }
  Scalar(uint8_t v) : type_(Type::kUInt8), uint8_value_(v) {
  }
  Scalar(uint32_t v) : type_(Type::kUInt32), uint32_value_(v) {
  }

  // NumericalDataType checking
  inline bool isFloatingPoint() const {
    return type_ == Type::kFloat32 || type_ == Type::kFloat64;
  }

  inline bool isIntegral() const {
    return type_ == Type::kInt32 || type_ == Type::kInt64;
  }

inline Type type() const {
    return type_;
  }

  template<typename T>
  inline T to() const {
    switch (type_) {
    case Type::kFloat32:
      return float_value_;
    case Type::kFloat64:
      return double_value_;
    case Type::kInt32:
      return int32_value_;
    case Type::kInt64:
      return int64_value_;
    case Type::kUInt8:
      return uint8_value_;
    case Type::kUInt32:
      return uint32_value_;
    }
    return T();
  }

  // Value getters
  inline float toFloat() const { return to<float>(); }
  inline double toDouble() const { return to<double>(); }
  inline int32_t toInt32() const { return to<int32_t>(); }
  inline int64_t toInt64() const { return to<int64_t>(); }
  inline uint8_t toUInt8() const { return to<uint8_t>(); }
  inline uint32_t toUInt32() const { return to<uint32_t>(); }

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
