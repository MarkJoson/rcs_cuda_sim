#ifndef __GTENSOR_TORCH_WRAPPER_H__
#define __GTENSOR_TORCH_WRAPPER_H__

#include "ITensor.h"

namespace cuda_simulator
{
namespace core
{


namespace internal
{
    class TorchTensorImpl;
}

class GTensorTorchWrapper : public ITensor
{
public:
    template <typename T>
    static constexpr TensorDataType getTensorDataType()
    {
        using DT = TensorDataType;
        if constexpr (std::is_same_v<T, float>)
            return DT::kFloat32;
        else if constexpr (std::is_same_v<T, double>)
            return DT::kFloat64;
        else if constexpr (std::is_same_v<T, int32_t>)
            return DT::kInt32;
        else if constexpr (std::is_same_v<T, int64_t>)
            return DT::kInt64;
        else
            return DT::kUInt8;

        // static_assert(always_false<T>::value, "Unsupported type");
    }

    template <typename T>
    explicit GTensorTorchWrapper(const std::vector<int64_t> &shape)
        : GTensorTorchWrapper(shape, getTensorDataType<T>()) {}
    explicit GTensorTorchWrapper(const std::vector<int64_t> &shape, TensorDataType dtype);
    ~GTensorTorchWrapper() override;

    // 基础信息实现
    std::vector<int64_t> shape() const override;
    size_t elemCount() const override;
    size_t elemSize() const override;
    size_t dim() const override;
    TensorDataType getTensorDataType() const override;

    // 数据访问实现
    void *ptr() override;
    const void *ptr() const override;

    // 数据操作实现
    void zero();
    template <typename T>
    void fill(const T &value)
    {
        fill(static_cast<const void *>(&value), getTensorDataType<T>());
    }

protected:
    void fill(const void *value, TensorDataType dtype);
    std::unique_ptr<internal::TorchTensorImpl> impl_;
};


} // namespace core
} // namespace cuda_simulator

#endif