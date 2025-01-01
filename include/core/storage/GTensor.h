#ifndef __GTENSOR_H__
#define __GTENSOR_H__

#include "GTensorTorchWrapper.h"
#include <cassert>

namespace cuda_simulator
{
namespace core
{

template <typename T>
class GTensor : public GTensorTorchWrapper
{
public:
    using value_type = T;
    using iterator = T *;
    using const_iterator = const T *;

    GTensor(const std::vector<int64_t> &shape)
        : GTensorTorchWrapper(shape, getTensorDataType<T>()) {}

    // 类型安全的数据访问
    T *data() { return static_cast<T *>(ptr()); }
    const T *data() const { return static_cast<const T *>(ptr()); }
};

} // namespace core
} // namespace cuda_simulator

#endif