#ifndef __ITENSOR_H__
#define __ITENSOR_H__

#include <memory>
#include <vector>
#include <string>
#include <typeindex>
#include <any>
namespace RSG_SIM
{

    enum class TensorDataType
    {
        kFloat32,
        kFloat64,
        kInt32,
        kInt64,
        kUInt8
    };

    class ITensor
    {
    public:
        virtual ~ITensor() = default;

        // 基础信息
        virtual std::vector<int64_t> shape() const = 0;
        virtual size_t elemCount() const = 0;
        virtual size_t elemSize() const = 0;
        virtual size_t dim() const = 0;
        virtual TensorDataType getTensorDataType() const = 0;

        // 数据访问
        virtual void *ptr() = 0;
        virtual const void *ptr() const = 0;
    };

} // namespace RSG_SIM

#endif