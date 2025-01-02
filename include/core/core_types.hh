#ifndef CUDASIM_CORE_TYPES_HH
#define CUDASIM_CORE_TYPES_HH

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>


namespace cuda_simulator
{

namespace core
{

// 枚举定义
enum class ReduceMethod {
    STACK,      // 堆叠
    REPLACE,    // 替换
    SUM,        // 求和
    MAX,        // 求最大值
    MIN,        // 求最小值
    AVERAGE     // 求平均值
};

using NodeId = std::uint32_t;
using NodeName = std::string;
using NodeNameRef = std::string_view;
using NodeTag = std::string;
using NodeTagRef = std::string_view;

using MessageId = std::uint32_t;
using MessageName = std::string;
using MessageNameRef = std::string_view;
using MessageShape = std::vector<int64_t>;

class MessageShapeRef {
public:
    MessageShapeRef(std::vector<int64_t> &shape) : shape_(shape.data()) {
        dim_ = shape.size();
    }

    MessageShapeRef(int64_t *shape) : shape_(shape) {
        // 以0为终止符
        for (dim_ = 0; shape[dim_] != 0; dim_++);
    }

    int64_t operator[](int index) const {
        return shape_[index];
    }

    int64_t size() const {
        return dim_;
    }

    bool operator==(const MessageShapeRef &other) const {
        if (dim_ != other.dim_) {
            return false;
        }

        for (int i = 0; i < dim_; i++) {
            if (shape_[i] != other.shape_[i]) {
                return false;
            }
        }

        return true;
    }
private:
    int64_t dim_;
    int64_t *shape_;
};


}
}


#endif // CUDASIM_CORE_TYPES_HH