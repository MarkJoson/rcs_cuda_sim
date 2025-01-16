#ifndef CUDASIM_CORE_TYPES_HH
#define CUDASIM_CORE_TYPES_HH

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "storage/GTensorConfig.hh"


namespace cuda_simulator
{

namespace core
{

// 枚举定义
enum class ReduceMethod {
    STACK,      // 堆叠
    SUM,        // 求和
    MAX,        // 求最大值
    MIN,        // 求最小值
    AVERAGE     // 求平均值
};

inline std::string reduce_method_to_string(ReduceMethod method) {
    switch (method) {
        case ReduceMethod::STACK: return "STACK";
        case ReduceMethod::SUM: return "SUM";
        case ReduceMethod::MAX: return "MAX";
        case ReduceMethod::MIN: return "MIN";
        case ReduceMethod::AVERAGE: return "AVERAGE";
        default: return "UNKNOWN";
    }
}

using NodeId = std::uint32_t;
using NodeName = std::string;
using NodeNameRef = std::string_view;
using NodeTag = std::string;
using NodeTagRef = std::string_view;

using MessageId = std::uint32_t;
using MessageName = std::string;
using MessageNameRef = std::string_view;
using MessageShape = std::vector<int64_t>;

using NodeExecInputType = std::unordered_map<MessageNameRef, const std::vector<TensorHandle>>;
using NodeExecOutputType = std::unordered_map<MessageNameRef, TensorHandle>;
using NodeExecStateType = std::unordered_map<MessageNameRef, TensorHandle>;


// 消息形状引用，该类没有设置对数据的检查，使用时需要保证数据的有效性！！！
class MessageShapeRef {
public:
    using iterator = int64_t*;
    using const_iterator = const int64_t*;

    MessageShapeRef(const MessageShapeRef &other) : shape_(other.shape_), dim_(other.dim_) {}
    MessageShapeRef(MessageShapeRef &&other) : shape_(other.shape_), dim_(other.dim_) {}
    MessageShapeRef(const MessageShape &shape) : shape_(shape.data()) {
        dim_ = shape.size();
    }

    MessageShapeRef(int64_t *shape) : shape_(shape) {
        // 数组输入时，以0为终止符
        for (dim_ = 0; shape[dim_] != 0; dim_++);
    }

    MessageShapeRef& operator=(const std::vector<int64_t> &shape) {
        shape_ = shape.data();
        dim_ = shape.size();
        return *this;
    }

    MessageShapeRef& operator=(const MessageShapeRef &other) {
        shape_ = other.shape_;
        dim_ = other.dim_;
        return *this;
    }

    MessageShapeRef& operator=(MessageShapeRef &&other) {
        shape_ = other.shape_;
        dim_ = other.dim_;
        return *this;
    }

    void copyTo(std::vector<int64_t> &shape) const {
        shape.assign(shape_, shape_ + dim_);
    }

    operator std::vector<int64_t>() const {
        return std::vector<int64_t>(shape_, shape_ + dim_);
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

    const_iterator begin() const { return shape_; }
    const_iterator end() const { return shape_ + dim_; }

    const_iterator cbegin() const { return shape_; }
    const_iterator cend() const { return shape_ + dim_; }

private:
    int64_t dim_;
    const int64_t *shape_;
};

} // namespace core
} // namespace cuda_simulator

#endif // CUDASIM_CORE_TYPES_HH