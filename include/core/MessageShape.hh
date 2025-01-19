#ifndef CUDASIM_MESSAGE_SHAPE_HH
#define CUDASIM_MESSAGE_SHAPE_HH

#include <cstdint>
#include <vector>
#include <string>

namespace cuda_simulator {
namespace core {

using MessageId = std::uint32_t;
using MessageName = std::string;
using MessageNameRef = std::string_view;
using MessageShape = std::vector<int64_t>;

using StateName = std::string;
using StateNameRef = std::string_view;
using StateShape = std::vector<int64_t>;

// 消息形状引用，该类没有设置对数据的检查，使用时需要保证数据的有效性！！！
class MessageShapeRef {
public:
    using iterator = int64_t*;
    using const_iterator = const int64_t*;

    MessageShapeRef(const MessageShapeRef &other) : dim_(other.dim_), shape_(other.shape_) {}
    MessageShapeRef(MessageShapeRef &&other) : dim_(other.dim_), shape_(other.shape_) {}
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

    inline void copyTo(std::vector<int64_t> &shape) const {
        shape.assign(shape_, shape_ + dim_);
    }

    inline operator std::vector<int64_t>() const {
        return std::vector<int64_t>(shape_, shape_ + dim_);
    }

    inline int64_t operator[](int index) const {
        return shape_[index];
    }

    inline int64_t size() const {
        return dim_;
    }

    inline bool operator==(const MessageShapeRef &other) const {
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

using StateShapeRef = MessageShapeRef;


} // namespace core
} // namespace cuda_simulator

#endif // CUDASIM_CORE_TYPES_HH