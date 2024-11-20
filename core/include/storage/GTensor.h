#ifndef __GTENSOR_H__
#define __GTENSOR_H__

#include "GTensorBase.h"
#include <cassert>

namespace RSG_SIM {

template<typename T>
class GTensor : public GTensorBase {
public:
    GTensor(const std::string& name, const std::vector<int64_t>& shape)
        : GTensorBase(TensorMeta::create<T>(name, shape)) {}
        
    // 类型安全的数据访问
    T* data() { return static_cast<T*>(ptr()); }
    const T* data() const { return static_cast<const T*>(ptr()); }
    
    // 类型安全的操作
    void fillValue(const T& value) {
        fill(&value);
    }
    
    void clipValue(const T& min_val, const T& max_val) {
        clip(&min_val, &max_val);
    }
    
    // 创建相同类型的新张量
    std::unique_ptr<GTensor<T>> typedSlice(
        int dim, 
        int64_t start, 
        int64_t end, 
        int64_t step = 1) const {
        auto base_slice = slice(dim, start, end, step);
        return std::unique_ptr<GTensor<T>>(
            static_cast<GTensor<T>*>(base_slice.release()));
    }
    
    // 其他类型安全的辅助方法
    std::unique_ptr<GTensor<T>> typedSum(int dim = -1, bool keepdim = false) const {
        auto base_sum = sum(dim, keepdim);
        return std::unique_ptr<GTensor<T>>(
            static_cast<GTensor<T>*>(base_sum.release()));
    }
    
    std::unique_ptr<GTensor<T>> typedMean(int dim = -1, bool keepdim = false) const {
        auto base_mean = mean(dim, keepdim);
        return std::unique_ptr<GTensor<T>>(
            static_cast<GTensor<T>*>(base_mean.release()));
    }
    };

} // namespace RSG_SIM

#endif