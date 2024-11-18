#ifndef __TENSOR_REGISTRY_H__
#define __TENSOR_REGISTRY_H__

#include <string>
#include <memory>
#include <unordered_map>
#include <vector>
#include "TensorMeta.h"
#include "ITensor.h"

namespace RSG_SIM {


template<typename T> class GTensor;
class TensorRegistry;
class TensorRegistryManager;

class TensorRegistry {
public:
    explicit TensorRegistry(int env_count=16);
    ~TensorRegistry();
    
    // 禁用拷贝
    TensorRegistry(const TensorRegistry&) = delete;
    TensorRegistry& operator=(const TensorRegistry&) = delete;
    
    // 启用移动
    TensorRegistry(TensorRegistry&&) noexcept;
    TensorRegistry& operator=(TensorRegistry&&) noexcept;

    // 创建张量的非模板接口
    ITensor* createTensor(const TensorMeta& meta);
    
    // 创建张量的模板接口
    template<typename T>
    GTensor<T>* createTensor(const std::string& name, const std::vector<int64_t>& shape) {
        auto meta = TensorMeta::create<T>(name, shape);
        auto* tensor = static_cast<GTensor<T>*>(createTensor(meta));
        return tensor;
    }
    
    // 获取张量
    ITensor* getTensor(const std::string& name);
    template<typename T>
    GTensor<T>* getTensor(const std::string& name) {
        auto* tensor = getTensor(name);
        return (tensor && tensor->isTypeMatch(typeid(T))) ?
            static_cast<GTensor<T>*>(tensor) : nullptr;
    }
    
    // 移除张量
    void removeTensor(const std::string& name);
    
    // 批量操作
    std::vector<ITensor*> getTensorsByPrefix(const std::string& prefix);
    void removeTensorsByPrefix(const std::string& prefix);
    
    // 信息查询
    size_t size() const;
    bool exists(const std::string& name) const;
    std::vector<std::string> getAllNames() const;
    
    // 环境信息
    int getEnvCount() const { return env_count_; }

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
    int env_count_;
};

} // namespace RSG_SIM

#endif