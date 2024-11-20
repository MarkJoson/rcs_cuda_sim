#ifndef __TENSOR_REGISTRY_H__
#define __TENSOR_REGISTRY_H__

#include <string>
#include <memory>
#include <unordered_map>
#include <vector>
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
    ITensor* createTensor(const std::string &uri, const std::vector<int64_t> &shape, TensorDataType dtype);
    
    // 创建张量的模板接口
    template<typename T>
    GTensor<T>* createTensor(const std::string &uri, const std::vector<int64_t>& shape) {
        auto* tensor = static_cast<GTensor<T>*>(createTensor(uri, shape, GTensor<T>::getTensorDataType()));
        return tensor;
    }
    
    // 获取张量
    ITensor* getTensor(const std::string& uri);
    
    template<typename T>
    GTensor<T>* getTensor(const std::string& uri) {
        auto* tensor = getTensor(uri);
        return static_cast<GTensor<T>*>(tensor);
    }
    
    // 移除张量
    void removeTensor(const std::string& uri);
    
    // 批量操作
    std::vector<ITensor*> getTensorsByPrefix(const std::string& prefix);
    void removeTensorsByPrefix(const std::string& prefix);
    
    // 信息查询
    size_t size() const;
    bool exists(const std::string& uri) const;
    std::vector<std::string> getAllTensorUri() const;
    
    // 环境信息
    int getEnvCount() const { return env_count_; }

private:
    std::unordered_map<std::string, std::unique_ptr<ITensor>> tensors;
    int env_count_;
};

} // namespace RSG_SIM

#endif