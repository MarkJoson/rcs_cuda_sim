#ifndef __TENSOR_REGISTRY_H__
#define __TENSOR_REGISTRY_H__

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "GTensor.hh"
#include "core/core_types.hh"


namespace cuda_simulator {
namespace core {


class TensorRegistry;
class TensorRegistryManager;

class TensorRegistry {
public:
    static TensorRegistry& getInstance() {
        static TensorRegistry instance;
        return instance;
    }

    // 删除拷贝和移动操作
    TensorRegistry(const TensorRegistry&) = delete;
    TensorRegistry& operator=(const TensorRegistry&) = delete;
    TensorRegistry(TensorRegistry&&) = delete;
    TensorRegistry& operator=(TensorRegistry&&) = delete;

    // 创建张量接口
    TensorHandle& createTensor(const std::string& uri, const std::vector<int64_t>& shape, NumericalDataType dtype=NumericalDataType::kFloat32) {
        tensors.insert(std::make_pair(uri, TensorHandle(shape, dtype)));
        return tensors.at(uri);
    }

    template<typename T>
    TensorHandle& createTensor(const std::string& uri, const std::vector<int64_t>& shape) {
        auto dtype = TensorHandle::convertTypeToTensorType<T>();
        return createTensor(uri, shape, dtype);
    }

    // 获取张量
    TensorHandle& getTensor(const std::string& uri) {
        auto it = tensors.find(uri);
        if (it == tensors.end()) {
            throw std::runtime_error("Tensor not found: " + uri);
        }
        return it->second;
    }

    const TensorHandle& getTensor(const std::string& uri) const {
        auto it = tensors.find(uri);
        if (it == tensors.end()) {
            throw std::runtime_error("Tensor not found: " + uri);
        }
        return it->second;
    }

    // 移除张量
    void removeTensor(const std::string &uri) { tensors.erase(uri); }

    // 批量操作
    std::vector<TensorHandle> getTensorsByPrefix(const std::string& prefix) {
        std::vector<TensorHandle> result;
        for (const auto& [uri, tensor] : tensors) {
            if (uri.substr(0, prefix.length()) == prefix) {
                result.push_back(tensor);
            }
        }
        return result;
    }

    void removeTensorsByPrefix(const std::string &prefix) {
        for (auto it = tensors.begin(); it != tensors.end();) {
            if (it->first.substr(0, prefix.length()) == prefix) {
                it = tensors.erase(it);
            } else {
                ++it;
            }
        }
    }

    // 信息查询
    size_t size() const { return tensors.size(); }

    bool exists(const std::string &uri) const {
        return tensors.find(uri) != tensors.end();
    }

    std::vector<std::string> getAllTensorUri() const {
        std::vector<std::string> uris;
        uris.reserve(tensors.size());
        for (const auto &[uri, _] : tensors) {
            uris.push_back(uri);
        }
        return uris;
    }

private:
    TensorRegistry() = default;  // 私有构造函数
    ~TensorRegistry() = default;

private:
    std::unordered_map<std::string, TensorHandle> tensors;
};

} // namespace core
} // namespace cuda_simulator

#endif