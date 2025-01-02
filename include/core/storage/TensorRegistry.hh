#ifndef __TENSOR_REGISTRY_H__
#define __TENSOR_REGISTRY_H__

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "GTensor.hh"
#include "ITensor.hh"

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


    // 创建张量的非模板接口
    ITensor *createTensor(const std::string &uri, const std::vector<int64_t> &shape, TensorDataType dtype) {
        auto tensor = std::make_unique<GTensorTorchWrapper>(shape, dtype);
        auto *ptr = tensor.get();

        tensors[uri] = std::move(tensor);
        return ptr;
    }

    // 创建张量的模板接口
    template <typename T>
    GTensor<T> *createTensor(const std::string &uri, const std::vector<int64_t> &shape) {
        TensorDataType type = GTensorTorchWrapper::getTensorDataType<T>();
        auto *tensor = static_cast<GTensor<T> *>(createTensor(uri, shape, type));
        return tensor;
    }

    // 获取张量
    ITensor *getTensor(const std::string &uri) {
        auto it = tensors.find(uri);
        return it != tensors.end() ? it->second.get() : nullptr;
    }

    template <typename T> GTensor<T> *getTensor(const std::string &uri) {
        auto *tensor = getTensor(uri);
        return static_cast<GTensor<T> *>(tensor);
    }

    // 移除张量
    void removeTensor(const std::string &uri) { tensors.erase(uri); }

    // 批量操作
    std::vector<ITensor *> getTensorsByPrefix(const std::string &prefix) {
        std::vector<ITensor *> result;
        for (const auto &[uri, tensor] : tensors) {
            if (uri.substr(0, prefix.length()) == prefix) {
                result.push_back(tensor.get());
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
    std::unordered_map<std::string, std::unique_ptr<ITensor>> tensors;
};

} // namespace core
} // namespace cuda_simulator

#endif