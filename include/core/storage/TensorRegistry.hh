#ifndef TENSOR_REGISTRY_H
#define TENSOR_REGISTRY_H

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "GTensorConfig.hh"


namespace cuda_simulator {
namespace core {

// 单例模式
class TensorRegistry {
public:
    // 删除拷贝和移动操作
    TensorRegistry(const TensorRegistry&) = delete;
    TensorRegistry& operator=(const TensorRegistry&) = delete;
    TensorRegistry(TensorRegistry&&) = delete;
    TensorRegistry& operator=(TensorRegistry&&) = delete;

    // 创建张量接口，返回刚刚创建Tensor的引用
    GTensor& createTensor(const std::string& uri, const TensorShape& shape, NumericalDataType dtype=NumericalDataType::kFloat32, DeviceType device_type=DeviceType::kCUDA) {
        tensors.insert(std::make_pair(uri, GTensor(shape, dtype, device_type)));
        return tensors.at(uri);
    }

    template<typename T>
    GTensor& createTensor(const std::string& uri, const TensorShape& shape, DeviceType device_type=DeviceType::kCUDA) {
        auto dtype = GTensor::convertTypeToTensorType<T>();
        return createTensor(uri, shape, dtype, device_type);
    }

    void createTensor(GTensor& target, const std::string& uri, const TensorShape& shape, NumericalDataType dtype=NumericalDataType::kFloat32, DeviceType device_type=DeviceType::kCUDA) {
        target.bindTensorRef(createTensor(uri, shape, dtype, device_type));
    }

    // 对已经定义的GTensor直接使用赋值运算符，不会使新GTensor的内部impl指针指向在TensorRegistry中的真实数据
    // 因此如果对已有符号赋值，需要使用带tensor参数的方法
    template<typename T>
    void createTensor(GTensor& target, const std::string& uri, const TensorShape& shape, DeviceType device_type=DeviceType::kCUDA) {
        auto dtype = GTensor::convertTypeToTensorType<T>();
        target.bindTensorRef(createTensor(uri, shape, dtype, device_type));
    }

    // 获取张量
    GTensor& getTensor(const std::string& uri) {
        auto it = tensors.find(uri);
        if (it == tensors.end()) {
            throw std::runtime_error("Tensor not found: " + uri);
        }
        return it->second;
    }

    const GTensor& getTensor(const std::string& uri) const {
        auto it = tensors.find(uri);
        if (it == tensors.end()) {
            throw std::runtime_error("Tensor not found: " + uri);
        }
        return it->second;
    }

    // 移除张量
    void removeTensor(const std::string &uri) { tensors.erase(uri); }

    // 批量操作
    std::vector<GTensor> getTensorsByPrefix(const std::string& prefix) {
        std::vector<GTensor> result;
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

    static TensorRegistry& getInstance() {
        static TensorRegistry instance;
        return instance;
    }

private:
    TensorRegistry() = default;  // 私有构造函数
    ~TensorRegistry() = default;

private:
    std::unordered_map<std::string, GTensor> tensors;
};

} // namespace core
} // namespace cuda_simulator

#endif // TENSOR_REGISTRY_H