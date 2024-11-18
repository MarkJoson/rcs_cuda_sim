#include "storage/TensorRegistry.h"
#include "storage/GTensor.h"
#include <torch/torch.h>

namespace RSG_SIM {

class TensorRegistry::Impl {
public:
    std::unordered_map<std::string, std::unique_ptr<ITensor>> tensors;
};

TensorRegistry::TensorRegistry(int env_count)
    : impl_(std::make_unique<Impl>())
    , env_count_(env_count) {
}

TensorRegistry::~TensorRegistry() = default;

TensorRegistry::TensorRegistry(TensorRegistry&&) noexcept = default;
TensorRegistry& TensorRegistry::operator=(TensorRegistry&&) noexcept = default;

ITensor* TensorRegistry::createTensor(const TensorMeta& meta) {
    auto full_shape = meta.shape;
    full_shape.insert(full_shape.begin(), env_count_);
    
    auto tensor_meta = meta;
    tensor_meta.shape = full_shape;
    
    auto tensor = std::make_unique<GTensorBase>(tensor_meta);
    auto* ptr = tensor.get();
    
    impl_->tensors[meta.name] = std::move(tensor);
    return ptr;
}

ITensor* TensorRegistry::getTensor(const std::string& name) {
    auto it = impl_->tensors.find(name);
    return it != impl_->tensors.end() ? it->second.get() : nullptr;
}

void TensorRegistry::removeTensor(const std::string& name) {
    impl_->tensors.erase(name);
}

std::vector<ITensor*> TensorRegistry::getTensorsByPrefix(const std::string& prefix) {
    std::vector<ITensor*> result;
    for (const auto& [name, tensor] : impl_->tensors) {
        if (name.substr(0, prefix.length()) == prefix) {
            result.push_back(tensor.get());
        }
    }
    return result;
}

void TensorRegistry::removeTensorsByPrefix(const std::string& prefix) {
    for (auto it = impl_->tensors.begin(); it != impl_->tensors.end();) {
        if (it->first.substr(0, prefix.length()) == prefix) {
            it = impl_->tensors.erase(it);
        } else {
            ++it;
        }
    }
}

size_t TensorRegistry::size() const {
    return impl_->tensors.size();
}

bool TensorRegistry::exists(const std::string& name) const {
    return impl_->tensors.find(name) != impl_->tensors.end();
}

std::vector<std::string> TensorRegistry::getAllNames() const {
    std::vector<std::string> names;
    names.reserve(impl_->tensors.size());
    for (const auto& [name, _] : impl_->tensors) {
        names.push_back(name);
    }
    return names;
}

} // namespace RSG_SIM