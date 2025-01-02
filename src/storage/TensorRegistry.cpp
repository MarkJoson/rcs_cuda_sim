
#include "core/storage/GTensor.hh"
#include "core/storage/TensorRegistry.hh"

namespace cuda_simulator
{
namespace core
{
TensorRegistry::TensorRegistry()
{
}

TensorRegistry::~TensorRegistry() = default;

TensorRegistry::TensorRegistry(TensorRegistry &&) noexcept = default;
TensorRegistry &TensorRegistry::operator=(TensorRegistry &&) noexcept = default;

ITensor *TensorRegistry::createTensor(const std::string &uri, const std::vector<int64_t> &shape, TensorDataType dtype)
{
    auto tensor = std::make_unique<GTensorTorchWrapper>(shape, dtype);
    auto *ptr = tensor.get();

    tensors[uri] = std::move(tensor);
    return ptr;
}

ITensor *TensorRegistry::getTensor(const std::string &uri)
{
    auto it = tensors.find(uri);
    return it != tensors.end() ? it->second.get() : nullptr;
}

void TensorRegistry::removeTensor(const std::string &uri)
{
    tensors.erase(uri);
}

std::vector<ITensor *> TensorRegistry::getTensorsByPrefix(const std::string &prefix)
{
    std::vector<ITensor *> result;
    for (const auto &[uri, tensor] : tensors)
    {
        if (uri.substr(0, prefix.length()) == prefix)
        {
            result.push_back(tensor.get());
        }
    }
    return result;
}

void TensorRegistry::removeTensorsByPrefix(const std::string &prefix)
{
    for (auto it = tensors.begin(); it != tensors.end();)
    {
        if (it->first.substr(0, prefix.length()) == prefix)
        {
            it = tensors.erase(it);
        }
        else
        {
            ++it;
        }
    }
}

size_t TensorRegistry::size() const
{
    return tensors.size();
}

bool TensorRegistry::exists(const std::string &uri) const
{
    return tensors.find(uri) != tensors.end();
}

std::vector<std::string> TensorRegistry::getAllTensorUri() const
{
    std::vector<std::string> uris;
    uris.reserve(tensors.size());
    for (const auto &[uri, _] : tensors)
    {
        uris.push_back(uri);
    }
    return uris;
}
}

} // namespace cuda_simulator