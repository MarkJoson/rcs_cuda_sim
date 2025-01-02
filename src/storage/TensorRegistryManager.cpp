#include <torch/torch.h>
#include "core/storage/TensorRegistryManager.hh"
namespace cuda_simulator
{
namespace core
{

bool TensorRegistryManager::initialized_ = false;

TensorRegistry TensorRegistryManager::createRegistry()
{
    if (!initialized_)
    {
        initialize();
    }
    return TensorRegistry();
}

void TensorRegistryManager::setDefaultDevice(bool use_gpu)
{
    // torch::set_default_tensor_type(
    //     use_gpu ? torch::kCUDA : torch::kCPU);
}

bool TensorRegistryManager::isGPUAvailable()
{
    return torch::cuda::is_available();
}

void TensorRegistryManager::initialize()
{
    if (!initialized_)
    {
        torch::manual_seed(0);
        if (torch::cuda::is_available())
        {
            torch::cuda::manual_seed_all(0);
        }
        initialized_ = true;
    }
}

void TensorRegistryManager::shutdown()
{
    if (initialized_)
    {
        // 清理PyTorch相关资源
        // torch::cuda::empty_cache();
        initialized_ = false;
    }
}

void TensorRegistryManager::clearCache()
{
    // torch::cuda::empty_cache();
}

size_t TensorRegistryManager::getCurrentAllocatedMemory()
{
    size_t allocated = 0;
    if (torch::cuda::is_available())
    {
        // allocated = torch::cuda::memory_allocated();
    }
    return allocated;
}

void TensorRegistryManager::setSeed(uint64_t seed)
{
    torch::manual_seed(seed);
    if (torch::cuda::is_available())
    {
        torch::cuda::manual_seed_all(seed);
    }
}
}

} // namespace cuda_simulator