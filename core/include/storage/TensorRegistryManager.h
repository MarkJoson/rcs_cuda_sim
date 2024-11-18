#ifndef __TENSOR_REGISTRY_MANAGER_H__
#define __TENSOR_REGISTRY_MANAGER_H__

#include "TensorRegistry.h"

namespace RSG_SIM {

class TensorRegistryManager {
public:
    // 禁止实例化
    TensorRegistryManager() = delete;
    
    // 创建注册表
    static TensorRegistry createRegistry(int env_count);
    
    // 设备管理
    static void setDefaultDevice(bool use_gpu);
    static bool isGPUAvailable();
    
    // 初始化和清理
    static void initialize();
    static void shutdown();
    
    // 内存管理
    static void clearCache();
    static size_t getCurrentAllocatedMemory();
    
    // 随机数种子
    static void setSeed(uint64_t seed);

private:
    static bool initialized_;
};

} // namespace RSG_SIM

#endif