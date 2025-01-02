#ifndef CUDASIM_ENVIRON_GROUP_MANAGER_HH
#define CUDASIM_ENVIRON_GROUP_MANAGER_HH

#include <cstdint>
#include <memory>
#include <string>
#include <iostream>
#include <unordered_map>
#include <optional>
#include <vector>
#include "config.h"
#include "cuda_helper.h"

namespace cuda_simulator
{
namespace core
{

namespace env_group_impl
{

extern __constant__ uint32_t constant_mem_pool[8192];

} // namespace env_group_impl


enum class MemoryType {
    GLOBAL,     // 普通全局内存
    CONSTANT,   // 常量内存
};

class EGConfigItemBase
{
public:
    EGConfigItemBase(int item_size, MemoryType mem_type, void *global_mem_ptr, int const_mem_offset)
        : item_size_(item_size), mem_type_(mem_type), global_mem_ptr_(global_mem_ptr), const_mem_offset_(const_mem_offset)
    {

    }

protected:
    int item_size_;
    MemoryType mem_type_;
    void *global_mem_ptr_;
    int const_mem_offset_ = -1;
};


template<typename T>
class EGConfigItem : public EGConfigItemBase
{
    using valueType = T;
public:
    __device__ T* getConfig(int env_group_idx) {
        // return static_cast<const T*>(d_arr_) + env_group_idx;
    }
private:
};


class EnvGroupManager {
public:
    EnvGroupManager()
        : constant_mempool_offset_(0)
    { }

    template<typename EGConfigCls>
    std::optional<EGConfigCls*> registerConfigItem(
        const std::string &name,
        EGConfigCls::valueType default_value,
        EGConfigCls::valueType* dev_ptr = nullptr,
        size_t dev_memsize = 0,
        MemoryType mem_type = MemoryType::GLOBAL
    ) {
        using T = typename EGConfigCls::valueType;

        if (registry_.find(name) != registry_.end()) {
            std::cerr << name << " already exists in registry, please check your code." << std::endl;
            return std::nullopt;
        }

        // 申请一块地址存放Config
        int const_mem_offset = -1;
        if(mem_type == MemoryType::CONSTANT) {
            // 需要在常量内存中的配置项，则在常量内存池和主机内存中分别申请，并在需要时将常量内存池的数据更新
            size_t pre_alloc_size = (sizeof(T) * MAX_NUM_ACTIVE_ENV_GROUP + (sizeof(uint32_t)-1)) / sizeof(uint32_t);
            if(constant_mempool_offset_ + pre_alloc_size >= sizeof(env_group_impl::constant_mem_pool) / sizeof(uint32_t)) {
                std::cerr << "constant memoy exhausted!" << std::endl;
                return std::nullopt;
            }
            const_mem_offset = constant_mempool_offset_;
            constant_mempool_offset_ += pre_alloc_size;

            // 初始化常量内存空间
            std::vector<typename EGConfigCls::valueType> host_mem(MAX_NUM_ACTIVE_ENV_GROUP, default_value);
            checkCudaErrors(cudaMemcpyToSymbol(env_group_impl::constant_mem_pool + const_mem_offset, host_mem.data(), sizeof(T)*MAX_NUM_ACTIVE_ENV_GROUP));
        }

        // 不论如何，会在Global memory中申请，所需内存是最大内存组的数量
        // 当已经提供了全局内存的缓冲区时，就不再申请全局内存
        if (dev_ptr == nullptr) {
            if(dev_memsize != 0 && dev_memsize != sizeof(T)*MAX_NUM_ENV_GROUP) {
                std::cerr << "wrong dev_memsize! it should be sizeof(T)*MAX_NUM_ENV_GROUP" << std::endl;
            }

            checkCudaErrors(cudaMalloc(dev_ptr, sizeof(T)*MAX_NUM_ENV_GROUP));

            // 初始化全局内存空间
            std::vector<typename EGConfigCls::valueType> host_mem(MAX_NUM_ENV_GROUP, default_value);
            checkCudaErrors(cudaMemcpy(dev_ptr, host_mem.data(), sizeof(T)*MAX_NUM_ENV_GROUP, cudaMemcpyHostToDevice));
        }

        auto item = std::make_unique<EGConfigCls>(sizeof(T), mem_type, dev_ptr, const_mem_offset);
        registry_[name] = item;
        return item.get();
    }

    template<typename EGConfigCls>
    std::optional<EGConfigCls*> getConfigItem(const std::string &name) {
        if (registry_.find(name) == registry_.end()) {
            std::cerr << name << " already exists in registry, please check your code." << std::endl;
            return std::nullopt;
        }
        return registry_[name];
    }



private:
    // 当前环境组的数量
    int num_env_group_;
    size_t constant_mempool_offset_;
    std::unordered_map<std::string, std::unique_ptr<EGConfigItemBase>> registry_;
};


} // namespace core
} // namespace cuda_simulator



#endif