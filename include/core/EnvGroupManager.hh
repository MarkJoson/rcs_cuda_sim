#ifndef CUDASIM_ENV_GROUP_MANAGER_HH
#define CUDASIM_ENV_GROUP_MANAGER_HH

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>
#include "config.h"
#include "core/storage/GTensorConfig.hh"
#include "core/storage/ITensor.hh"
#include "core/storage/Scalar.hh"

#include "storage/TensorRegistry.hh"

namespace cuda_simulator
{
namespace core
{

namespace env_group_impl
{

extern __constant__ uint32_t constant_mem_pool[8192];

inline std::size_t& getConstantMemAllocatedWords() {
    static std::size_t allocated_words = 0;
    return allocated_words;
}

} // namespace env_group_impl


enum class MemoryType {
    GLOBAL_GPU_MEM,     // 普通全局内存
    CONSTANT_GPU_MEM,   // 常量内存
    HOST_MEM
};

// ------------------- 内存管理 -------------------

template<typename T>
class GlobalMemoryAllocator {
public:
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using size_type = std::size_t;

    pointer allocate(size_type n) {
        pointer p;
        cudaError_t err = cudaMalloc(&p, n * sizeof(T));
        if (err != cudaSuccess) {
            throw std::bad_alloc();
        }
        return p;
    }

    void deallocate(pointer p, size_type) {
        cudaError_t err = cudaFree(p);
        if (err != cudaSuccess) {
            throw std::runtime_error("CUDA deallocation failed");
        }
    }

    template <typename U>
    struct rebind {
        using other = GlobalMemoryAllocator<U>;
    };
};

// 常量内存分配器
template<typename T>
class ConstantMemoryAllocator {
public:
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using size_type = std::size_t;

    pointer allocate(size_type n) {
        size_type words = ((n * sizeof(T) + sizeof(env_group_impl::constant_mem_pool[0]) + 1) / sizeof(env_group_impl::constant_mem_pool[0])) ;    // 按照字大小对齐分配
        if(env_group_impl::getConstantMemAllocatedWords() + words
            > sizeof(env_group_impl::constant_mem_pool) / sizeof(env_group_impl::constant_mem_pool[0])) {
            throw std::bad_alloc();
        }
        pointer p = reinterpret_cast<pointer>(env_group_impl::constant_mem_pool + env_group_impl::getConstantMemAllocatedWords());
        env_group_impl::getConstantMemAllocatedWords() += words;
        return p;
    }

    void deallocate(pointer, size_type) {
        // do nothing
    }

    size_type max_size() const noexcept {
        return (sizeof(env_group_impl::constant_mem_pool) - env_group_impl::getConstantMemAllocatedWords()) / sizeof(T);
    }

    template <typename U>
    struct rebind {
        using other = ConstantMemoryAllocator<U>;
    };

};

class EGConfigItemBase {
    MemoryType mem_type_;
    int64_t num_env_grp_;
public:
    EGConfigItemBase(MemoryType mem_type, int64_t num_env_grp=1)
        : mem_type_(mem_type), num_env_grp_(num_env_grp) {}

    MemoryType getMemoryType() const {
        return mem_type_;
    }

    int64_t getNumEnvGroup() const {
        return num_env_grp_;
    }
};

template<typename T>
class EGHostMemConfigItem : public EGConfigItemBase {
    std::vector<T> host_data_;
public:
    EGHostMemConfigItem(int64_t num_env_grp, MemoryType alternative=MemoryType::HOST_MEM)
        : EGConfigItemBase(alternative, num_env_grp), host_data_(num_env_grp)
    { }

    T& at(int64_t idx) {
        return host_data_[idx];
    }

    std::vector<T>& host_data()  {
        return host_data_;
    }
};

// 常量内存区域句柄，仅支持基本数据类型，包含常量内存池偏移量，主机内存句柄
template<typename T>
class EGConstMemConfigItem : public EGHostMemConfigItem<T> {
    std::vector<T, ConstantMemoryAllocator<T>> device_data_;
    int64_t num_active_env_grp;

public:
    EGConstMemConfigItem(int64_t num_env_grp, int64_t num_active_env_grp)
        : EGHostMemConfigItem<T>(num_env_grp, MemoryType::CONSTANT_GPU_MEM), device_data_(num_active_env_grp), num_active_env_grp(num_active_env_grp)
    { }

};

// 全局内存区域句柄，
template<typename T>
class EGGlobalMemConfigItem : public EGHostMemConfigItem<T> {
    std::vector<T, GlobalMemoryAllocator<T>> device_data_;
public:
    EGGlobalMemConfigItem(int64_t num_env_grp)
        : EGHostMemConfigItem<T>(num_env_grp, MemoryType::GLOBAL_GPU_MEM), device_data_(num_env_grp)
    { }
};


template<typename T>
class EGGlobalMemConfigTensor : public EGConfigItemBase {
    TensorHandle* host_tensor_;
    TensorHandle* device_tensor_;
public:
    EGGlobalMemConfigTensor(const std::string& name, int64_t num_env_grp, const TensorShape& shape)
        : EGConfigItemBase(MemoryType::GLOBAL_GPU_MEM)
    {
        TensorShape new_shape = shape;
        new_shape.insert(new_shape.begin(), num_env_grp);

        host_tensor_ = TensorRegistry::getInstance().createTensor<T>(name + "_Config@CPU", new_shape, DeviceType::kCPU);
        device_tensor_ = TensorRegistry::getInstance().createTensor<T>( name + "_Config@CUDA", new_shape, DeviceType::kCUDA);
    }

    TensorHandle* getHostTensor() {
        return host_tensor_;
    }

    const TensorHandle* getDeviceTensor() {
        return device_tensor_;
    }

    template<typename... Args>
    auto at(int64_t env_grp_id, Args... indices) {
        return (*host_tensor_)[{env_grp_id, indices...}];
    }
};


// ------------------- 环境组管理器 -------------------

class EnvGroupManager {
public:
    EnvGroupManager(int max_num_active_env_group, int max_num_env_group)
        : max_num_env_group_(max_num_env_group)
        , max_num_active_env_group_(max_num_active_env_group)
        , num_env_group_(0)
    { }

    constexpr static int SHAPE_PLACEHOLDER_ENV_GRP = -100;
    constexpr static int SHAPE_PLACEHOLDER_ENV = -101;

    template<typename T, MemoryType mem_type>
    auto registerConfigItem(const std::string &name) {
        if (registry_.find(name) != registry_.end()) {
            throw std::runtime_error("name already exists in registry, please check your code.");
        }

        if constexpr (mem_type == MemoryType::CONSTANT_GPU_MEM) {
            std::unique_ptr<EGConstMemConfigItem<T>> item = std::make_unique<EGConstMemConfigItem<T>> (max_num_env_group_, max_num_active_env_group_);
            EGConstMemConfigItem<T>* ptr = item.get();
            registry_.insert({name, std::move(item)});
            return ptr;
        } else if constexpr (mem_type == MemoryType::GLOBAL_GPU_MEM) {
            std::unique_ptr<EGGlobalMemConfigItem<T>> item = std::make_unique<EGGlobalMemConfigItem<T>> (max_num_env_group_);
            EGGlobalMemConfigItem<T>* ptr = item.get();
            registry_.insert({name, std::move(item)});
            return ptr;
        } else if constexpr (mem_type == MemoryType::HOST_MEM) {
            std::unique_ptr<EGHostMemConfigItem<T>> item = std::make_unique<EGHostMemConfigItem<T>> (max_num_env_group_);
            EGHostMemConfigItem<T>* ptr = item.get();
            registry_.insert({name, std::move(item)});
            return ptr;
        } else {
            throw std::runtime_error("Unsupported memory type");
        }
    }

    template<typename T>
    auto registerConfigTensor(const std::string &name, const TensorShape& shape) {
        if (registry_.find(name) != registry_.end()) {
            throw std::runtime_error("name already exists in registry, please check your code.");
        }
        // TODO. 到底用max_num_env_group_还是num_env_group_
        std::unique_ptr<EGGlobalMemConfigTensor<T>> item = std::make_unique<EGGlobalMemConfigTensor<T>> (name, max_num_env_group_, shape);
        EGGlobalMemConfigTensor<T>* ptr = item.get();
        registry_.insert({name, std::move(item)});
        return ptr;
    }

    void setEnvGroupCount(int num_env_group) {

        if(num_env_group > max_num_env_group_) {
            throw std::runtime_error("num_env_group exceeds max_num_env_group_");
        }


        if(num_env_group > num_env_group_) {
            // TODO. 调用所有回调函数初始化对应的环境组参数
        }

        num_env_group_ = num_env_group;
    }

    template<typename T>
    TensorHandle* createTensor(const std::string& name, const TensorShape& shape_with_placeholder, DeviceType device_type=DeviceType::kCUDA) {

        std::vector<int64_t> shape;
        for(auto s : shape_with_placeholder) {
            if(s == SHAPE_PLACEHOLDER_ENV_GRP) {
                shape.push_back(num_env_group_);
            } else if(s == SHAPE_PLACEHOLDER_ENV) {
                shape.push_back(num_env_per_group_);
            } else {
                shape.push_back(s);
            }
        }

        return TensorRegistry::getInstance().createTensor<T>(name, shape, device_type);
    }

    int getNumEnvGroup() const { return num_env_group_; }
    int getNumEnvPerGroup() const { return num_env_per_group_; }

private:
    // 最大环境组的数量
    int max_num_env_group_;
    // 最大活跃环境组的数量
    int max_num_active_env_group_;
    // 每个环境组的环境数量
    int num_env_per_group_;
    // 当前环境组的数量
    int num_env_group_;
    std::unordered_map<std::string, std::unique_ptr<EGConfigItemBase>> registry_;
};


} // namespace core
} // namespace cuda_simulator



#endif // CUDASIM_ENV_GROUP_MANAGER_HH