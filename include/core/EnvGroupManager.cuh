#ifndef CUDASIM_ENV_GROUP_MANAGER_HH
#define CUDASIM_ENV_GROUP_MANAGER_HH

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

// #include <cuda_runtime_api.h>
#include "cuda_helper.h"
#include "core/storage/GTensorConfig.hh"
#include "core/storage/ITensor.hh"

#include "storage/TensorRegistry.hh"

namespace cuda_simulator
{
namespace core
{

namespace env_group_impl
{

constexpr int CONST_MEM_WORD_SIZE = 8192;
extern __constant__ int constant_mem_pool[CONST_MEM_WORD_SIZE];
extern __constant__ uint32_t d_num_active_group;

// TODO. 原有定义

// 单个group的所有常量数据是顺序排列的，每个group的数据是分开的
// group 1: |[var1, var2, var3, ...]|
// group 2: |[var1, var2, var3, ...]|
// group 3: |[var1, var2, var3, ...]|

inline int& getConstMemAllocPerGroup() {
    // TODO. 多线程互斥锁
    static int allocated_words = 0;
    return allocated_words;
}

inline int addConstMemAllocWords(int inc_words) {
    int old_words = getConstMemAllocPerGroup();
    getConstMemAllocPerGroup() = inc_words;
    return old_words;
}

inline int& getNumActiveGroup() {
    static int active_group_count = 0;
    return active_group_count;
}

inline bool constMemFreeSpaceLeft(int words_needed=0) {
    return getConstMemAllocPerGroup() + words_needed < CONST_MEM_WORD_SIZE / getNumActiveGroup();
}

inline void setNumActiveGroup(int count) {
    getNumActiveGroup() = count;
    if(!constMemFreeSpaceLeft()) {
        throw std::runtime_error("constant memory pool is full after setting active group count!");
    }
    checkCudaErrors(cudaMemcpyToSymbol(d_num_active_group, &count, sizeof(uint32_t)));
}

template<typename T>
inline int allocate() {
    // 向上取整
    int words_per_elem = (sizeof(T) + sizeof(env_group_impl::constant_mem_pool[0]) + 1) / sizeof(env_group_impl::constant_mem_pool[0]);
    if(!constMemFreeSpaceLeft(words_per_elem))
        throw std::bad_alloc();

    int offset = env_group_impl::getConstMemAllocPerGroup();
    env_group_impl::getConstMemAllocPerGroup() += words_per_elem;
    return offset;
}

template<typename T>
inline __device__ T getConstData(int group_id, int offset) {
    return env_group_impl::constant_mem_pool[group_id * (CONST_MEM_WORD_SIZE / d_num_active_group) + offset];
}

inline int getConstantMemPoolOffset(int group_id, int offset) {
    // return group_id * (getConstMemAllocPerGroup()) + offset;
    return group_id * (CONST_MEM_WORD_SIZE / getNumActiveGroup()) + offset;
}

} // namespace env_group_impl


enum class MemoryType {
    HOST_MEM,           // 主机内存
    CONSTANT_GPU_MEM,   // 常量内存
    GLOBAL_GPU_MEM,     // 普通全局内存
    GLOBAL_TENSOR       // 全局张量
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

template<typename T>
class ConstantMemoryVector {
    int offset_;
    static constexpr bool CHECK_BOUND = true;
public:
    ConstantMemoryVector() {
        offset_ = env_group_impl::allocate<T>();
    }

    __device__ inline const T operator[](int idx) {
        if(idx >= env_group_impl::d_num_active_group) {
            printf("index out of range: %d\n", idx);
            return T();
        }
            // throw std::out_of_range("index out of range");
        return env_group_impl::constant_mem_pool[offset_ + idx];
    }

    void set(int idx, const T& value) {
        if(CHECK_BOUND && idx >= env_group_impl::getNumActiveGroup()) {
            throw std::out_of_range("index out of range");
        }
        env_group_impl::constant_mem_pool[offset_ + idx] = value;
        int pool_offset = env_group_impl::getConstantMemPoolOffset(idx, offset_);
        checkCudaErrors(cudaMemcpyToSymbol(env_group_impl::constant_mem_pool, &value, sizeof(T), pool_offset));
    }
};


class EGConfigItemBase {
protected:
    MemoryType mem_type_;
    int num_group_;
public:
    EGConfigItemBase(int num_group, MemoryType mem_type)
        : mem_type_(mem_type), num_group_(num_group) {}

    virtual ~EGConfigItemBase() = default;

    virtual void syncToDevice() {};

    MemoryType getMemoryType() const {
        return mem_type_;
    }

    int getNumEnvGroup() const {
        return num_group_;
    }
};

template<typename T>
class EGHostMemConfigItem : public EGConfigItemBase {
protected:
    std::vector<T> host_data_;
public:
    EGHostMemConfigItem(int64_t num_group, MemoryType alternative=MemoryType::HOST_MEM)
        : EGConfigItemBase(num_group, alternative), host_data_(num_group)
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
    // std::vector<T, ConstantMemoryAllocator<T>> device_data_;
    ConstantMemoryVector<T> device_data_;
    int num_active_group_;
    const std::vector<int> &active_group_indices_;

public:
    EGConstMemConfigItem(int64_t num_group, int64_t num_active_group, const std::vector<int> &active_group_indices)
        : EGHostMemConfigItem<T>(num_group, MemoryType::CONSTANT_GPU_MEM)
        , device_data_()
        , num_active_group_(num_active_group)
        , active_group_indices_(active_group_indices)
    {
        env_group_impl::setNumActiveGroup(num_active_group);
    }

    // 根据活跃组id同步到设备
    void syncToDevice() override {
        for(auto group_id : active_group_indices_) {
            // reflect_data.push_back(EGHostMemConfigItem<T>::host_data_[group_id]);
            device_data_.set(group_id, EGHostMemConfigItem<T>::host_data_[group_id]);
        }
    }
};

// 全局内存区域句柄
template<typename T>
class EGGlobalMemConfigItem : public EGHostMemConfigItem<T> {
    std::vector<T, GlobalMemoryAllocator<T>> device_data_;
public:
    EGGlobalMemConfigItem(int64_t num_group)
        : EGHostMemConfigItem<T>(num_group, MemoryType::GLOBAL_GPU_MEM), device_data_(num_group)
    { }

    void syncToDevice() override {
        checkCudaErrors(cudaMemcpy(
            device_data_.data(),
            EGHostMemConfigItem<T>::host_data().data(),
            EGHostMemConfigItem<T>::host_data().size() * sizeof(T), cudaMemcpyHostToDevice
        ));
    }
};


template<typename T>
class EGGlobalMemConfigTensor : public EGConfigItemBase {
    TensorHandle host_tensor_;
    TensorHandle device_tensor_;
public:
    EGGlobalMemConfigTensor(const std::string& name, int64_t num_group, const TensorShape& shape)
        : EGConfigItemBase(num_group, MemoryType::GLOBAL_TENSOR)
    {
        TensorShape new_shape = shape;
        new_shape.insert(new_shape.begin(), num_group);

        TensorRegistry::getInstance().createTensor<T>(host_tensor_, name + "_Config@CPU", new_shape, DeviceType::kCPU);
        TensorRegistry::getInstance().createTensor<T>(device_tensor_, name + "_Config@CUDA", new_shape, DeviceType::kCUDA);
    }

    TensorHandle& getHostTensor() {
        return host_tensor_;
    }

    const TensorHandle& getDeviceTensor() {
        return device_tensor_;
    }

    void syncToDevice() {
        host_tensor_.copyTo(device_tensor_);
    }

    template<typename... Args>
    auto at(int64_t group_id, Args... indices) {
        if (group_id >= num_group_) {
            throw std::out_of_range("group_id out of range");
        }
        return host_tensor_[{group_id, indices...}];
    }
};


// ------------------- 环境组管理器 -------------------

class EnvGroupManager {
public:
    EnvGroupManager(int num_env_per_group=1, int num_group=1, int num_active_group=1)
        : num_env_per_group_(num_env_per_group)
        , num_group_(num_group)
        , num_active_group_(num_active_group)
    {
        if(num_active_group > num_group) {
            throw std::runtime_error("num_active_group should be less than or equal to num_group");
        }
        env_group_impl::setNumActiveGroup(num_active_group);
    }

    constexpr static int SHAPE_PLACEHOLDER_GROUP = -100;
    constexpr static int SHAPE_PLACEHOLDER_ENV = -101;

    template<typename T, MemoryType mem_type>
    auto registerConfigItem(const std::string &name) {
        if (registry_.find(name) != registry_.end()) {
            throw std::runtime_error("name already exists in registry, please check your code.");
        }

        if constexpr (mem_type == MemoryType::CONSTANT_GPU_MEM) {
            std::unique_ptr<EGConstMemConfigItem<T>> item = std::make_unique<EGConstMemConfigItem<T>> (num_group_, num_active_group_, active_group_indices_);
            EGConstMemConfigItem<T>* ptr = item.get();
            registry_.insert({name, std::move(item)});
            return ptr;
        } else if constexpr (mem_type == MemoryType::GLOBAL_GPU_MEM) {
            std::unique_ptr<EGGlobalMemConfigItem<T>> item = std::make_unique<EGGlobalMemConfigItem<T>> (num_group_);
            EGGlobalMemConfigItem<T>* ptr = item.get();
            registry_.insert({name, std::move(item)});
            return ptr;
        } else if constexpr (mem_type == MemoryType::HOST_MEM) {
            std::unique_ptr<EGHostMemConfigItem<T>> item = std::make_unique<EGHostMemConfigItem<T>> (num_group_);
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
        std::unique_ptr<EGGlobalMemConfigTensor<T>> item = std::make_unique<EGGlobalMemConfigTensor<T>> (name, num_group_, shape);
        EGGlobalMemConfigTensor<T>* ptr = item.get();
        registry_.insert({name, std::move(item)});
        return ptr;
    }

    template<typename T>
    TensorHandle& createTensor(
            const std::string& name,
            const TensorShape& shape_with_placeholder,
            DeviceType device_type=DeviceType::kCUDA) {

        std::vector<int64_t> shape;
        for(auto s : shape_with_placeholder) {
            if(s == SHAPE_PLACEHOLDER_GROUP) {
                // TODO. 这里使用的是num_group，因为某些核函数需要数据紧密排列。因此需要在调整group数量后的二次分配
                shape.push_back(num_env_per_group_);
            } else if(s == SHAPE_PLACEHOLDER_ENV) {
                shape.push_back(num_env_per_group_);
            } else {
                shape.push_back(s);
            }
        }

        return TensorRegistry::getInstance().createTensor<T>(name, shape, device_type);
    }

    template<typename T>
    void createTensor(
            TensorHandle &target,
            const std::string& name,
            const TensorShape& shape_with_placeholder,
            DeviceType device_type=DeviceType::kCUDA) {

        std::vector<int64_t> shape;
        for(auto s : shape_with_placeholder) {
            if(s == SHAPE_PLACEHOLDER_GROUP) {
                shape.push_back(num_group_);
            } else if(s == SHAPE_PLACEHOLDER_ENV) {
                shape.push_back(num_env_per_group_);
            } else {
                shape.push_back(s);
            }
        }

        TensorRegistry::getInstance().createTensor<T>(target, name, shape, device_type);
    }

    int getNumGroup() const { return num_group_; }
    int getNumActiveGroup() const { return active_group_indices_.size(); }
    int getNumEnvPerGroup() const { return num_env_per_group_; }

    void syncToDevice() {
        for(auto& [name, item] : registry_) {
            item->syncToDevice();
        }
    }

private:
    // 最大环境组的数量
    // int max_num_group_;
    // 最大活跃环境组的数量
    // int max_num_active_group_;

    // 每个环境组的环境数量
    const int num_env_per_group_;
    // 当前环境组的数量
    const int num_group_;
    // 当前活跃环境组的数量
    const int num_active_group_;
    // 活跃环境组的索引
    std::vector<int> active_group_indices_;
    // 环境组配置项注册表
    std::unordered_map<std::string, std::unique_ptr<EGConfigItemBase>> registry_;
};


} // namespace core
} // namespace cuda_simulator



#endif // CUDASIM_ENV_GROUP_MANAGER_HH