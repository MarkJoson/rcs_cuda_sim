#ifndef CUDASIM_ENV_GROUP_MANAGER_HH
#define CUDASIM_ENV_GROUP_MANAGER_HH

#include <cstdint>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

// #include <cuda_runtime_api.h>
#include "cuda_helper.h"
#include "config.h"
#include "core/storage/GTensorConfig.hh"
#include "core/storage/ITensor.hh"
#include "core/storage/Scalar.hh"

#include "storage/TensorRegistry.hh"

namespace cuda_simulator
{
namespace core
{

constexpr int MAX_NUM_ACTIVE_GROUP = 64;    // 最大活跃环境组数量

namespace env_group_impl
{

constexpr int CONST_MEM_WORD_SIZE = 8192;
extern __constant__ __device__ int constant_mem_pool[CONST_MEM_WORD_SIZE];
extern __constant__ __device__ uint32_t dc_active_group_count;
extern __constant__ __device__ int dc_const_mem_alloc_words;

// 单个group的所有常量数据是顺序排列的，每个group的数据是分开的
// group 1: |[var1, var2, var3, ...]|
// group 2: |[var1, var2, var3, ...]|
// group 3: |[var1, var2, var3, ...]|

inline int& getConstMemAllocWords() {
    // TODO. 多线程互斥锁
    static int allocated_words = 0;
    return allocated_words;
}

inline int addConstMemAllocWords(int inc_words) {
    int old_words = getConstMemAllocWords();
    getConstMemAllocWords() = inc_words;
    checkCudaErrors(cudaMemcpyToSymbol(dc_const_mem_alloc_words, &inc_words, sizeof(uint32_t)));
    return old_words;
}

inline int& getActiveGroupCount() {
    static int active_group_count = 0;
    return active_group_count;
}

inline bool constMemFreeSpaceLeft(int words_needed=0) {
    return (getConstMemAllocWords()+words_needed)*getActiveGroupCount() < CONST_MEM_WORD_SIZE;
}

inline void setActiveGroupCount(int count) {
    getActiveGroupCount() = count;
    if(!constMemFreeSpaceLeft()) {
        throw std::runtime_error("constant memory pool is full after setting active group count!");
    }
    checkCudaErrors(cudaMemcpyToSymbol(dc_active_group_count, &count, sizeof(uint32_t)));
}

template<typename T>
inline int allocate() {
    // 向上取整
    int words_per_elem = (sizeof(T) + sizeof(env_group_impl::constant_mem_pool[0]) + 1) / sizeof(env_group_impl::constant_mem_pool[0]);
    if(!constMemFreeSpaceLeft(words_per_elem))
        throw std::bad_alloc();

    int offset = env_group_impl::getConstMemAllocWords();
    env_group_impl::getConstMemAllocWords() += words_per_elem;
    return offset;
}

template<typename T>
inline __device__ T getConstData(int group_id, int offset) {
    return env_group_impl::constant_mem_pool[group_id * dc_const_mem_alloc_words + offset];
}

inline int getConstantMemPoolOffset(int group_id, int offset) {
    return group_id * getConstMemAllocWords() + offset;
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

template<typename T>
class ConstantMemoryVector {
    int offset_;
    static constexpr bool CHECK_BOUND = true;
public:
    ConstantMemoryVector() {
        offset_ = env_group_impl::allocate<T>();
    }

    __device__ inline const T operator[](int idx) {
        if(idx >= env_group_impl::dc_active_group_count) {
            printf("index out of range: %d\n", idx);
            return T();
        }
            // throw std::out_of_range("index out of range");
        return env_group_impl::constant_mem_pool[offset_ + idx];
    }

    void set(int idx, const T& value) {
        if(CHECK_BOUND && idx >= env_group_impl::getActiveGroupCount()) {
            throw std::out_of_range("index out of range");
        }
        env_group_impl::constant_mem_pool[offset_ + idx] = value;
        int offset = env_group_impl::getConstantMemPoolOffset(idx, offset_);
        checkCudaErrors(cudaMemcpyToSymbol(env_group_impl::constant_mem_pool, value, sizeof(T), offset));
    }
};


class EGConfigItemBase {
protected:
    MemoryType mem_type_;
    int num_group_;
public:
    EGConfigItemBase(MemoryType mem_type, int num_group=1)
        : mem_type_(mem_type), num_group_(num_group) {}

    virtual ~EGConfigItemBase() = default;

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
        : EGConfigItemBase(alternative, num_group), host_data_(num_group)
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
        env_group_impl::setActiveGroupCount(num_active_group);
    }

    // 根据活跃组id同步到设备
    void syncToDevice() {
        for(auto group_id : active_group_indices_) {
            // reflect_data.push_back(EGHostMemConfigItem<T>::host_data_[group_id]);
            device_data_.set(group_id, EGHostMemConfigItem<T>::host_data_[group_id]);
        }
    }
};

// 全局内存区域句柄，
template<typename T>
class EGGlobalMemConfigItem : public EGHostMemConfigItem<T> {
    std::vector<T, GlobalMemoryAllocator<T>> device_data_;
public:
    EGGlobalMemConfigItem(int64_t num_group)
        : EGHostMemConfigItem<T>(num_group, MemoryType::GLOBAL_GPU_MEM), device_data_(num_group)
    { }
};


template<typename T>
class EGGlobalMemConfigTensor : public EGConfigItemBase {
    TensorHandle* host_tensor_;
    TensorHandle* device_tensor_;
public:
    EGGlobalMemConfigTensor(const std::string& name, int64_t num_group, const TensorShape& shape)
        : EGConfigItemBase(MemoryType::GLOBAL_GPU_MEM)
    {
        TensorShape new_shape = shape;
        new_shape.insert(new_shape.begin(), num_group);

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
    auto at(int64_t group_id, Args... indices) {
        return (*host_tensor_)[{group_id, indices...}];
    }
};


// ------------------- 环境组管理器 -------------------

class EnvGroupManager {
public:
    EnvGroupManager(int max_num_active_env_group, int max_num_env_group)
        : max_num_group_(max_num_env_group)
        , max_num_active_group_(max_num_active_env_group)
        , num_group_(0)
    { }

    constexpr static int SHAPE_PLACEHOLDER_GROUP = -100;
    constexpr static int SHAPE_PLACEHOLDER_ENV = -101;

    template<typename T, MemoryType mem_type>
    auto registerConfigItem(const std::string &name) {
        if (registry_.find(name) != registry_.end()) {
            throw std::runtime_error("name already exists in registry, please check your code.");
        }

        if constexpr (mem_type == MemoryType::CONSTANT_GPU_MEM) {
            std::unique_ptr<EGConstMemConfigItem<T>> item = std::make_unique<EGConstMemConfigItem<T>> (max_num_group_, max_num_active_group_, active_group_indices_);
            EGConstMemConfigItem<T>* ptr = item.get();
            registry_.insert({name, std::move(item)});
            return ptr;
        } else if constexpr (mem_type == MemoryType::GLOBAL_GPU_MEM) {
            std::unique_ptr<EGGlobalMemConfigItem<T>> item = std::make_unique<EGGlobalMemConfigItem<T>> (max_num_group_);
            EGGlobalMemConfigItem<T>* ptr = item.get();
            registry_.insert({name, std::move(item)});
            return ptr;
        } else if constexpr (mem_type == MemoryType::HOST_MEM) {
            std::unique_ptr<EGHostMemConfigItem<T>> item = std::make_unique<EGHostMemConfigItem<T>> (max_num_group_);
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
        std::unique_ptr<EGGlobalMemConfigTensor<T>> item = std::make_unique<EGGlobalMemConfigTensor<T>> (name, max_num_group_, shape);
        EGGlobalMemConfigTensor<T>* ptr = item.get();
        registry_.insert({name, std::move(item)});
        return ptr;
    }

    void setEnvGroupCount(int new_num_group) {

        if(new_num_group > max_num_group_) {
            throw std::runtime_error("new_num_group exceeds max_num_env_group_");
        }


        if(new_num_group > num_group_) {
            // TODO. 调用所有回调函数初始化对应的环境组参数
        }

        num_group_ = new_num_group;
    }

    template<typename T>
    TensorHandle* createTensor(const std::string& name, const TensorShape& shape_with_placeholder, DeviceType device_type=DeviceType::kCUDA) {

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

        return TensorRegistry::getInstance().createTensor<T>(name, shape, device_type);
    }

    int getNumGroup() const { return num_group_; }
    int getNumEnvPerGroup() const { return num_env_per_group_; }

private:
    // 最大环境组的数量
    int max_num_group_;
    // 最大活跃环境组的数量
    int max_num_active_group_;
    // 每个环境组的环境数量
    int num_env_per_group_;
    // 当前环境组的数量
    int num_group_;
    // 活跃环境组的索引
    std::vector<int> active_group_indices_;
    // 环境组配置项注册表
    std::unordered_map<std::string, std::unique_ptr<EGConfigItemBase>> registry_;
};


} // namespace core
} // namespace cuda_simulator



#endif // CUDASIM_ENV_GROUP_MANAGER_HH