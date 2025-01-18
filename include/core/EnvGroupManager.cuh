#ifndef CUDASIM_ENV_GROUP_MANAGER_HH
#define CUDASIM_ENV_GROUP_MANAGER_HH

#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/storage/Scalar.hh"
#include "cuda_helper.h"
#include "storage/GTensorConfig.hh"
#include "storage/ITensor.hh"
#include "storage/TensorRegistry.hh"

namespace cuda_simulator {
namespace core {

enum class MemoryType {
  HOST_MEM,         // 主机内存
  CONSTANT_GPU_MEM, // 常量内存
  GLOBAL_GPU_MEM,   // 普通全局内存
  GLOBAL_TENSOR     // 全局张量
};

const int MAX_NUM_ACTIVE_GROUP = 16;

namespace env_group_impl {

// 活跃环境组的索引 活跃环境组 -> 环境组 映射管理器
struct ActiveGroupMapperStorage {
  int num_active_group = 0;
  int id_map[MAX_NUM_ACTIVE_GROUP] = {0};
};

extern __constant__ ActiveGroupMapperStorage d_agm_storage_;

class EGActiveGroupMapper {
  static ActiveGroupMapperStorage h_agm_storage_;

public:
  __device__ static inline int devGetGroupId(int active_group_id) { return d_agm_storage_.id_map[active_group_id]; }

  __host__ static inline int hostGetGroupId(int active_group_id) { return h_agm_storage_.id_map[active_group_id]; }

  __device__ static inline int devGetNumActiveGroup() { return d_agm_storage_.num_active_group; }

  __host__ static inline int hostGetNumActiveGroup() { return h_agm_storage_.num_active_group; }

  __host__ static inline void hostSetNumActiveGroup(int num_active_group) {
    if (num_active_group > MAX_NUM_ACTIVE_GROUP) {
      throw std::runtime_error("num_active_group exceeds maximum value when set num_active_group!\n");
    }
    h_agm_storage_.num_active_group = num_active_group;
    checkCudaErrors(cudaMemcpyToSymbol(d_agm_storage_, &h_agm_storage_, sizeof(h_agm_storage_)));
  }

  __host__ static inline void replaceActiveGroups(const std::vector<int> &active_group_ids) {
    if (active_group_ids.size() > MAX_NUM_ACTIVE_GROUP) {
      throw std::runtime_error("active group ids exceeds maximum value when replace ActiveGroups!\n");
    }
    h_agm_storage_.num_active_group = active_group_ids.size();
    std::memcpy(h_agm_storage_.id_map, active_group_ids.data(), sizeof(h_agm_storage_.id_map));
    checkCudaErrors(cudaMemcpyToSymbol(d_agm_storage_, &h_agm_storage_, sizeof(h_agm_storage_)));
  }
};

struct ConstMemPoolConfig {
  int num_group_alloc_elem_;
  int capacity_per_group_;
  __host__ __device__ __forceinline__ int getMemPoolOffset(int group_id, int offset) const {
    return group_id * capacity_per_group_ + offset;
  }
};

static constexpr int CONST_MEM_WORD_SIZE = 8192;
extern __constant__ int constant_mem_pool[CONST_MEM_WORD_SIZE];
extern __constant__ ConstMemPoolConfig d_cmp_config_;

class EGConstantMemoryPool {
  static ConstMemPoolConfig h_cmp_config_;
  using ConstMemElemType = decltype(constant_mem_pool[0]);

public:
  template <typename T> static inline int allocate() {
    // 向上取整
    int words_per_group = (sizeof(T) + sizeof(ConstMemElemType) + 1) / sizeof(ConstMemElemType);
    if (h_cmp_config_.num_group_alloc_elem_ + words_per_group > h_cmp_config_.capacity_per_group_) {
      throw std::bad_alloc();
    }
    h_cmp_config_.num_group_alloc_elem_ += words_per_group;
    syncConfigToDevice();
    return h_cmp_config_.num_group_alloc_elem_ - words_per_group;
  }

  __host__ static inline int hostGetMemPoolOffset(int group_id, int offset) {
    // return group_id * (getConstMemAllocPerGroup()) + offset;
    return h_cmp_config_.getMemPoolOffset(group_id, offset);
  }

  __device__ static inline int devGetMemPoolOffset(int group_id, int offset) {
    return d_cmp_config_.getMemPoolOffset(group_id, offset);
  }

  template <typename T> static inline __device__ const T &getData(int group_id, int offset) {
    return *(reinterpret_cast<T *>(&constant_mem_pool[devGetMemPoolOffset(group_id, offset)]));
  }

  template <typename T> static inline __host__ void setData(int group_id, int offset, const T &data) {
    checkCudaErrors(cudaMemcpyToSymbol(constant_mem_pool, &data, sizeof(T), hostGetMemPoolOffset(group_id, offset)));
  }

  // 当num_active_group发生变化时调用
  __host__ static inline void updateConfig() {
    // 当前存储模式下，所有 active group 均分constant memory
    h_cmp_config_.capacity_per_group_ = CONST_MEM_WORD_SIZE / EGActiveGroupMapper::hostGetNumActiveGroup();
    if (h_cmp_config_.num_group_alloc_elem_ > h_cmp_config_.capacity_per_group_) {
      throw std::runtime_error("constant memory pool is full after setting active group count!");
    }

    // 修改后自动同步到device
    syncConfigToDevice();
  }

private:
  __host__ static inline void syncConfigToDevice() {
    checkCudaErrors(cudaMemcpyToSymbol(d_cmp_config_, &h_cmp_config_, sizeof(ConstMemPoolConfig)));
  }
};

} // namespace env_group_impl

// ------------------- 访问器 -------------------
template <typename Derived, typename T> class MemItemAccessor {
  __host__ __device__ Derived &derived() { return static_cast<Derived &>(*this); }
  __host__ __device__ const Derived &derived() const { return static_cast<const Derived &>(*this); }

public:
  __device__ inline const T &operator[](int group_id) const { return derived().devGet(group_id); }

protected:
  MemItemAccessor() {}
};

template <typename U> class HostMemItemHandle;
template <typename T> class HostMemItemAccessor : public MemItemAccessor<HostMemItemAccessor<T>, T> {
public:
  __host__ inline const T &get(int active_group_id) const {
    int offset = env_group_impl::EGActiveGroupMapper::hostGetGroupId(active_group_id) * group_stride_;
    return *(reinterpret_cast<T *>(host_ptr_ + offset));
  }

private:
  HostMemItemAccessor(const T *host_ptr, int group_stride) : host_ptr_(host_ptr), group_stride_(group_stride) {}

  __host__ static inline HostMemItemAccessor<T> assignAccessor(const T *host_ptr, int group_stride) {
    return HostMemItemAccessor<T>(host_ptr, group_stride);
  }

  const uint8_t *host_ptr_;
  const int group_stride_; // bytes
  friend class HostMemItemHandle<T>;
};

template <typename U> class ConstMemItemHandle;
template <typename T> class ConstMemItemAccessor : public MemItemAccessor<ConstMemItemAccessor<T>, T> {
public:
  __device__ inline const T &devGet(int active_group_id) const {
    return env_group_impl::EGConstantMemoryPool::getData<T>(active_group_id, pool_offset_);
  }

  __host__ inline void hostSet(int group_id, const T &data) const {
    int offset = env_group_impl::EGConstantMemoryPool::hostGetMemPoolOffset(group_id, pool_offset_);
    env_group_impl::EGConstantMemoryPool::setData(group_id, offset, data);
  }

protected:
  ConstMemItemAccessor(int mem_pool_offset) : pool_offset_(mem_pool_offset) {}

  __host__ static inline ConstMemItemAccessor<T> assignAccessor(int pool_offset) {
    return ConstMemItemAccessor<T>(pool_offset);
  }

  const int pool_offset_;
  friend class ConstMemItemHandle<T>;
};

template <typename U> class GlobalMemItemHandle;
template <typename T> class GlobalMemItemAccessor : public MemItemAccessor<GlobalMemItemAccessor<T>, T> {
public:
  __host__ __device__ inline T *getAddr(int group_id) {
    uint32_t offset = group_id * group_stride_;
    return reinterpret_cast<T *>(dev_ptr_ + offset);
  }

  __host__ __device__ inline const T *getAddr(int group_id) const {
    uint32_t offset = group_id * group_stride_;
    return reinterpret_cast<const T *>(dev_ptr_ + offset);
  }

  __device__ inline const T &get(int active_group_id) const {
    int group_id = env_group_impl::EGActiveGroupMapper::devGetGroupId(active_group_id);
    return *(getAddr(group_id));
  }

  __host__ inline void hostSet(int group_id, const T &data) const {
    uint32_t offset = group_id * group_stride_;
    checkCudaErrors(cudaMemcpy(offset + dev_ptr_, &data, sizeof(T), cudaMemcpyHostToDevice));
  }

protected:
  GlobalMemItemAccessor(const T *dev_ptr, int group_stride)
      : dev_ptr_(reinterpret_cast<const uint8_t *>(dev_ptr)), group_stride_(group_stride) {}

  __host__ static inline GlobalMemItemAccessor<T> assignAccessor(const T *dev_ptr) {
    uint32_t stride = ((sizeof(T) + 3) / 4) * 4; // 4字节对齐
    return GlobalMemItemAccessor<T>(dev_ptr, stride);
  }

  __host__ static inline GlobalMemItemAccessor<T> assignAccessor(const T *dev_ptr, uint32_t group_stride) {
    if (group_stride % 4 != 0) {
      throw std::runtime_error("group stride should align 4byte");
    }
    return GlobalMemItemAccessor<T>(dev_ptr, group_stride);
  }

  const uint8_t *dev_ptr_;
  const uint32_t group_stride_; // 单位：字节

  friend class GlobalMemItemHandle<T>;
};

template <typename U> class TensorItemHandle;
template <typename T> class TensorItemAccessor : public GlobalMemItemAccessor<T> {
public:
  __host__ __device__ int getNumElement() const { return numel_; }

protected:
  TensorItemAccessor(const T *dev_ptr, int group_stride, int numel)
      : GlobalMemItemAccessor<T>(dev_ptr, group_stride), numel_(numel) {}

  template <typename U>
  __host__ static inline TensorItemAccessor<T> assignAccessor(const T *dev_ptr, const std::vector<U> &shape) {
    int numel = 1;
    for (auto i : shape) {
      numel *= i;
    }
    // 假定使用连续存储
    return TensorItemAccessor<T>(dev_ptr, numel, numel * sizeof(T));
  }

  const int numel_;
  friend class TensorItemHandle<T>;
};

// ------------------- 配置项句柄 -------------------
class ItemHandleBase {
protected:
  MemoryType mem_type_;
  int num_group_;

public:
  ItemHandleBase(int num_group, MemoryType mem_type) : mem_type_(mem_type), num_group_(num_group) {}

  virtual ~ItemHandleBase() = default;

  virtual void syncToDevice() {}

  MemoryType getMemoryType() const { return mem_type_; }

  int getNumEnvGroup() const { return num_group_; }
};

// 主机内存区域，表明该项配置不会在
template <typename T> class HostMemItemHandle : public ItemHandleBase {
protected:
  std::vector<T> host_data_;

public:
  HostMemItemHandle(int num_group, MemoryType alternative = MemoryType::HOST_MEM)
      : ItemHandleBase(num_group, alternative), host_data_(num_group) {}
  virtual ~HostMemItemHandle() = default;

  __host__ T &groupAt(int group_id) { return host_data_[group_id]; }
  __host__ const T &groupAt(int group_id) const { return host_data_[group_id]; }

  __host__ T &activeGroupAt(int active_group_id) {
    int group_id = env_group_impl::EGActiveGroupMapper::hostGetGroupId(active_group_id);
    return host_data_[group_id];
  }
  __host__ const T &activeGroupAt(int active_group_id) const {
    int group_id = env_group_impl::EGActiveGroupMapper::hostGetGroupId(active_group_id);
    return host_data_[group_id];
  }

  std::vector<T> &hostData() { return host_data_; }

  HostMemItemAccessor<T> getAccessor() { return HostMemItemAccessor<T>(host_data_.data(), sizeof(T)); }
  const HostMemItemAccessor<T> getAccessor() const { return HostMemItemAccessor<T>(host_data_.data(), sizeof(T)); }
};

// 常量内存区域句柄，仅支持基本数据类型，包含常量内存池偏移量，主机内存句柄
template <typename T> class ConstMemItemHandle : public HostMemItemHandle<T> {
  int pool_offset_;

public:
  ConstMemItemHandle(int64_t num_group)
      : HostMemItemHandle<T>(num_group, MemoryType::CONSTANT_GPU_MEM),
        pool_offset_(env_group_impl::EGConstantMemoryPool::allocate<T>()) {}

  // 根据活跃组id同步到设备
  void syncToDevice() override {
    ConstMemItemAccessor<T> accessor = getAccessor();
    for (int ag = 0; ag < env_group_impl::EGActiveGroupMapper::hostGetNumActiveGroup(); ag++) {
      int group_id = env_group_impl::EGActiveGroupMapper::hostGetGroupId(ag);
      accessor.hostSet(group_id, HostMemItemHandle<T>::host_data_[group_id]);
    }
  }

  ConstMemItemAccessor<T> getAccessor() { return ConstMemItemAccessor<T>::assignAccessor(pool_offset_); }
  const ConstMemItemAccessor<T> getAccessor() const { return ConstMemItemAccessor<T>::assignAccessor(pool_offset_); }
};

// 全局内存区域句柄
template <typename T> class GlobalMemItemHandle : public HostMemItemHandle<T> {
  // std::vector<T, GlobalMemoryAllocator<T>> device_data_;
  T *dev_ptr_;
  // static constexpr int elem_size_ = (sizeof(T)+3)/4;
  static constexpr int elem_size_ = sizeof(T);

public:
  GlobalMemItemHandle(int64_t num_group) : HostMemItemHandle<T>(num_group, MemoryType::GLOBAL_GPU_MEM) {
    checkCudaErrors(cudaMalloc(&dev_ptr_, elem_size_ * num_group));
  }

  void syncToDevice() override {
    checkCudaErrors(cudaMemcpy(dev_ptr_, HostMemItemHandle<T>::host_data_.data(),
                               HostMemItemHandle<T>::host_data_.size() * elem_size_, cudaMemcpyHostToDevice));
  }

  GlobalMemItemAccessor<T> getAccessor() { return GlobalMemItemAccessor<T>::assignAccessor(dev_ptr_, elem_size_); }
  const GlobalMemItemAccessor<T> getAccessor() const {
    return GlobalMemItemAccessor<T>::assignAccessor(dev_ptr_, elem_size_);
  }
};

// 全局内存区域张量数据句柄，由TensorHandle管理
template <typename T> class TensorItemHandle : public ItemHandleBase {
  TensorHandle host_data_;
  TensorHandle device_data_;
  TensorShape shape_;

public:
  TensorItemHandle(const std::string &name, int64_t num_group, const TensorShape &shape)
      : ItemHandleBase(num_group, MemoryType::GLOBAL_TENSOR), shape_(shape) {

    // 在最开始插入num_group维度
    TensorShape new_shape = shape;
    new_shape.insert(new_shape.begin(), num_group);

    // 创建配置项的内存张量
    TensorRegistry::getInstance().createTensor<T>(host_data_, name + "_Config@CPU", new_shape, DeviceType::kCPU);
    TensorRegistry::getInstance().createTensor<T>(device_data_, name + "_Config@CUDA", new_shape, DeviceType::kCUDA);
  }

  void syncToDevice() override { host_data_.copyTo(device_data_); }

  template <typename... Args> __host__ auto groupAt(int64_t group_id, Args... indices) {
    if (group_id >= num_group_) {
      throw std::out_of_range("group_id out of range");
    }
    return host_data_[{group_id, indices...}];
  }

  template <typename... Args> __host__ const auto groupAt(int64_t group_id, Args... indices) const {
    if (group_id >= num_group_) {
      throw std::out_of_range("group_id out of range");
    }
    return host_data_[{group_id, indices...}];
  }

  template <typename... Args> __host__ auto activeGroupAt(int64_t active_group_id, Args... indices) {
    if (active_group_id >= env_group_impl::EGActiveGroupMapper::hostGetNumActiveGroup()) {
      throw std::out_of_range("active_group_id out of range");
    }
    int group_id = env_group_impl::EGActiveGroupMapper::hostGetGroupId(active_group_id);
    return host_data_[{group_id, indices...}];
  }

  template <typename... Args> __host__ const auto activeGroupAt(int64_t active_group_id, Args... indices) const {
    if (active_group_id >= env_group_impl::EGActiveGroupMapper::hostGetNumActiveGroup()) {
      throw std::out_of_range("active_group_id out of range");
    }
    int group_id = env_group_impl::EGActiveGroupMapper::hostGetGroupId(active_group_id);
    return host_data_[{group_id, indices...}];
  }

  TensorItemAccessor<T> getAccessor() {
    return TensorItemAccessor<T>::assignAccessor(device_data_.typed_data<T>(), shape_);
  }
  const TensorItemAccessor<T> getAccessor() const {
    return TensorItemAccessor<T>::assignAccessor(device_data_.typed_data<T>(), shape_);
  }
};

// ------------------- 环境组管理器 -------------------
class EnvGroupManager {
public:
  constexpr static int SHAPE_PLACEHOLDER_ACTIVE_GROUP = -100;
  constexpr static int SHAPE_PLACEHOLDER_ENV = -101;

  EnvGroupManager(int num_env_per_group = 1, int num_group = 1, int num_active_group = 1)
      : num_env_per_group_(num_env_per_group), num_group_(num_group), num_active_group_(num_active_group) {
    if (num_active_group > num_group) {
      throw std::runtime_error("num_active_group should be less than or equal to num_group");
    }
    if (num_active_group > MAX_NUM_ACTIVE_GROUP) {
      throw std::runtime_error("num_active_group exceed the max number of active groups");
    }
    if (num_env_per_group <= 0 || num_group <= 0 || num_active_group <= 0) {
      throw std::runtime_error("num_env_per_group, num_group, num_active_group should be greater than 0");
    }
    env_group_impl::EGActiveGroupMapper::hostSetNumActiveGroup(num_active_group);
    env_group_impl::EGConstantMemoryPool::updateConfig();
  }

  template <typename T, MemoryType mem_type> auto registerConfigItem(const std::string &name) {
    if (registry_.find(name) != registry_.end()) {
      throw std::runtime_error("name already exists in registry, please check your code.");
    }

    if constexpr (mem_type == MemoryType::CONSTANT_GPU_MEM) {
      std::unique_ptr<ConstMemItemHandle<T>> item = std::make_unique<ConstMemItemHandle<T>>(num_group_);
      ConstMemItemHandle<T> *ptr = item.get();
      registry_.insert({name, std::move(item)});
      return ptr;
    } else if constexpr (mem_type == MemoryType::GLOBAL_GPU_MEM) {
      std::unique_ptr<GlobalMemItemHandle<T>> item = std::make_unique<GlobalMemItemHandle<T>>(num_group_);
      GlobalMemItemHandle<T> *ptr = item.get();
      registry_.insert({name, std::move(item)});
      return ptr;
    } else if constexpr (mem_type == MemoryType::HOST_MEM) {
      std::unique_ptr<HostMemItemHandle<T>> item = std::make_unique<HostMemItemHandle<T>>(num_group_);
      HostMemItemHandle<T> *ptr = item.get();
      registry_.insert({name, std::move(item)});
      return ptr;
    } else {
      throw std::runtime_error("Unsupported memory type");
    }
  }

  template <typename T> auto registerConfigTensor(const std::string &name, const TensorShape &shape) {
    if (registry_.find(name) != registry_.end()) {
      throw std::runtime_error("name already exists in registry, please check your code.");
    }
    std::unique_ptr<TensorItemHandle<T>> item = std::make_unique<TensorItemHandle<T>>(name, num_group_, shape);
    TensorItemHandle<T> *ptr = item.get();
    registry_.insert({name, std::move(item)});
    return ptr;
  }

  // 返回新Tensor引用，包含了Group和Env维度
  TensorHandle &createTensor(const std::string &name, const TensorShape &shape_with_placeholder,
                             NumericalDataType dtype, DeviceType device_type = DeviceType::kCUDA) {
    std::vector<int64_t> shape;
    for (auto s : shape_with_placeholder) {
      if (s == SHAPE_PLACEHOLDER_ACTIVE_GROUP) {
        // TODO. 这里使用的是num_group，因为某些核函数需要数据紧密排列。因此需要在调整group数量后的二次分配
        shape.push_back(num_active_group_);
      } else if (s == SHAPE_PLACEHOLDER_ENV) {
        shape.push_back(num_env_per_group_);
      } else {
        shape.push_back(s);
      }
    }
    return TensorRegistry::getInstance().createTensor(name, shape, dtype, device_type);
  }

  // 原地替换Tensor内部数据
  void createTensor(TensorHandle &target, const std::string &name, const TensorShape &shape_with_placeholder,
                    NumericalDataType dtype, DeviceType device_type = DeviceType::kCUDA) {

    std::vector<int64_t> shape;
    for (auto s : shape_with_placeholder) {
      if (s == SHAPE_PLACEHOLDER_ACTIVE_GROUP) {
        shape.push_back(num_group_);
      } else if (s == SHAPE_PLACEHOLDER_ENV) {
        shape.push_back(num_env_per_group_);
      } else {
        shape.push_back(s);
      }
    }
    TensorRegistry::getInstance().createTensor(target, name, shape, dtype, device_type);
  }

  // 返回新Tensor引用（模板类型）
  template <typename T>
  TensorHandle &createTensor(const std::string &name, const TensorShape &shape_with_placeholder,
                             DeviceType device_type = DeviceType::kCUDA) {

    return createTensor(name, shape_with_placeholder, TensorHandle::convertTypeToTensorType<T>(), device_type);
  }
  // 原地替换Tensor内部数据
  template <typename T>
  void createTensor(TensorHandle &target, const std::string &name, const TensorShape &shape_with_placeholder,
                    DeviceType device_type = DeviceType::kCUDA) {

    createTensor(target, name, shape_with_placeholder, TensorHandle::convertTypeToTensorType<T>(), device_type);
  }

  void sampleActiveGroupIndices() {
    // TODO. 目前是顺序取，后续可以考虑随机取
    std::vector<int> active_indices;
    for (int i = 0; i < num_active_group_; i++) {
      active_indices.push_back(i);
    }
    // 替换当前的活跃组，同时更新ActiveGroup
    env_group_impl::EGActiveGroupMapper::replaceActiveGroups(active_indices);
    env_group_impl::EGConstantMemoryPool::updateConfig();
  }

  void syncToDevice() {
    for (auto &[name, item] : registry_) {
      item->syncToDevice();
    }
  }

  int getNumGroup() const { return num_group_; }
  int getNumActiveGroup() const { return num_active_group_; }
  int getNumEnvPerGroup() const { return num_env_per_group_; }

private:
  // 每个环境组的环境数量
  const int num_env_per_group_;
  // 当前环境组的数量
  const int num_group_;
  // 当前活跃环境组的数量
  const int num_active_group_;
  // 环境组配置项注册表
  std::unordered_map<std::string, std::unique_ptr<ItemHandleBase>> registry_;
};

} // namespace core
} // namespace cuda_simulator

#endif // CUDASIM_ENV_GROUP_MANAGER_HH