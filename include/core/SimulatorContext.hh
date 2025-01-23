#ifndef CUDASIM_SIMULATOR_CONTEXT_HH
#define CUDASIM_SIMULATOR_CONTEXT_HH

#include "Component.hh"
#include "core_types.hh"
#include <memory>

#include "EnvGroupManager.cuh"

namespace cuda_simulator {
namespace core {

class MessageBus;
// class EnvGroupManager;

namespace geometry {
class GeometryManager;
}

class SimulatorContext final {
public:
  SimulatorContext();
  ~SimulatorContext();

  static SimulatorContext *getContext() {
    static SimulatorContext context;
    return &context;
  }

  static inline void setDefaultCudaDeviceId(int device_id) {
    GTensor::setTensorDefaultDeviceId(device_id);
  }

  static inline void setSeed(uint64_t seed) {
    GTensor::setSeed(seed);
  }

  template <typename T, typename... Args> T *createComponent(Args... args) {
    std::unique_ptr<T> com_handle = std::make_unique<T>(args...);

    Component *com = pushComponent(std::move(com_handle));
    return dynamic_cast<T *>(com);
  }

  // 添加component到注册表
  Component *pushComponent(std::unique_ptr<Component> &&component);
  // 获得MessageBus实例
  MessageBus *getMessageBus();
  // 获得EnvGroupMgr实例
  EnvGroupManager *getEnvGroupMgr();
  // 获得GeometryManager实例
  geometry::GeometryManager *getGeometryManager();
  // 获得InputInfo
  Component::NodeInputInfo getInputInfo(const NodeNameRef &component_name,
                                        const MessageName &message_name);
  // 获得OutputInfo
  Component::NodeOutputInfo getOutputInfo(const NodeNameRef &component_name,
                                          const MessageName &message_name);
  void initialize(int num_env_per_group = 1, int num_group = 1, int num_active_group = 1);
  // 初始化组件
  void setup(const std::vector<NodeTagRef> &entrances = {"default", "observe"});
  // 运行
  void trigger(const NodeTagRef &name);

private:
  class Impl;
  std::unique_ptr<Impl> impl;
};

// 工具函数
static inline SimulatorContext *getContext() {
  return SimulatorContext::getContext();
}

static inline EnvGroupManager *getEnvGroupMgr() {
  return SimulatorContext::getContext()->getEnvGroupMgr();
}

static inline MessageBus *getMessageBus() {
  return SimulatorContext::getContext()->getMessageBus();
}

static inline geometry::GeometryManager *getGeometryManager() {
  return SimulatorContext::getContext()->getGeometryManager();
}

} // namespace core
} // namespace cuda_simulator

#endif // CUDASIM_SIMULATOR_CONTEXT_HH
