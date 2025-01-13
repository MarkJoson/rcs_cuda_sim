#ifndef CUDASIM_SIMULATOR_CONTEXT_HH
#define CUDASIM_SIMULATOR_CONTEXT_HH

#include <memory>
#include "MessageBus.hh"
#include "Component.hh"
#include "EnvGroupManager.cuh"

namespace cuda_simulator
{
namespace core
{

class SimulatorContext
{
public:
    SimulatorContext() {
        message_bus = std::make_unique<MessageBus>(this);
        env_group_manager = std::make_unique<EnvGroupManager>(1,2,1);
    }
    virtual ~SimulatorContext() {}

    template<typename T, typename ...Args>
    T* createComponent(Args... args) {
        components.push_back(std::make_unique<T>(args...));
        return components.back().get();
    }

    void registerComponent() {}

    MessageBus* getMessageBus() {
        return message_bus.get();
    }

    void setDefaultCudaDeviceId(int device_id) {
        GTensor::setTensorDefaultDeviceId(device_id);
    }

    void setSeed(uint64_t seed) {
        GTensor::setSeed(seed);
    }

    void initialize() {
    }


    EnvGroupManager* getEnvironGroupManager() {
        return env_group_manager.get();
    }

    void loadBasicComponent() {}

private:
    std::unique_ptr<MessageBus> message_bus;
    std::unique_ptr<EnvGroupManager> env_group_manager;
    std::vector<std::unique_ptr<Component>> components;
};

} // namespace core
} // namespace cuda_simulator



#endif // CUDASIM_SIMULATOR_CONTEXT_HH