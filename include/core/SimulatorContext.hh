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
    }
    virtual ~SimulatorContext() {}

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
};

} // namespace core
} // namespace cuda_simulator



#endif // CUDASIM_SIMULATOR_CONTEXT_HH