#ifndef CUDASIM_SIMULATOR_CONTEXT_HH
#define CUDASIM_SIMULATOR_CONTEXT_HH

#include <memory>
#include "MessageBus.hh"
#include "Component.hh"


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

    void setDefaultDevice(const std::string &device_name ) {
        GTensor::setTensorDefaultDevice(device_name);
    }

    void setSeed(uint64_t seed) {
        GTensor::setSeed(seed);
    }

    void initialize() {
    }

    void getEnvironGroupManager() {}

    void loadBasicComponent() {}

private:
    std::unique_ptr<MessageBus> message_bus;
};

} // namespace core
} // namespace cuda_simulator



#endif // CUDASIM_SIMULATOR_CONTEXT_HH