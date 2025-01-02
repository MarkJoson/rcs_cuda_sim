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
    void getEnvironGroupManager() {}

    void loadBasicComponent() {}

private:
    std::unique_ptr<MessageBus> message_bus;
};

} // namespace core
} // namespace cuda_simulator



#endif // CUDASIM_SIMULATOR_CONTEXT_HH