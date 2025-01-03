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


    // 中介转发
    void registerInput(
        Component* component,
        const MessageNameRef &message_name,
        const MessageShape &shape,
        int history_offset = 0,
        ReduceMethod reduce_method = ReduceMethod::STACK
    ) {
        message_bus->registerInput(component, message_name, shape, history_offset, reduce_method);
    }


    void registerOutput(
        Component* component,
        const MessageNameRef &message_name,
        const MessageShape &shape,
        std::optional<TensorHandle> history_padding_val = std::nullopt
    ) {
        message_bus->registerOutput(component, message_name, shape, history_padding_val);
    }


    void getEnvironGroupManager() {}

    void loadBasicComponent() {}

private:
    std::unique_ptr<MessageBus> message_bus;
};

} // namespace core
} // namespace cuda_simulator



#endif // CUDASIM_SIMULATOR_CONTEXT_HH