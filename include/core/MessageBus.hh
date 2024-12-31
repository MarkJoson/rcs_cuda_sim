#ifndef CUDASIM_MESSAGEBUS_HH
#define CUDASIM_MESSAGEBUS_HH

#include "Component.hh"

namespace cuda_simulator
{
namespace core
{

class MessageBus
{
public:
    MessageBus() {}
    virtual ~MessageBus() {}

    void registerNode(ComponentBase* component);
    void registerInput() {}
    void registerOutput() {}
};

} // namespace core
} // namespace cuda_simulator


#endif // CUDASIM_MESSAGEBUS_HH