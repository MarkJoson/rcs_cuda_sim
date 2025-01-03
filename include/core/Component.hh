#ifndef CUDASIM_COMPONENT_HH
#define CUDASIM_COMPONENT_HH

#include <string>
#include <unordered_map>

#include "core_types.hh"
#include "storage/GTensor.hh"

namespace cuda_simulator
{
namespace core
{

class SimulatorContext;

class ComponentBase
{
public:
    ComponentBase(const NodeName &name, const NodeTag &tag = "default")
        : component_name_(name), tag_(tag) { }
    virtual ~ComponentBase() {}

    virtual void onRegister(SimulatorContext* context) = 0;

    virtual void onEnvironGroupInit(SimulatorContext* context) = 0;

    virtual void onExecute(
        SimulatorContext* context,
        const NodeExecInputType &input,
        const NodeExecOutputType &output) = 0;

    virtual void onReset(
        TensorHandle reset_flags,
        NodeExecStateType &state) = 0; //


    const NodeName& getName() const { return component_name_; }

    const NodeTag& getTag() const { return tag_; }

private:
    NodeName component_name_;
    NodeTag tag_;
};

} // namespace core
} // namespace cuda_simulator


#endif // CUDASIM_COMPONENT_HH