#ifndef CUDASIM_COMPONENT_HH
#define CUDASIM_COMPONENT_HH

#include <string>
#include <unordered_map>
#include "storage/ITensor.h"

namespace cuda_simulator
{
namespace core
{

class SimulatorContext;

class ComponentBase
{
public:
    ComponentBase(const std::string &name, const std::string &tag = "default")
        : component_name_(name), tag_(tag) { }
    virtual ~ComponentBase() {}

    virtual void onRegister(SimulatorContext* context) = 0;

    virtual void onEnvironGroupInit(SimulatorContext* context) = 0;

    virtual void onExecute(
        SimulatorContext* context,
        const std::unordered_map<std::string, TensorHandle> input,
        const std::unordered_map<std::string, TensorHandle> output) = 0;

    virtual void onReset(
        TensorHandle reset_flags,
        std::unordered_map<std::string, TensorHandle> &state) = 0; //


    const std::string& getName() const { return component_name_; }

    const std::string& getTag() const { return tag_; }

private:
    std::string component_name_;
    std::string tag_;
};

} // namespace core
} // namespace cuda_simulator


#endif // CUDASIM_COMPONENT_HH