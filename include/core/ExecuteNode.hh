#ifndef CUDASIM_EXECUTENODE_HH
#define CUDASIM_EXECUTENODE_HH

#include "core_types.hh"

namespace cuda_simulator {
namespace core {

class SimulatorContext;

class ExecuteNode {
public:
    ExecuteNode(const NodeName &name, const NodeTag &tag = "default")
        : name_(name), tag_(tag) {}
    virtual ~ExecuteNode() = default;

    // MessageBus调用的接口
    virtual void onRegister(SimulatorContext* context) = 0;
    virtual void onExecute(
        SimulatorContext* context,
        const NodeExecInputType &input,
        NodeExecOutputType &output) = 0;

    // Getters
    const NodeName& getName() const { return name_; }
    const NodeTag& getTag() const { return tag_; }

protected:
    NodeName name_;
    NodeTag tag_;
};

} // namespace core
} // namespace cuda_simulator

#endif // CUDASIM_EXECUTENODE_HH
