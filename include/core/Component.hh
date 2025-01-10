#ifndef CUDASIM_COMPONENT_HH
#define CUDASIM_COMPONENT_HH

#include <string>
#include "core_types.hh"
#include "storage/GTensorConfig.hh"
#include "ExecuteNode.hh"

namespace cuda_simulator {
namespace core {

class SimulatorContext;

class Component : public ExecuteNode {
public:
    Component(const NodeName &name, const NodeTag &tag = "default")
        : ExecuteNode(name, tag) {}
    virtual ~Component() = default;

    // Component特有的接口
    virtual void onEnvironGroupInit(SimulatorContext* context) = 0;
    virtual void onReset(
        const TensorHandle& reset_flags,
        NodeExecStateType &state) = 0;
};

template<typename Derived>
class CountableComponent : public Component {
public:
    using Base = CountableComponent<Derived>::Component;
    using CountableBase = CountableComponent<Derived>;

    CountableComponent(const NodeName &name, const NodeTag &tag = "default")
        : Base("#TBD", tag), inst_id_(componentInstCntInc()) {
            name_ = name + "_" + std::to_string(inst_id_);
        }

    static int componentInstCntInc() {
        static int id = 0;
        return id++;
    }

    int getInstId() const { return inst_id_; }

protected:
    int inst_id_;
};

} // namespace core
} // namespace cuda_simulator

#endif // CUDASIM_COMPONENT_HH