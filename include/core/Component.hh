#ifndef CUDASIM_COMPONENT_HH
#define CUDASIM_COMPONENT_HH

#include <string>
#include "core_types.hh"
#include "storage/GTensor.hh"

namespace cuda_simulator
{
namespace core
{

class SimulatorContext;

class Component
{
public:
    Component(const NodeName &name, const NodeTag &tag = "default")
        : name_(name), tag_(tag) { }
    virtual ~Component() {}

    virtual void onRegister(SimulatorContext* context) = 0;

    virtual void onEnvironGroupInit(SimulatorContext* context) = 0;

    virtual void onExecute(
        SimulatorContext* context,
        const NodeExecInputType &input,
        const NodeExecOutputType &output) = 0;

    virtual void onReset(
        TensorHandle reset_flags,
        NodeExecStateType &state) = 0; //

    const NodeName& getName() const { return name_; }
    const NodeTag& getTag() const { return tag_; }


protected:
    NodeName name_;
    NodeTag tag_;
};


template<typename Derived>
class CountableComponent : public Component {
public:
    using Base = CountableComponent<Derived>::Component;
    using CountableBase = CountableComponent<Derived>;

    CountableComponent(const NodeName &name, const NodeTag &tag = "default")
        : Base("#TBD", tag), inst_id_(componentInstCntInc()) {
            name_ = name + "_" + std::to_string(inst_id_);  // 修改Component的name，增加实例id
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