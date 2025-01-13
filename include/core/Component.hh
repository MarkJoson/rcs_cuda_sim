#ifndef CUDASIM_COMPONENT_HH
#define CUDASIM_COMPONENT_HH

#include <string>
#include "core_types.hh"
#include "ExecuteNode.hh"

namespace cuda_simulator {
namespace core {

class SimulatorContext;

class Component : public ExecuteNode {
public:
    Component(const NodeName &name, const NodeTag &tag = "default")
        : ExecuteNode(name, tag) {}
    virtual ~Component() = default;

    virtual void onEnvironGroupInit() = 0;

    void addDependence(const std::string &dependence) {
        dependences_.push_back(dependence);
    }

    const std::vector<std::string> &getDependences() const {
        return dependences_;
    }

    using ExecuteNode::getName;
    using ExecuteNode::getTag;

protected:
    std::vector<std::string> dependences_;
};


template<typename Derived>
class MultiInstComponent : public Component {
public:
    using Base = MultiInstComponent<Derived>::Component;
    using CountableBase = MultiInstComponent<Derived>;

    MultiInstComponent(const NodeName &name, const NodeTag &tag = "default")
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