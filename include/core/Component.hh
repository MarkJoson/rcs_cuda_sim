#ifndef CUDASIM_COMPONENT_HH
#define CUDASIM_COMPONENT_HH

#include <string>
#include <unordered_map>


namespace cuda_simulator
{
namespace core
{

class SimulatorContext;

class ComponentBase
{
public:
    ComponentBase(const std::string &name, const std::string &graph)
        : component_name_(name), exec_graph_(graph) { }
    virtual ~ComponentBase() {}

    virtual void onRegister(SimulatorContext* context) = 0;

    virtual void onEnvironGroupInit(SimulatorContext* context) = 0;

    virtual void onExecute(
        SimulatorContext* context,
        const std::unordered_map<std::string, GTensor> input,
        const std::unordered_map<std::string, GTensor> output) = 0;

    virtual void onReset(
        GTensor reset_flags,
        std::unordered_map<std::string, GTensor> &state) = 0; //


    const std::string& getName() const { return component_name_; }

    const std::string& getExecGraph() const { return exec_graph_; }

private:
    std::string component_name_;
    std::string exec_graph_;
};

} // namespace core
} // namespace cuda_simulator


#endif // CUDASIM_COMPONENT_HH