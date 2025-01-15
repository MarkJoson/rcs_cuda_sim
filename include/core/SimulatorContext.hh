#ifndef CUDASIM_SIMULATOR_CONTEXT_HH
#define CUDASIM_SIMULATOR_CONTEXT_HH

#include <boost/graph/graph_selectors.hpp>
#include <boost/graph/topological_sort.hpp>
#include <memory>
#include <boost/graph/adjacency_list.hpp>
#include <string_view>
#include <unordered_map>

#include "Component.hh"
#include "storage/GTensorConfig.hh"


namespace cuda_simulator
{
namespace core
{
namespace geometry {
    class GeometryManager;
}

class MessageBus;
class EnvGroupManager;


class SimulatorContext final
{
public:
    SimulatorContext() { }
    ~SimulatorContext() {}

    static SimulatorContext* getContext() {
        static SimulatorContext context;
        return &context;
    }

    MessageBus* getMessageBus() {
        if(!message_bus) {
            throw std::runtime_error("Context not initialized");
        }
        return message_bus.get();
    }

    EnvGroupManager* getEnvGroupMgr() {
        if(!env_group_manager) {
            throw std::runtime_error("Context not initialized");
        }
        return env_group_manager.get();
    }

    geometry::GeometryManager* getGeometryManager() {
        if(!geometry_manager) {
            throw std::runtime_error("Context not initialized");
        }
        return geometry_manager.get();
    }

    static inline void setDefaultCudaDeviceId(int device_id) {
        GTensor::setTensorDefaultDeviceId(device_id);
    }

    static inline void setSeed(uint64_t seed) {
        GTensor::setSeed(seed);
    }

    template<typename T, typename ...Args>
    T* createComponent(Args... args) {
        std::unique_ptr<T> com_handle = std::make_unique<T>(args...);
        if(component_map.find(com_handle->getName()) != component_map.end()) {
            throw std::runtime_error("Component has been registered!");
        }

        ComponentId com_id = components.size();
        component_map[com_handle->getName()] = com_id;
        components.push_back(std::move(com_handle));

        return dynamic_cast<T*>(components[com_id].get());
    }

    std::optional<Component::NodeInputInfo> getInputInfo(const std::string &component_name, const std::string &message_name) {
        auto com_id = component_map.find(component_name);
        if(com_id == component_map.end()) {
            return std::nullopt;
        }

        return components[com_id->second]->getInputInfo(message_name);
    }

    std::optional<Component::NodeOutputInfo> getOutputInfo(const std::string &component_name, const std::string &message_name) {
        auto com_id = component_map.find(component_name);
        if(com_id == component_map.end()) {
            return std::nullopt;
        }

        return components[com_id->second]->getOutputInfo(message_name);
    }

    // 初始化子系统: MessageBus, EnvGroupManager
    void initialize();

    // 初始化组件
    void setup(const std::vector<std::string> &entrances = {"default", "observe"});

    void trigger(const std::string &name);

protected:
    // 生成依赖序列
    void createDepSeq();

private:
    using ComponentName = std::string_view;
    using ComponentId = std::size_t;
    using Graph = boost::adjacency_list<boost::vecS, boost::vecS, boost::bidirectionalS, ComponentId>;

    // using
    std::unique_ptr<MessageBus> message_bus;
    std::unique_ptr<EnvGroupManager> env_group_manager;
    std::unique_ptr<geometry::GeometryManager> geometry_manager;

    std::vector<std::unique_ptr<Component>> components;
    std::unordered_map<ComponentName, ComponentId> component_map;

    Graph dependency_graph;
    std::vector<ComponentId> dep_seq;
};

// 工具函数
static inline SimulatorContext* getContext(){
    return SimulatorContext::getContext();
}

static inline EnvGroupManager *getEnvGroupMgr(){
    return SimulatorContext::getContext()->getEnvGroupMgr();
}

static inline MessageBus *getMessageBus() {
    return SimulatorContext::getContext()->getMessageBus();
}

static inline geometry::GeometryManager *getGeometryManager() {
    return SimulatorContext::getContext()->getGeometryManager();
}

} // namespace core
} // namespace cuda_simulator

#endif // CUDASIM_SIMULATOR_CONTEXT_HH
