#include <stdexcept>
#include <boost/graph/topological_sort.hpp>
#include <boost/graph/adjacency_list.hpp>

#include "core/core_types.hh"
#include "core/SimulatorContext.hh"
#include "core/MessageBus.hh"
#include "core/EnvGroupManager.cuh"

#include "geometry/GeometryManager.cuh"

namespace cuda_simulator {
namespace core {


class SimulatorContext::Impl {
public:
    Impl() {}

    Component* pushComponent(std::unique_ptr<Component>&& component) {
        if(component_map.find(component->getName()) != component_map.end()) {
            throw std::runtime_error("Component has been registered!");
        }

        ComponentId com_id = components.size();
        component_map[component->getName()] = com_id;
        components.push_back(std::move(component));
        return components[com_id].get();
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

    Component::NodeInputInfo getInputInfo(const NodeNameRef &component_name, const MessageName &message_name) {
        auto com_id = component_map.find(component_name);
        if(com_id == component_map.end()) {
            throw std::runtime_error("Component not found");
        }

        return components.at(com_id->second)->getInputInfo(message_name);
    }

    Component::NodeOutputInfo getOutputInfo(const NodeNameRef &component_name, const MessageName &message_name) {
        auto com_id = component_map.find(component_name);
        if(com_id == component_map.end()) {
            throw std::runtime_error("Component not found");
        }

        return components.at(com_id->second)->getOutputInfo(message_name);
    }

    // 初始化子系统: MessageBus, EnvGroupManager
    void initialize() {
        message_bus = std::make_unique<MessageBus>();
        env_group_manager = std::make_unique<EnvGroupManager>(1,1,1);
        geometry_manager = std::make_unique<geometry::GeometryManager>();
    }

    // 初始化组件
    void setup(const std::vector<NodeTagRef> &entrances = {"default", "observe"}) {
        // 创建所有节点的依赖列表
        createDependSeqence();

        // 调用组件的初始化函数
        for(auto &com_id : dep_seq) {
            components[com_id]->onNodeInit();
        }

        for(auto &com_id : dep_seq) {
            //components[com_id]->onEnvironInit();
            message_bus->registerComponent(components[com_id].get());
            for(auto &input : components[com_id]->getInputs()) {
                message_bus->registerInput(components[com_id].get(), input.second);
            }
            for(auto &output : components[com_id]->getOutputs()) {
                message_bus->registerOutput(components[com_id].get(), output.second);
            }
        }

        // 添加计算图入口
        for(const auto& ent : entrances) {
            message_bus->addTrigger({ent});
        }

        // 创建计算图
        message_bus->buildGraph();

        // 调用组件的环境组初始化函数
        for(auto &com_id : dep_seq) {
            components[com_id]->onEnvironGroupInit();
        }

        // TODO. 修改GeometryManager和EnvGroupManager的标志位，禁止再次添加、修改或删除任何配置

        // 组装环境，生成ESDF地图，生成动态物体边集合
        geometry_manager->assemble();

        // 采样一组环境，并将环境组的活跃参数同步到GPU设备
        env_group_manager->sampleActiveGroupIndices();
        env_group_manager->syncToDevice();

        // 调用组件的启动函数
        for(auto &com_id : dep_seq) {
            components[com_id]->onNodeStart();
        }
    }

    // 运行
    void trigger(const NodeTagRef &trigger_tag) {
        // 调用计算图触发对应的启动节点
        message_bus->trigger(trigger_tag);
    }

protected:
    // 生成依赖序列
    void createDependSeqence() {
        for(size_t com_id = 0; com_id < components.size(); com_id++) {
            boost::add_vertex({com_id}, dependency_graph);
        }

        for(auto &com : components) {
            for(auto &dep : com->getDependences()) {
                if(component_map.find(dep) == component_map.end()) {
                    throw std::runtime_error("Dependence not found");
                }
                ComponentId dep_id = component_map[dep];
                boost::add_edge(dep_id, components.size()-1, dependency_graph);
            }
        }

        std::list<Graph::vertex_descriptor> topo_order;
        boost::topological_sort(dependency_graph, std::front_inserter(topo_order));

        std::transform(topo_order.begin(), topo_order.end(), std::back_inserter(dep_seq),
            [this](auto v) {
            return dependency_graph[v];
        });
    }

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

SimulatorContext::SimulatorContext() : impl(std::make_unique<Impl>()) { }

SimulatorContext::~SimulatorContext() = default;

Component* SimulatorContext::pushComponent(std::unique_ptr<Component> &&component) {
    return impl->pushComponent(std::move(component));
}

Component::NodeInputInfo SimulatorContext::getInputInfo(const NodeNameRef &component_name, const MessageName &message_name) {
    return impl->getInputInfo(component_name, message_name);
}

Component::NodeOutputInfo SimulatorContext::getOutputInfo(const NodeNameRef &component_name, const MessageName &message_name) {
    return impl->getOutputInfo(component_name, message_name);
}

// 获得EnvGroupMgr实例
EnvGroupManager* SimulatorContext::getEnvGroupMgr() {
    return impl->getEnvGroupMgr();
}

// 获得GeometryManager实例
geometry::GeometryManager* SimulatorContext::getGeometryManager() {
    return impl->getGeometryManager();
}

// 初始化子系统: MessageBus, EnvGroupManager
void SimulatorContext::initialize() {
    impl->initialize();
}

void SimulatorContext::setup(const std::vector<NodeTagRef> &entrances) {
    impl->setup(entrances);
}

void SimulatorContext::trigger(const NodeTagRef &name) {
    impl->trigger(name);
}

} // namespace core
} // namespace cuda_simulator