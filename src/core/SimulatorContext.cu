#include "core/SimulatorContext.hh"
#include "core/MessageBus.hh"

namespace cuda_simulator {
namespace core {

void SimulatorContext::initialize()  {
    message_bus = std::make_unique<MessageBus>(this);
    env_group_manager = std::make_unique<EnvGroupManager>(1,2,1);
    geometry_manager = std::make_unique<geometry::GeometryManager>();
}

void SimulatorContext::setup(const std::vector<std::string> &entrances) {
    // 创建所有节点的依赖列表
    createDepSeq();

    // 调用组件的初始化函数
    for(auto &com_id : dep_seq) {
        components[com_id]->onNodeInit();
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

    // 调用组件的启动函数
    for(auto &com_id : dep_seq) {
        components[com_id]->onNodeStart();
    }
}

void SimulatorContext::trigger(const std::string &trigger_tag) {
    message_bus->trigger(trigger_tag);
}


void SimulatorContext::createDepSeq()  {
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

} // namespace core
} // namespace cuda_simulator