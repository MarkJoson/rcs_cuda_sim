#include "core/SimulatorContext.hh"
#include "core/MessageBus.hh"
#include "core/EnvGroupManager.cuh"
#include "geometry/GeometryManager.cuh"

namespace cuda_simulator {
namespace core {

void SimulatorContext::initialize()  {
    message_bus = std::make_unique<MessageBus>();
    env_group_manager = std::make_unique<EnvGroupManager>(1,1,1);
    geometry_manager = std::make_unique<geometry::GeometryManager>();
}

void SimulatorContext::setup(const std::vector<std::string> &entrances) {
    // 创建所有节点的依赖列表
    createDepSeq();

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

    // 将环境组的活跃参数同步到GPU设备
    env_group_manager->syncToDevice();

    // 调用组件的启动函数
    for(auto &com_id : dep_seq) {
        components[com_id]->onNodeStart();
    }

}

void SimulatorContext::trigger(const std::string &trigger_tag) {
    // 调用计算图触发对应的启动节点
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