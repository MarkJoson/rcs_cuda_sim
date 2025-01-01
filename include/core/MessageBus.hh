#ifndef CUDASIM_MESSAGEBUS_HH
#define CUDASIM_MESSAGEBUS_HH


#include <string>
#include <optional>
#include <iostream>
#include <deque>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/topological_sort.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/property_map/property_map.hpp>
#include <boost/graph/properties.hpp>
#include <boost/graph/breadth_first_search.hpp>
#include "Component.hh"
#include "storage/ITensor.h"

namespace cuda_simulator
{
namespace core
{

// 枚举定义
enum class ReduceMethod {
    STACK,      // 堆叠
    REPLACE,    // 替换
    SUM,        // 求和
    MAX,        // 求最大值
    MIN,        // 求最小值
    AVERAGE     // 求平均值
};

class MessageQueue {
public:
    explicit MessageQueue(size_t maxHistoryLen)
        : maxHistoryLen_(maxHistoryLen)
        , validCount_(0)
        , latestUpdateTime_(0) {}

    ITensor* getHistory(int offset) {
        if (auto it = cache_.find(offset); it != cache_.end()) {
            return it->second;
        }

        auto result = history_[history_.size() - 1 - offset];
        cache_[offset] = result;
        return result;
    }

    void append(ITensor*& data) {
        cache_.clear();
        history_.push_back(data);
        if (history_.size() > maxHistoryLen_) {
            history_.pop_front();
        }
        validCount_ = std::min(maxHistoryLen_, validCount_ + 1);
        latestUpdateTime_++;
    }

    void reset() {
        history_.clear();
        cache_.clear();
        validCount_ = 0;
        latestUpdateTime_ = 0;
    }

    size_t getValidCount() const { return validCount_; }

private:
    std::deque<ITensor*> history_;
    size_t maxHistoryLen_;
    size_t validCount_;
    size_t latestUpdateTime_;
    std::unordered_map<int, ITensor*> cache_;
};

using NodeName = std::string;
using NodeId = std::uint32_t;

using MessageName = std::string;
using MessageId = std::uint32_t;
using MessageShape = std::vector<int64_t>;

struct InputDescription {
    // 输入组件id
    // NodeId node_id;
    NodeName node_name;
    // 输入消息id
    MessageName message_name;
    // 输入消息形状
    MessageShape shape;
    // 消息历史偏移
    int history_offset;
    // 无效历史数据的填充值
    std::optional<ITensor*> history_padding_val;
    // 多消息时的归约方法，当reduce_method为STACK时，stack_dim和stack_order有效
    ReduceMethod reduce_method;
    // 多输出消息时的堆叠维度
    int stack_dim;
    // 多输出消息时的堆叠顺序
    std::vector<NodeId> stack_order;
};

struct OutputDescription {
    // NodeId node_id;
    NodeName node_name;
    MessageName message_name;
    MessageShape shape;
};

struct NodeDescription {
    NodeId node_id;
    std::vector<InputDescription> inputs;
    std::vector<OutputDescription> outputs;
};

class HistorySourceComponent : public ComponentBase {
public:
    HistorySourceComponent(const std::string message_name, MessageShape shape, int history_offset )
        : ComponentBase(message_name + std::to_string(history_offset)) {
        message_name_ = message_name;
        shape_ = shape;
        history_offset_ = history_offset;
    }

    std::string getPublishMessageName() const {
        return "";
    }

private:
    std::string message_name_;
    MessageShape shape_;
    int history_offset_;
};

class ReducerComponent : public ComponentBase {

public:
    ReducerComponent(const MessageName &message_name, ReduceMethod reduce_method, MessageShape shape)
        : ComponentBase(generateNodeName(message_name, reduce_method)), message_shape_(shape) {

    }

    static MessageName generateNodeName(const MessageName &message_name, ReduceMethod reduce_method) {
        return message_name + "." + std::to_string(static_cast<int>(reduce_method));
    }

    // void onInit
private:
    MessageShape message_shape_;
};

class MessageBus
{
public:
    using GraphId = std::string;
    using MessageRoute = std::pair<std::set<int>, std::set<int>>;


    void registerComponent(ComponentBase* component) {
        const auto &node_name = component->getName();
        const auto &graph_name = component->getExecGraph();
        if(node_id_map_.find(node_name) != node_id_map_.end()) {
            std::cerr << "node has been registered!" << std::endl;
            return;
        }
        NodeId node_id = next_node_id_++;
        node_id_map_[node_name] = node_id;
        nodes_.push_back(component);

        // 所有执行图的集合
        graphs_.insert(graph_name);
    }

    void registerInput(
        ComponentBase* component,
        const std::string name,
        const std::vector<int64_t> shape,
        int history_offset = 0,
        ITensor* history_padding_val = nullptr,
        ReduceMethod reduce_method = ReduceMethod::STACK
        ) {
            // TODO. 历史消息以$N_MessageName重命名
        }


    void registerOutput(
        ComponentBase* component,
        const std::string name,
        const std::vector<int64_t> shape
    ) {}


    void buildGraph() {
        std::lock_guard<std::mutex> lock(mutex_);

        // 1. 处理多发布者对一个消息的情况
        adjustMessageRoutes();

        // 2. 建立消息发送图
        buildMessageGraph();

        // 3. 禁用无源组件
        buildActiveGraph();

        // 4. 构建活动图
        buildActiveGraph();

        // 5. 检查消息兼容性
        checkActiveGraphMessageCompatible();

        // 6. 检查循环依赖
        checkActiveGraphCycles();

        // 7. 更新组件路由引用
        updateComponentsRouteRef();

        // 8. 构建执行图
        buildExecutionGraph();

        // 9. 生成执行顺序
        generateExecutionOrder();
    }

private:
    // using VertexProperties = boost::property<boost::vertex_index_t, NodeId>;
    using EdgeProperties = boost::property<boost::edge_index_t, MessageId>;

    struct VertexProperties {
        NodeId node_id;
        bool enabled;
    };

    using Graph = boost::adjacency_list<boost::vecS, boost::vecS, boost::directedS,
        VertexProperties, EdgeProperties>;

    NodeId findOrCreateReducerNode(const MessageName& message_name, const ReduceMethod& reduce_method, const MessageShape &shape) {
        // reducer 组件的命名规则为 messageName.reduceMethod
        // 新的消息id为 messageName.reduceMethod
        NodeName reducer_name = ReducerComponent::generateNodeName(message_name, reduce_method);
        NodeId reducer_id = -1;
        if (node_id_map_.find(reducer_name) == node_id_map_.end()) {
            auto new_component = std::make_unique<ReducerComponent>(message_name, reduce_method, shape);
            reducers_.insert(new_component);
            registerComponent(new_component.get());
        }
        return node_id_map_[reducer_name];
    }

    void adjustMessageRoutes() {
        std::unordered_map<MessageId, MessageRoute> newRoutes = message_routes_;

        // // 遍历所有节点的输入，如果需要历史数据，则创建history_source组件，更新路由，将原节点的输入替换为history_source组件的输出
        // for (auto& [message_id, pub_subs] : message_routes_) {
        //     const auto & [pub_ids, sub_ids] = pub_subs;

        //     for (auto& sub_id : sub_ids) {
        //         InputDescription& sub = input_descriptions_[sub_id];

        //         if (sub.history_offset > 0) {
        //             auto history_src_node = new HistorySourceComponent(
        //                 sub.message_name,
        //                 sub.shape,
        //                 sub.history_offset);

        //             // 调用history_source组件注册回调，在回调中注册发送消息OutputDescription
        //             history_src_node->onInit();

        //             // 更改原有消息的订阅
        //             MessageName history_msg_name = history_src_node->getPublishMessageName();
        //             MessageId history_msg_id = message_id_map_[history_msg_name];
        //             newRoutes[message_id].second.erase(sub_id);
        //             newRoutes[history_msg_id].second.insert(sub_id);

        //             sub.message_id = history_msg_id;
        //             sub.message_name = history_msg_name;
        //         }
        //     }
        // }

        // 遍历所有消息, 处理多发布者对一个消息的情况
        for (auto& [messageId, routes] : message_routes_) {
            auto& [pub_ids, sub_ids] = routes;

            std::unordered_set<NodeId> reducerNodes;

            for (auto& sub_id : sub_ids) {
                auto& sub = input_descriptions_[sub_id];
                if (pub_ids.size() <= 1) continue;

                if (sub.reduce_method == core::ReduceMethod::STACK) {
                    sub.stack_dim = 1;
                    sub.stack_order.clear();
                    for (const auto& pub_id : pub_ids) {
                        sub.stack_order.push_back(output_descriptions_[pub_id].node_id);
                    }
                } else {
                    // 创建reducer组件
                    NodeId reducer_id = findOrCreateReducerNode(sub.message_name, sub.reduce_method, sub.shape);

                    // 更改subscriber的接收消息id
                    sub.message_id = newMessageId;
                    newRoutes[newMessageId].second.push_back(sub);
                    auto it = std::find(newRoutes[messageId].second.begin(), newRoutes[messageId].second.end(), sub);
                    if (it != newRoutes[messageId].second.end()) {
                        newRoutes[messageId].second.erase(it);
                    }
                }
            }
        }
        active_message_routes_ = std::move(newRoutes);
    }

    void buildMessageGraph() {
        message_graph_.clear();

        // 添加所有组件节点
        for (const auto& node  : nodes_) {
            NodeId node_id = node_id_map_[node->getName()];
            Vertex v = boost::add_vertex({node_id, false}, message_graph_);
            vertex_map[node->getName()] = v;
        }

        // 添加辅助节点，用于表示无发布者的消息
        Vertex no_pub_vertex = boost::add_vertex(message_graph_);
        vertex_map["no_pub"] = no_pub_vertex;

        // 建立消息依赖关系
        for (const auto& [messageId, routes] : message_routes_) {   // 遍历所有消息
            const auto& [publishers, subscribers] = routes;
            for (const auto& sub_id : subscribers) {   // 遍历所有订阅者
                const auto& sub = input_descriptions_[sub_id];
                if (publishers.empty()) {   // 没有发布者, 添加边到no_pub_vertex
                    Vertex sub_vertex = vertex_map[sub.node_name];
                    boost::add_edge(no_pub_vertex, sub_vertex, messageId, message_graph_);
                } else {
                    for (const auto& pub_id : publishers) {
                        const auto& pub = output_descriptions_[pub_id];
                        Vertex pub_vertex = vertex_map[pub.node_name];
                        Vertex sub_vertex = vertex_map[sub.node_name];
                        boost::add_edge(pub_vertex, sub_vertex, messageId, message_graph_);
                    }
                }
            }
        }
    }


    void buildActiveGraph() {
        std::vector<Vertex> inactive_nodes;

        // 删除no_pub的后续节点
        boost::breadth_first_search(
            message_graph_,
            vertex_map["no_pub"],
            boost::visitor(boost::make_bfs_visitor(
                boost::record_predecessors(
                    inactive_nodes,
                    boost::on_tree_edge()))));

        // 删除出度为0的节点
        for (auto it = boost::vertices(active_graph_).first; it != boost::vertices(active_graph_).second; ++it) {
            if (boost::out_degree(*it, active_graph_) == 0) {
                inactive_nodes.push_back(*it);
            }
        }

        // 删除no_pub节点
        inactive_nodes.push_back(vertex_map["no_pub"]);

        active_graph_ = message_graph_;
        for (const Vertex& v : inactive_nodes) {
            boost::remove_vertex(v, active_graph_);
        }
    }

    void checkActiveGraphMessageCompatible() {
        for (const auto& [messageId, routes] : message_routes_) {
            const auto& [pub_ids, sub_ids] = routes;
            if (pub_ids.empty() || sub_ids.empty()) continue;

            const auto& pubShape = output_descriptions_[*pub_ids.begin()].shape;

            // 检查所有发布者的形状
            for (size_t i = 1; i < pub_ids.size(); ++i) {

                if (publishers[i].shape != pubShape) {
                    throw std::runtime_error(
                        "Inconsistent shapes for publishers of message " + std::to_string(messageId));
                }
            }

            // 检查所有订阅者的形状
            for (const auto& sub : sub_ids) {
                if (sub.shape != pubShape) {
                    throw std::runtime_error(
                        "Shape mismatch for subscriber of message " + std::to_string(messageId));
                }
            }
        }
    }

    void check_active_graph_cycles() {
        try {
            std::vector<NodeId> order;
            boost::topological_sort(active_graph_, std::back_inserter(order));
        } catch (const boost::not_a_dag&) {
            throw std::runtime_error("Circular dependency detected in message graph");
        }
    }

private:
    std::mutex mutex_;
    std::unordered_set<GraphId> graphs_;

    NodeId next_node_id_ = 0;
    std::vector<ComponentBase*> nodes_;
    std::set<std::unique_ptr<ReducerComponent>> reducers_;
    std::unordered_map<NodeName, NodeId> node_id_map_;

    // std::vector<ComponentBase*> components_;
    std::vector<MessageName> messages_;
    std::unordered_map<MessageName, MessageId> message_id_map_;

    std::unordered_map<MessageId, MessageQueue> message_queues_;
    std::unordered_map<MessageId, MessageRoute> message_routes_;

    std::vector<InputDescription> input_descriptions_;
    std::vector<OutputDescription> output_descriptions_;

    std::unordered_map<MessageId, MessageRoute> active_message_routes_;
    // std::unordered_map<MessageId, std::vector<NodeId>> message_graph_;

    Graph message_graph_, active_graph_;
    using Vertex = boost::graph_traits<Graph>::vertex_descriptor;
    std::unordered_map<NodeName, Vertex> vertex_map;

};

} // namespace core
} // namespace cuda_simulator


#endif // CUDASIM_MESSAGEBUS_HH