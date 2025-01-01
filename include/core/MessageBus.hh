#ifndef CUDASIM_MESSAGEBUS_HH
#define CUDASIM_MESSAGEBUS_HH


#include <cstdint>
#include <memory>
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
#include <utility>
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

using NodeName = std::string;
using NodeId = std::uint32_t;

using MessageName = std::string;
using MessageId = std::uint32_t;
using MessageShape = std::vector<int64_t>;
using MessageQueueId = std::uint32_t;
using DescriptionId = std::int32_t;

class MessageQueue {
public:
    explicit MessageQueue(NodeId pub_node_id, MessageId message_id, MessageName message_name, MessageShape shape, size_t max_history_len)
        : pub_node_id_(pub_node_id)
        ,message_id_(message_id)
        , message_name_(message_name)
        , shape_(shape)
        , max_history_len_(max_history_len)
        , valid_count_(0) { }

    // ITensor* getHistory(int offset) {
    //     if (auto it = cache_.find(offset); it != cache_.end()) {
    //         return it->second;
    //     }

    //     auto result = history_[history_.size() - 1 - offset];
    //     cache_[offset] = result;
    //     return result;
    // }

    // void append(ITensor*& data) {
    //     history_.push_back(data);
    //     if (history_.size() > maxHistoryLen_) {
    //         history_.pop_front();
    //     }
    //     validCount_ = std::min(maxHistoryLen_, validCount_ + 1);
    //     latestUpdateTime_++;
    // }

    // void reset() {
    //     history_.clear();
    //     validCount_ = 0;
    //     latestUpdateTime_ = 0;
    // }

    size_t getValidCount() const { return valid_count_; }

private:
    NodeId      pub_node_id_;
    MessageId   message_id_;
    MessageName message_name_;
    MessageShape shape_;
    size_t max_history_len_;
    size_t valid_count_;
    std::deque<ITensor*> history_;
    // size_t latestUpdateTime_;
    // std::unordered_map<int, ITensor*> cache_;
};



struct InputDescription {
    // 输入组件id
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
    NodeName node_name;
    MessageName message_name;
    MessageShape shape;
    MessageQueueId queue_id = -1;
};


struct NodeDescription {
    NodeId node_id;
    NodeName node_name;
    std::vector<DescriptionId> active_inputs;
    std::vector<DescriptionId> active_outputs;
    std::unordered_map<MessageName, std::vector<MessageQueueId>> active_input_map;
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

    MessageName getOutputMessageName() {
        return getName();
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

    void registerComponent(ComponentBase* component) {
        const auto &node_name = component->getName();
        const auto &graph_name = component->getExecGraph();
        if(node_id_map_.find(node_name) != node_id_map_.end()) {
            std::cerr << "node has been registered!" << std::endl;
            return;
        }
        NodeId node_id = nodes_.size();
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
        generateExecuationOrder();

        // 8. 构建执行图
        buildExecutionGraph();

        // 9. 生成执行顺序
        generateExecutionOrder();

        // 7. 更新组件路由引用
        updateNodeDescription();
    }

private:
    using VertexProperties = NodeDescription;
    // using EdgeProperties = boost::property<boost::edge_index_t, MessageId>;

    struct EdgeProperties {
        MessageId message_id;
        DescriptionId pub_des_id;
        DescriptionId sub_des_id;
    };

    using Graph = boost::adjacency_list<boost::vecS, boost::vecS, boost::directedS,
        VertexProperties, EdgeProperties>;

    ReducerComponent* findOrCreateReducerNode(const MessageName& message_name, const ReduceMethod& reduce_method, const MessageShape &shape) {
        // reducer 组件的命名规则为 messageName.reduceMethod
        // 新的消息id为 messageName.reduceMethod
        NodeName reducer_name = ReducerComponent::generateNodeName(message_name, reduce_method);
        ReducerComponent *p_com;

        auto finder = node_id_map_.find(reducer_name);
        if (finder == node_id_map_.end()) {
            auto new_component = std::make_unique<ReducerComponent>(message_name, reduce_method, shape);
            reducers_.insert(new_component);
            p_com = new_component.get();
            registerComponent(p_com);
        } else {
            p_com = static_cast<ReducerComponent*>(nodes_[finder->second]);
        }

        return p_com;
    }

    void adjustMessageRoutes() {
        std::unordered_map<MessageId, MessageRoute> newRoutes = message_routes_;

        // 遍历所有消息, 处理多发布者发布同一个一个消息的情况
        for (auto& [messageId, routes] : message_routes_) {
            auto& [pub_ids, sub_ids] = routes;
            if (pub_ids.size() <= 1) continue;

            for (auto sub_id : sub_ids) {
                auto& sub = input_descriptions_[sub_id];

                // Reduce方法如果是STACK，则按照stack_order进行堆叠。当stack_order不存在时根据pub_ids顺序重建
                if (sub.reduce_method == core::ReduceMethod::STACK) {
                    if (sub.stack_order.size() == 0) {
                        for (auto pub_id : pub_ids) {
                            int pub_node_id = node_id_map_[ output_descriptions_[pub_id].node_name ];
                            sub.stack_order.push_back(pub_node_id);
                        }
                    }
                } else {    // 如果不是STACK，则需要创建reducer组件
                    auto reducer = findOrCreateReducerNode(sub.message_name, sub.reduce_method, sub.shape);

                    // 更改newRoute中的subscriber的接收消息id，为Reducer发布的消息ID
                    MessageName msg_name = reducer->getOutputMessageName();
                    sub.message_name = msg_name;
                    newRoutes[message_id_map_[msg_name]].second.insert(sub_id);
                    // 删除原有message到subscriber的消息路由
                    auto it = std::find(newRoutes[messageId].second.begin(), newRoutes[messageId].second.end(), sub);
                    if (it != newRoutes[messageId].second.end()) {
                        newRoutes[messageId].second.erase(it);
                    }
                }
            }
        }
        adjusted_message_routes_ = std::move(newRoutes);
    }

    void buildMessageGraph() {
        message_graph_.clear();

        // 添加所有组件节点
        for (const auto& node  : nodes_) {
            NodeId node_id = node_id_map_[node->getName()];
            Vertex v = boost::add_vertex({node_id}, message_graph_);
            vertex_map[node->getName()] = v;
        }

        // 添加辅助节点，用于表示无发布者的消息
        Vertex no_pub_vertex = boost::add_vertex({NOPUB_NODE_ID, NOPUB_NODE_NAME}, message_graph_);
        vertex_map["no_pub"] = no_pub_vertex;

        // 建立消息依赖关系
        for (const auto& [message_id, routes] : message_routes_) {   // 遍历所有消息
            const auto& [pub_ids, sub_ids] = routes;

            // 以$开头的消息为历史消息，没有依赖关系
            if(messages_[message_id][0] == '$') continue;

            for (const auto& sub_id : sub_ids) {   // 遍历所有订阅者
                const auto& sub = input_descriptions_[sub_id];
                if (pub_ids.empty()) {   // 没有发布者, 添加边到no_pub_vertex
                    Vertex sub_vertex = vertex_map[sub.node_name];
                    boost::add_edge(no_pub_vertex, sub_vertex, {message_id, NOPUB_NODE_ID, sub_id}, message_graph_);
                } else {
                    for (const auto& pub_id : pub_ids) {
                        const auto& pub = output_descriptions_[pub_id];
                        Vertex pub_vertex = vertex_map[pub.node_name];
                        Vertex sub_vertex = vertex_map[sub.node_name];
                        boost::add_edge(pub_vertex, sub_vertex, {message_id, pub_id, sub_id}, message_graph_);
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
        // TODO. 创建消息队列后，跟消息队列的形状进行检查
        for (const auto& [messageId, routes] : message_routes_) {
            const auto& [pub_ids, sub_ids] = routes;
            if (pub_ids.empty() || sub_ids.empty()) continue;

            const auto& pubShape = output_descriptions_[*pub_ids.begin()].shape;

            // 检查所有发布者的形状
            for (auto pub_id : pub_ids) {
                if (output_descriptions_[pub_id].shape != pubShape) {
                    throw std::runtime_error(
                        "Inconsistent shapes for publishers of message " + std::to_string(messageId));
                }
            }

            // 检查所有订阅者的形状
            for (auto sub_id : sub_ids) {
                if (input_descriptions_[sub_id].shape != pubShape) {

                }
            }
        }
    }

    void generateExecuationOrder() {
        // try {
        //     // std::vector<NodeId> order;
        //     // boost::topological_sort(active_graph_, std::back_inserter(order));

        //     execution_order_.clear();

        //     std::vector<Vertex> order;
        //     boost::topological_sort(active_graph_, std::back_inserter(order));

        //     // 将排序结果分组
        //     std::vector<std::vector<NodeId>> groups;
        //     for (const auto& Vertex : order) {
        //         std::vector<NodeId> group;
        //         for (const auto& [componentId, component] : components_) {
        //             if (component->getGraphId() == graphId && component->isEnabled()) {
        //                 group.push_back(componentId);
        //             }
        //         }
        //         if (!group.empty()) {
        //             execution_order_.push_back(std::move(group));
        //         }
        //     }
        // } catch (const boost::not_a_dag&) {
        //     throw std::runtime_error("Circular dependency detected in message graph");
        // }
    }

    void createMessageQueues() {

        // 生成当前的活跃消息路由表
        for(auto [ei, eind] = boost::edges(active_graph_); ei != eind; ei++) {
            auto &edge_properties = active_graph_[*ei];
            MessageId message_id = edge_properties.message_id;

            if(active_messages_routes_.find(message_id) == active_messages_routes_.end()) {
                active_messages_routes_[message_id] = {{},{}};
            }

            auto &[pub_ids, sub_ids] = active_messages_routes_[message_id];
            pub_ids.insert(edge_properties.pub_des_id);
            sub_ids.insert(edge_properties.sub_des_id);
        }

        // 遍历每一个节点的出边，建立消息队列
        for(auto [vi, vend] = boost::vertices(active_graph_); vi!=vend; vi++) {
            NodeId node_id = active_graph_[*vi].node_id;
            VertexOutEdgeIterator ei, eind;
            for(std::tie(ei, eind) = boost::out_edges(*vi, active_graph_); ei!=eind; ei++) {
                const EdgeProperties &edge_props = active_graph_[*ei];
                MessageId message_id = edge_props.message_id;
                const MessageName& message_name = messages_[message_id];
                OutputDescription &pub = output_descriptions_[ edge_props.pub_des_id ];

                int mq_history_len = 0;

                // 遍历当前消息的所有接收者，得到最大历史长度
                for(auto sub_id : active_messages_routes_[message_id].second) {
                    const auto &sub = input_descriptions_[sub_id];

                    if(sub.shape != pub.shape) {
                        throw std::runtime_error("Shape mismatch for subscriber of message " + message_name);
                    }

                    if (sub.history_offset > mq_history_len)
                        mq_history_len = sub.history_offset;
                }

                // 创建MQ_Id, 修改OutputDescription，注明MessageQueueId
                pub.queue_id = message_queues_.size();
                message_queues_.push_back(std::make_unique<MessageQueue>(
                    node_id, message_id, message_name, pub.shape, mq_history_len));
            }
        }
    }

    void updateNodeDescription() {
        for(auto [vi, vend] = boost::vertices(active_graph_); vi!=vend; vi++) {
            VertexProperties &node_des = active_graph_[*vi];

            // 填充active_input, active_output
            VertexOutEdgeIterator ei, eind;
            for(std::tie(ei, eind) = boost::out_edges(*vi, active_graph_); ei!=eind; ei++) {
                const EdgeProperties &edge_props = active_graph_[*ei];
                node_des.active_inputs.push_back(edge_props.sub_des_id);
                node_des.active_outputs.push_back(edge_props.pub_des_id);
            }

            // 对于所有的active_input, 查找对应消息的pub, 得到message_queue_id, 填充input_map
            for(auto sub_id : node_des.active_inputs) {
                const MessageName &message_name = input_descriptions_[sub_id].message_name;
                std::vector<MessageQueueId> mq_ids {};
                MessageId message_id = message_id_map_[message_name];
                for(auto pub_id : active_messages_routes_[message_id].first) {
                    const auto &pub = output_descriptions_[pub_id];
                    mq_ids.push_back(pub.queue_id);
                }
                node_des.active_input_map[message_name] = mq_ids;
            }

        }
    }


private:
    static constexpr const char* NOPUB_NODE_NAME = "no_pub";
    static constexpr NodeId NOPUB_NODE_ID = 65536;

    using GraphId = std::string;
    using MessageRoute = std::pair<std::unordered_set<DescriptionId>, std::unordered_set<DescriptionId>>;
    using Vertex = boost::graph_traits<Graph>::vertex_descriptor;
    using VertexIterator = boost::graph_traits<Graph>::vertex_iterator;
    using EdgeIterator = boost::graph_traits<Graph>::edge_iterator;
    using VertexOutEdgeIterator = boost::graph_traits<Graph>::out_edge_iterator;

    std::mutex mutex_;
    std::unordered_set<GraphId> graphs_;

    // 节点
    std::vector<ComponentBase*> nodes_;
    std::unordered_map<NodeName, NodeId> node_id_map_;

    // Reducer
    std::unordered_set<std::unique_ptr<ReducerComponent>> reducers_;

    // 消息
    std::vector<MessageName> messages_;
    std::unordered_map<MessageName, MessageId> message_id_map_;
    std::vector<InputDescription> input_descriptions_;
    std::vector<OutputDescription> output_descriptions_;

    // 消息转发
    std::unordered_map<MessageId, MessageRoute> message_routes_;            // 原路由
    std::unordered_map<MessageId, MessageRoute> adjusted_message_routes_;   // 增加了reducer的路由

    // 节点消息图
    Graph message_graph_;
    Graph active_graph_;
    std::unordered_map<NodeName, Vertex> vertex_map;

    // 消息队列
    std::unordered_map<MessageId, MessageRoute> active_messages_routes_;
    std::vector<std::unique_ptr<MessageQueue>> message_queues_;

    // 执行顺序
    std::vector<std::vector<NodeId>> execution_order_;

};

} // namespace core
} // namespace cuda_simulator


#endif // CUDASIM_MESSAGEBUS_HH