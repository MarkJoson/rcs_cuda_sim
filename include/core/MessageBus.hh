#ifndef CUDASIM_MESSAGEBUS_HH
#define CUDASIM_MESSAGEBUS_HH

#include <boost/config.hpp>

#include <cassert>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <iostream>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include <fstream>
#include <sstream>

#include <boost/utility.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/topological_sort.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/property_map/property_map.hpp>
#include <boost/graph/properties.hpp>
#include <boost/graph/breadth_first_search.hpp>
#include <boost/graph/depth_first_search.hpp>
#include <boost/graph/visitors.hpp>

#include <boost/format.hpp>

#include "console_style.h"
#include "core_types.hh"
#include "ExecuteNode.hh"
#include "MessageQueue.hh"
#include "ReducerNode.hh"

namespace cuda_simulator
{
namespace core
{

class SimulatorContext;

using MessageQueueId = std::uint32_t;
using DescriptionId = std::int32_t;

// &---------------------------------------------- MessageBus -------------------------------------------------------
class MessageBus
{
public:
    MessageBus()  {};

    void registerComponent(ExecuteNode* node) {    // 改为接收ExecuteNode
        // std::lock_guard<std::mutex> lock(mutex_);
        // TODO. 增加锁

        createNodeId(node->getName(), node->getTag(), node);

        // TODO. 调用注册回调函数，初始化输入输出
        // node->onRegister(context_);
    }

    const MessageShape& getMessageShape(const MessageNameRef& message_name) {
        int message_id = message_id_map_[message_name];
        return messages_.at(message_id).second;
    }

    void registerInput(ExecuteNode* component, const ExecuteNode::NodeInputInfo &info) {
        // std::lock_guard<std::mutex> lock(mutex_);
        // TODO.

        if (info.history_offset < 0) {
            throw std::runtime_error("Invalid history offset");
        }

        MessageId message_id = lookUpOrCreateMessageId(info.message_name, info.shape);

        NodeId node_id = node_id_map_[component->getName()];

        NodeDescription &node_desc = node_descriptions_[node_id];
        if (node_desc.input_map.find(message_id) != node_desc.input_map.end()) {
            throw std::runtime_error("Input message already registered");
        }

        DescriptionId input_des_id = input_descriptions_.size();
        input_descriptions_.push_back({
            node_id,
            component->getName(),
            message_id,
            info.message_name,
            info.shape,
            info.history_offset,
            info.reduce_method,
            {}, {}
            }
        );

        // 更新节点接收消息列表
        node_descriptions_[node_id].input_map[message_id] = input_des_id;

        // 更新路由表
        message_routes_[message_id].second.insert(input_des_id);
    }

    void registerOutput( ExecuteNode* component, ExecuteNode::NodeOutputInfo &info ) {
        // std::lock_guard<std::mutex> lock(mutex_);
        // TODO.

        MessageId message_id = lookUpOrCreateMessageId(info.message_name, info.shape);

        NodeId node_id = node_id_map_[component->getName()];

        NodeDescription &node_desc = node_descriptions_[node_id];
        if (node_desc.output_map.find(message_id) != node_desc.output_map.end()) {
            throw std::runtime_error("Input message already registered");
        }

        DescriptionId output_des_id = output_descriptions_.size();
        output_descriptions_.push_back({
            node_id,
            component->getName(),
            message_id,
            info.message_name,
            info.shape,
            info.history_padding_val,
            INT_MAX
        });

        // 更新节点接收消息列表
        node_descriptions_[node_id].output_map[message_id] = output_des_id;

        message_routes_[message_id].first.insert(output_des_id);
    }

    MessageQueue* getMessageQueue(const NodeNameRef &node_name, const MessageNameRef &message_name) {
        NodeId node_id = node_id_map_[node_name];
        try {
            MessageQueueId mq_id = node_descriptions_[node_id].active_output_map[message_name];
            return message_queues_[mq_id].get();
        } catch (const std::out_of_range &e) {
            throw std::runtime_error("MessageQueue not found! Maybe the message is not active.");
        }
    }

    void addTrigger(NodeTagRef trigger_tag) {
        // std::lock_guard<std::mutex> lock(mutex_);
        // TODO.

        if (triggers_.find(trigger_tag) != triggers_.end()) {
            throw std::runtime_error("Trigger already exists");
        }

        MessageId trigger_msg_id = lookUpOrCreateMessageId(trigger_tag, {});

        triggers_[trigger_tag] = {
            0,
            trigger_tag,
            trigger_msg_id,
            {}
        };
    }

    void buildGraph() {
        // std::lock_guard<std::mutex> lock(mutex_);
        // TODO.

        // 1. 处理多发布者对一个消息的情况
        adjustMessageRoutes();

        // 2. 建立消息发送图
        buildMessageGraph();
        exportGraphVisualization(message_graph_, "MessageGraph", "message_graph.dot");

        // 3. 检查循环依赖
        checkActiveGraphLoop();

        // 4. 创建消息队列
        createMessageQueues();

        // 5. 更新Node的Input-MessageQueue映射
        updateNodeDescription();

        // 6. 构建活动图
        pruneGraphToOneStepIter();
        exportGraphVisualization(active_graph_, "ActiveGraph", "active_graph.dot");

        // 7. 添加触发器到图中
        addTriggerToGraph();
        exportGraphVisualization(trigger_graph_, "TriggerGraph", "trigger_graph.dot", true);
    }

    void clearAll() {
        // std::lock_guard<std::mutex> lock(mutex_);
        // TODO.

        node_descriptions_.clear();
        node_id_map_.clear();
        reducers_.clear();

        messages_.clear();
        message_id_map_.clear();
        input_descriptions_.clear();
        output_descriptions_.clear();

        message_routes_.clear();
        adjusted_message_routes_.clear();

        message_graph_.clear();
        active_graph_.clear();
        vertex_map.clear();

        active_messages_routes_.clear();
        message_queues_.clear();
    }

    void trigger(NodeTagRef trigger_tag) {
        /// 触发传递，执行某个Tag下的所有节点
        // std::lock_guard<std::mutex> lock(mutex_);
        // TODO.

        auto &trigger_desc = triggers_[trigger_tag];

        for (const auto &[node_order, node_ids] : trigger_desc.node_order) {
            if (node_order >= current_execute_order_ + 1) {
                std::cout << "Trigger [" << trigger_tag << "] stopped at order " << current_execute_order_ << std::endl;
                break;
            }

            // 找到本次需要执行的节点然后停下
            if (node_order < current_execute_order_)
                continue;

            // TODO. 下述节点可以并行执行
            for (NodeId node_id : node_ids) {
                executeNode(node_id);
            }
            current_execute_order_ ++;
        }
    }

    void resetExecuteOrder() {
        current_execute_order_ = 0;
    }

    // 当前MessageBus的执行位次
    int getCurrentExecuteOrder() {
        return current_execute_order_;
    }

    int getNodeOrder(NodeId node_id) {
        return node_order_[node_id];
    }

private:    // ^---- 私有定义 -----

    enum class NodeType {
        COMPONENT_NODE = 0,
        TRIGGER_NODE = 1,
        AUX_NOPUB_NODE = 2,
    };

    // **---- 图 -----
    struct VertexProperties {
        NodeId node_id;
        NodeNameRef node_name;
        NodeTagRef node_tag;
        NodeType node_type = NodeType::COMPONENT_NODE;
    };
    // using EdgeProperties = boost::property<boost::edge_index_t, MessageId>;

    struct EdgeProperties {
        MessageId message_id;
        DescriptionId pub_des_id;
        DescriptionId sub_des_id;
    };


    using Graph = boost::adjacency_list<boost::vecS, boost::vecS, boost::bidirectionalS, VertexProperties, EdgeProperties>;
    using GraphId = std::string;
    using MessageRoute = std::pair<std::unordered_set<DescriptionId>, std::unordered_set<DescriptionId>>;
    using Vertex = boost::graph_traits<Graph>::vertex_descriptor;
    using VertexIterator = boost::graph_traits<Graph>::vertex_iterator;
    using EdgeIterator = boost::graph_traits<Graph>::edge_iterator;

    using VertexInEdgeIterator = boost::graph_traits<Graph>::in_edge_iterator;
    using VertexOutEdgeIterator = boost::graph_traits<Graph>::out_edge_iterator;

    // **---- Descriptions -----
    struct InputDescription {
        // 所属组件名
        NodeId node_id;
        NodeNameRef node_name;
        // 接收消息名
        MessageId message_id;
        MessageNameRef message_name;
        // 消息形状
        MessageShape shape;
        // 历史偏移
        int history_offset;
        // 多输出消息时的归约方法，当reduce_method为STACK时，stack_dim和stack_order有效
        ReduceMethod reduce_method;
        // 多输出消息时的堆叠顺序
        std::vector<NodeNameRef> stack_order;
        std::vector<NodeId> stack_order_pub_id;
    };

    struct OutputDescription {
        // 所属组件名
        NodeId node_id;
        NodeNameRef node_name;
        // 发布消息名
        MessageId message_id;
        MessageNameRef message_name;
        // 消息形状
        MessageShape shape;
        // 无效历史数据的填充值
        std::optional<TensorHandle> history_padding_val;
        // 消息队列ID
        MessageQueueId queue_id = INT_MAX;
    };

    struct NodeDescription {
        NodeId node_id;
        NodeNameRef node_name;
        NodeTagRef node_tag;
        ExecuteNode* node;      // 改为ExecuteNode*

        std::unordered_map<MessageId, DescriptionId> input_map;
        std::unordered_map<MessageId, DescriptionId> output_map;

        std::unordered_map<MessageId, DescriptionId> active_input_desc;
        std::unordered_map<MessageId, DescriptionId> active_outputs_desc;

        std::unordered_map<MessageNameRef, std::vector<
            std::pair<MessageQueueId, int>>> active_input_map;      // MessageQueue Id, history_offset
        std::unordered_map<MessageNameRef, MessageQueueId> active_output_map;
    };


    // Trigger 在依赖图中连接所有Tag == trigger_tag的节点, 用于触发消息传递
    struct TriggerDescription {
        // 触发节点标识符
        boost::graph_traits<Graph>::vertex_descriptor trigger_vertex;
        // 触发标签
        NodeTagRef trigger_tag;
        // 触发消息id
        MessageId trigger_message_id;
        // 拓扑排序结果，用于并行执行
        std::vector<std::pair<int, std::unordered_set<NodeId>>> node_order;
    };


    struct CycleDetector : public boost::dfs_visitor<> {
        CycleDetector(bool& has_cycle) : has_cycle_(has_cycle) {}
        void back_edge(typename boost::graph_traits<Graph>::edge_descriptor e, const Graph& g) const {
            has_cycle_ = true;
            std::cerr << "Cycle detected! Message " << g[e].message_id
                      << " from " << g[e].pub_des_id << " to" << g[e].sub_des_id << std::endl;
        }
    protected:
        bool& has_cycle_;
    };

    class VertexVisitor : public boost::default_bfs_visitor {
    public:
        VertexVisitor(std::vector<Vertex>& inactive_nodes) : inactive_nodes_(inactive_nodes) {}
        template <typename Vertex, typename Graph>
        void discover_vertex(Vertex u, const Graph&) const { inactive_nodes_.push_back(u); }
    private:
        std::vector<Vertex>& inactive_nodes_;
    };

    NodeId createNodeId(const NodeNameRef& node_name, const NodeTagRef& node_tag, ExecuteNode* node) {
        if(node_id_map_.find(node_name) != node_id_map_.end()) {
            throw std::runtime_error("Node has been registered!");
        }
        NodeId node_id = node_descriptions_.size();
        node_id_map_[node_name] = node_id;

        node_descriptions_.push_back({
            node_id,
            node_name,
            node_tag,
            node,
            {},{},
            {}, {},
            {}, {}
            }
        );

        return node_id;
    }


    MessageId lookUpOrCreateMessageId(const MessageNameRef& message_name, const MessageShape& shape) {
        MessageId message_id = 0;
        if (message_id_map_.find(message_name) == message_id_map_.end()) {
            message_id = messages_.size();
            messages_.push_back({message_name, shape});
            message_id_map_[message_name] = message_id;
            message_routes_[message_id] = { {}, {} };
        } else {
            message_id = message_id_map_[message_name];
        }
        return message_id;
    }

    MessageName generateReducerNodeName(const MessageNameRef &message_name, ReduceMethod reduce_method) {
        return std::string(message_name) + "." + reduce_method_to_string(reduce_method);
    }

    ReducerNode* lookUpOrCreateReducerNode(
        const NodeTagRef& sub_node_tag, const MessageNameRef& message_name,
        const ReduceMethod& reduce_method, int history_offset, const MessageShape &shape
    ) {
        // reducer 组件的命名规则为 messageName.reduceMethod
        // 新的消息id为 messageName.reduceMethod
        NodeName reducer_name = generateReducerNodeName(message_name, reduce_method);
        ReducerNode *p_com = nullptr;

        auto finder = node_id_map_.find(reducer_name);
        if (finder == node_id_map_.end()) {
            auto new_component = std::make_unique<ReducerNode>(
                reducer_name,
                std::string(sub_node_tag),
                message_name,
                reducer_name,               // 使用reducer_name作为消息发送名称
                reduce_method,
                history_offset,
                shape);
            p_com = new_component.get();
            registerComponent(p_com);
            MessageNameRef reducer_output_message_name = p_com->getOutputMessageName();
            reducers_.insert(std::make_pair(reducer_output_message_name, std::move(new_component)));
        } else {
            p_com = static_cast<ReducerNode*>(node_descriptions_[ finder->second ].node);
        }

        return p_com;
    }

    void adjustMessageRoutes() {
        /// 处理多发布者发布同一个消息的情况，为ReduceMethod!=STACK的订阅者创建Reducer节点

        // 由于需要修改message_routes_，因此使用原有的路由表进行遍历
        auto origin_routes = message_routes_;

        // 处理所有的ReuceMethod!=STACK的订阅者，向路由中增加Reducer节点
        for (auto& [message_id, routes] : origin_routes) {
            auto& [pub_ids, sub_ids] = routes;

            for (auto sub_id : sub_ids) {
                auto *sub = &input_descriptions_[sub_id];

                // 仅有一个publisher时，统一将reduce_method设置为stack
                if (pub_ids.size() == 1) {
                    sub->stack_order.clear();
                    sub->reduce_method = ReduceMethod::STACK;
                }

                // 如果不是STACK，则需要创建reducer组件
                if (sub->reduce_method != ReduceMethod::STACK) {
                    const NodeTagRef &node_tag = node_descriptions_.at(sub->node_id).node_tag;
                    auto reducer = lookUpOrCreateReducerNode(
                        node_tag,
                        sub->message_name,
                        sub->reduce_method,
                        sub->history_offset,
                        sub->shape);

                    // ! 需要重新索引sub，否则会因为向input_descriptions_插入导致引用失效
                    sub = &input_descriptions_[sub_id];

                    // 更改newRoute中的subscriber的接收消息id，为Reducer发布的消息ID
                    MessageNameRef reducer_pub_msg_name = reducer->getOutputMessageName();
                    sub->message_name = reducer_pub_msg_name;
                    sub->message_id = message_id_map_.at(reducer_pub_msg_name);
                    sub->reduce_method = ReduceMethod::STACK;       // reduce_method由reducer继承，重置reduce_method为STACK
                    sub->history_offset = 0;                        // history_offset由reducer继承，重置history_offset为0

                    message_routes_.at(message_id_map_.at(reducer_pub_msg_name)).second.insert(sub_id);
                    // 删除原有message到subscriber的消息路由
                    message_routes_.at(message_id).second.erase(sub_id);
                }
            }
        }

        adjusted_message_routes_ = message_routes_;     // 此时的路由表添加了reducer节点，为新的路由表
        message_routes_ = std::move(origin_routes);     // 回复原有的路由表

        // &*------ 修改stack_order，使用新的路由表 -------
        for (auto& [message_id, routes] : adjusted_message_routes_) {
            auto& [pub_ids, sub_ids] = routes;

            for (auto sub_id : sub_ids) {
                auto *sub = &input_descriptions_[sub_id];

                printf(FG_CYAN "stack_order size: %lu" FG_DEFAULT LINE_ENDL, sub->stack_order.size());

                if (sub->stack_order.size() == 0) {
                    for (auto pub_id : pub_ids) {
                        sub->stack_order.push_back( output_descriptions_[pub_id].node_name );
                    }
                }

                assert(sub->stack_order.size() == pub_ids.size());

                // 将stack_order转换为node_id
                for (auto& stack_node : sub->stack_order) {
                    if (node_id_map_.find(stack_node) == node_id_map_.end()) {
                        throw std::runtime_error("Invalid stack order");
                    }

                    // NodeNameRef -> NodeId -> Node Description -> Publisher DescriptionId
                    DescriptionId pub_id = node_descriptions_.at( node_id_map_.at(stack_node) ).output_map.at(message_id);
                    sub->stack_order_pub_id.push_back(pub_id);
                }
            }
        }

    }

    void buildMessageGraph() {
        /// 我们首先建立一个由Component作为节点，消息作为边的图。检查是否有仅订阅消息但没有发布消息的节点。
        /// 如果有，我们将其连接到一个名为no_pub的辅助节点上。之后，我们将删除no_pub节点及其后续节点。

        message_graph_.clear();

        // 添加所有组件节点
        for (const auto& node_des  : node_descriptions_) {
            Vertex v = boost::add_vertex({
                node_des.node_id,
                node_des.node_name,
                node_des.node_tag,
                NodeType::COMPONENT_NODE}, message_graph_);
            vertex_map[node_des.node_name] = v;
        }

        // 添加辅助节点，用于表示无发布者的消息
        Vertex no_pub_vertex = boost::add_vertex({
            NOPUB_NODE_ID,
            NOPUB_NODE_NAME,
            NOPUB_NODE_TAG,
            NodeType::AUX_NOPUB_NODE}, message_graph_);
        vertex_map["no_pub"] = no_pub_vertex;

        // 建立消息依赖关系
        for (const auto& [message_id, routes] : adjusted_message_routes_) {   // 遍历所有消息
            const auto& [pub_ids, sub_ids] = routes;

            for (const auto& sub_id : sub_ids) {   // 遍历所有订阅者
                const auto& sub = input_descriptions_[sub_id];
                if (pub_ids.empty()) {   // 没有发布者, 添加边到no_pub_vertex
                    Vertex sub_vertex = vertex_map.at(sub.node_name);
                    boost::add_edge(no_pub_vertex, sub_vertex, {message_id, NOPUB_NODE_ID, sub_id}, message_graph_);
                } else {
                    for (const auto& pub_id : pub_ids) {
                        const auto& pub = output_descriptions_[pub_id];
                        Vertex pub_vertex = vertex_map.at(pub.node_name);
                        Vertex sub_vertex = vertex_map.at(sub.node_name);
                        boost::add_edge(pub_vertex, sub_vertex, {message_id, pub_id, sub_id}, message_graph_);
                    }
                }
            }
        }

        // 删除no_pub的后续节点
        std::vector<Vertex> inactive_nodes;
        boost::breadth_first_search(message_graph_, vertex_map.at("no_pub"),
            boost::visitor(VertexVisitor(inactive_nodes)));

        // // 删除出度为0的节点
        // for (auto it = boost::vertices(active_graph_).first; it != boost::vertices(active_graph_).second; ++it) {
        //     if (boost::out_degree(*it, active_graph_) == 0) {
        //         inactive_nodes.push_back(*it);
        //     }
        // }

        active_graph_ = message_graph_;

        // 修剪后的图作为活动图
        for (const Vertex& v : inactive_nodes) {
            boost::remove_vertex(v, active_graph_);
        }

        // 打印所有会被删除的节点，用于调试
        for (auto v : inactive_nodes) {
            std::cout << "Inactive node: " << active_graph_[v].node_name << std::endl;
        }
    }

    void checkActiveGraphLoop() {
        /// 检查活动图中是否有循环依赖，如果有，则说明在消息传递过程中会出现循环。

        bool has_cycle = false;
        CycleDetector vis(has_cycle);

        boost::depth_first_search(active_graph_, boost::visitor(vis));

        if (has_cycle) {
            throw std::runtime_error("Circular dependency detected in message graph");
        }
    }

    void createMessageQueues() {
        /// 为每个消息创建一个消息队列，消息队列的历史长度为所有订阅者中的最大历史长度。


        // 生成当前的活跃消息路由表
        for(auto [ei, eind] = boost::edges(active_graph_); ei != eind; ei++) {
            auto &edge_properties = active_graph_[*ei];
            MessageId message_id = edge_properties.message_id;

            if(active_messages_routes_.find(message_id) == active_messages_routes_.end()) {
                active_messages_routes_[message_id] = {{},{}};
            }

            auto &[pub_ids, sub_ids] = active_messages_routes_.at(message_id);
            pub_ids.insert(edge_properties.pub_des_id);
            sub_ids.insert(edge_properties.sub_des_id);
        }

        // 遍历每一个节点的出边，建立消息队列
        for(auto [vi, vend] = boost::vertices(active_graph_); vi!=vend; vi++) {
            NodeId node_id = active_graph_[*vi].node_id;
            NodeNameRef node_name = active_graph_[*vi].node_name;
            VertexOutEdgeIterator ei, eind;
            for(std::tie(ei, eind) = boost::out_edges(*vi, active_graph_); ei!=eind; ei++) {
                const EdgeProperties &edge_props = active_graph_[*ei];
                MessageId message_id = edge_props.message_id;
                const MessageNameRef& message_name = messages_.at(message_id).first;
                OutputDescription &pub = output_descriptions_[ edge_props.pub_des_id ];

                int max_mq_history_offset = 0;

                // 遍历当前消息的所有接收者，得到最大历史长度
                for(auto sub_id : active_messages_routes_.at(message_id).second) {
                    const auto &sub = input_descriptions_[sub_id];

                    if(pub.shape != sub.shape) {
                        throw std::runtime_error("Shape mismatch for subscriber of message " + std::string(message_name));
                    }

                    if (sub.history_offset > max_mq_history_offset)
                        max_mq_history_offset = sub.history_offset;
                }

                // 创建MQ_Id, 修改OutputDescription，注明MessageQueueId
                pub.queue_id = message_queues_.size();
                message_queues_.push_back(std::make_unique<MessageQueue>(
                    node_id,
                    node_name,
                    message_id,
                    message_name,
                    pub.shape,
                    max_mq_history_offset+1));
            }
        }
    }

    void updateNodeDescription() {
        /// 更新节点描述信息，包括active_inputs, active_outputs, active_input_map, active_output_map
        /// 这些信息将在执行时用于消息传递。特别注意，如果使用了Reducer节点，那么需要将Reducer节点的输入消息名作为标识符。

        for(auto [vi, vend] = boost::vertices(active_graph_); vi!=vend; vi++) {
            NodeDescription &node_des = node_descriptions_.at( active_graph_[*vi].node_id );

            VertexInEdgeIterator ein_i, ein_ind;
            for(std::tie(ein_i, ein_ind) = boost::in_edges(*vi, active_graph_); ein_i!=ein_ind; ein_i++) {
                const EdgeProperties &edge_props = active_graph_[*ein_i];
                node_des.active_input_desc.insert({edge_props.message_id, edge_props.sub_des_id});
            }

            // 填充active_input, active_output
            VertexOutEdgeIterator eout_i, eout_ind;
            for(std::tie(eout_i, eout_ind) = boost::out_edges(*vi, active_graph_); eout_i!=eout_ind; eout_i++) {
                const EdgeProperties &edge_props = active_graph_[*eout_i];
                node_des.active_outputs_desc.insert({edge_props.message_id, edge_props.pub_des_id});
            }

            // 填充active_output_map
            for(auto [message_id, pub_id] : node_des.active_outputs_desc) {
                const MessageNameRef &message_name = output_descriptions_[pub_id].message_name;
                node_des.active_output_map[message_name] = output_descriptions_[pub_id].queue_id;
            }

            // 对于所有的active_input, 根据stack order的顺序查找对应消息的pub, 得到message_queue_id, 填充input_map
            for(auto [message_id, sub_id] : node_des.active_input_desc) {
                const MessageNameRef &message_name = input_descriptions_[sub_id].message_name;
                InputDescription &sub = input_descriptions_[sub_id];

                auto &active_pubs = active_messages_routes_.at(message_id).first;

                // 所有其他方法都由Reducer节点处理了
                assert (sub.reduce_method == ReduceMethod::STACK);

                std::vector<std::pair<MessageQueueId, int>> mq_ids {};
                for(DescriptionId pub_id : sub.stack_order_pub_id) {
                    // 如果消息不是活跃的，就跳过
                    if (active_pubs.find(pub_id) == active_pubs.end()) continue;

                    const auto &pub = output_descriptions_[pub_id];
                    mq_ids.push_back({pub.queue_id, sub.history_offset});
                }

                // 这里需要做一下特殊处理，如果是由Reducer节点发布的消息，需要将消息名的后缀.XXX去掉
                auto finder = reducers_.find(message_name);
                if (finder != reducers_.end()) {
                    // 使用原消息名作为节点的输入
                    node_des.active_input_map[finder->second->getMessageName()] = mq_ids;
                } else {
                    node_des.active_input_map[message_name] = mq_ids;
                }

            }

        }
    }

    void pruneGraphToOneStepIter() {
        /// 我们将带有历史消息的边删除，用于后续建立依赖图和执行顺序
        auto [ei, eind] = boost::edges(active_graph_);
        while (ei != eind) {
            auto current = ei++;
            const auto &sub = input_descriptions_[ active_graph_[*current].sub_des_id ];
            if (sub.history_offset > 0)
                boost::remove_edge(*current, active_graph_);
        }
    }

    void addTriggerToGraph() {
        /// 为每个触发器创建一个顶点，连接所有具有相同tag的节点，计算所有节点的距离

        trigger_graph_ = active_graph_;

        // 为每个触发器创建一个顶点
        for (auto& [trigger_tag, trigger_desc] : triggers_) {
            Vertex trigger_vertex = boost::add_vertex({
                TRIGGER_NODE_ID,
                TRIGGER_NODE_NAME,
                TRIGGER_NODE_TAG,
                NodeType::TRIGGER_NODE}, trigger_graph_);
            trigger_desc.trigger_vertex = trigger_vertex;

            // 连接所有具有相同tag的节点
            typename boost::graph_traits<Graph>::vertex_iterator vi, vend;
            for (std::tie(vi, vend) = boost::vertices(trigger_graph_); vi != vend; ++vi) {
                if (trigger_graph_[*vi].node_tag == trigger_tag) {
                    boost::add_edge(trigger_vertex, *vi, {
                        trigger_desc.trigger_message_id,
                        INVALID_PUB_ID,
                        INVALID_SUB_ID
                    }, trigger_graph_);
                }
            }
        }
        // ---- 拓扑排序确定依赖 ----

        // 拓扑排序
        std::list<Vertex> topo_order;
        boost::topological_sort(
            trigger_graph_,
            std::front_inserter(topo_order),
            boost::vertex_index_map(boost::get(boost::vertex_index, trigger_graph_))
        );

        // 根据拓扑顺序计算每个顶点的层级
        std::vector<int> orders(boost::num_vertices(trigger_graph_), 0);
        for (Vertex v : topo_order) {
            // 找所有的前驱，令 layer[v] = max(layer[u] + 1)
            Graph::in_edge_iterator in_i, in_end;
            for (std::tie(in_i, in_end) = boost::in_edges(v, trigger_graph_); in_i != in_end; ++in_i) {
                auto u = boost::source(*in_i, trigger_graph_);
                orders[v] = std::max(orders[v], orders[u] + 1);
            }
            // 记录节点的order
            if (trigger_graph_[v].node_id != TRIGGER_NODE_ID)
                node_order_[trigger_graph_[v].node_id] = orders[v];
        }

        // 收集每一层中的节点
        int max_layer = 0;
        for (auto l : orders) {
            max_layer = std::max(max_layer, l);
        }

        for (auto& [trigger_tag, trigger_desc] : triggers_) {
            trigger_desc.node_order.clear();
            trigger_desc.node_order.resize(max_layer + 1);

            // 第一个单元填充具体的order
            for(int i=0; i<=max_layer; i++) {
                trigger_desc.node_order[i] = {i-1, {}}; // 由于trigger节点是0，其他节点的order从1开始，所以减1，保证从0开始。
            }
            // 第二个单元填充并行执行的节点
            for (std::size_t i = 0; i < orders.size(); ++i) {
                if (trigger_graph_[i].node_tag == trigger_tag && trigger_graph_[i].node_id != TRIGGER_NODE_ID) {
                    trigger_desc.node_order[orders[i]].second.insert(trigger_graph_[i].node_id);
                }
            }
        }
    }

    void executeNode(NodeId node) {
        /// 执行节点，将节点的输出写入消息队列

        auto &node_des = node_descriptions_.at(node);
        auto &node_ptr = node_des.node;

        // 收集所有输入数据
        NodeExecInputType input_data;

        for (const auto& [message_name, sub_mq_ids] : node_des.active_input_map) {
            // input_data[message_name] = {};
            std::vector<TensorHandle> input_data_list;
            for (auto [mq_id, hist_off] : sub_mq_ids) {
                input_data_list.push_back(message_queues_[mq_id]->getHistoryTensorHandle(hist_off));
            }
            input_data.insert(std::make_pair(message_name, input_data_list));
        }

        NodeExecOutputType output_data;

        for (const auto& [message_name, mq_id] : node_des.active_output_map) {
            output_data.insert(std::make_pair(message_name, message_queues_[mq_id]->getWriteTensorPtr()));
        }

        // 所有输入就绪时执行组件
        node_ptr->onNodeExecute(input_data, output_data);
    }

    std::string generateGraphDot(const Graph& g, const std::string& graph_name, bool with_order=false) const {
        std::stringstream ss;
        ss << "digraph " << graph_name << " {\n";
        ss << "  rankdir=LR;\n";  // 从左到右的布局
        ss << "  node [shape=box];\n";  // 节点样式

        // 添加节点
        typename boost::graph_traits<Graph>::vertex_iterator vi, vend;
        for (std::tie(vi, vend) = boost::vertices(g); vi != vend; ++vi) {
            const auto& props = g[*vi];
            std::string color = "white";
            if (props.node_type == NodeType::TRIGGER_NODE) {
                color = "lightblue";
            } else if (props.node_type == NodeType::AUX_NOPUB_NODE) {
                color = "lightgray";
            }

            ss << "  \"" << props.node_name << "\" [style=filled,fillcolor=" << color
               << ",label=\"" << props.node_name << "\\n(" << props.node_tag << ")" +
                (with_order && props.node_id<node_order_.size() ? "\\norder=" + std::to_string(node_order_.at(props.node_id)) : "")
                + "\"];\n";
        }

        // 添加边
        typename boost::graph_traits<Graph>::edge_iterator ei, eend;
        for (std::tie(ei, eend) = boost::edges(g); ei != eend; ++ei) {
            const auto& src = g[boost::source(*ei, g)];
            const auto& dst = g[boost::target(*ei, g)];
            const auto& edge = g[*ei];

            ss << "  \"" << src.node_name << "\" -> \"" << dst.node_name
               << "\" [label=\"" << messages_[edge.message_id].first << "\"];\n";
        }

        ss << "}\n";
        return ss.str();
    }

    // 将DOT内容写入文件
    void saveGraphToDotFile(const std::string& dot_content, const std::string& filename) const {
        std::ofstream file(filename);
        if (file.is_open()) {
            file << dot_content;
            file.close();
        } else {
            std::cerr << "Failed to open file: " << filename << std::endl;
        }
    }

    // 导出图的可视化
    void exportGraphVisualization(const Graph& g, const std::string& name, const std::string& filename, bool with_order=false) const {
        std::string dot_content = generateGraphDot(g, name, with_order);
        saveGraphToDotFile(dot_content, filename);
    }

private:
    static constexpr const char* NOPUB_NODE_NAME = "no_pub";
    static constexpr const char* NOPUB_NODE_TAG = "default";
    static constexpr NodeId NOPUB_NODE_ID = 65536;


    static constexpr const char* TRIGGER_NODE_NAME = "trigger";
    static constexpr const char* TRIGGER_NODE_TAG = "$trigger";
    static constexpr NodeId TRIGGER_NODE_ID = 65537;


    static constexpr NodeId INVALID_NODE_ID = std::numeric_limits<NodeId>::max();
    static constexpr MessageId INVALID_MESSAGE_ID = std::numeric_limits<MessageId>::max();
    static constexpr DescriptionId INVALID_PUB_ID = std::numeric_limits<DescriptionId>::max();
    static constexpr DescriptionId INVALID_SUB_ID = std::numeric_limits<DescriptionId>::max();

    std::mutex mutex_;

    // 节点
    // std::vector<ComponentBase*> nodes_;
    std::vector<NodeDescription> node_descriptions_;
    std::unordered_map<NodeNameRef, NodeId> node_id_map_;

    // Reducer，由映射后的消息名到Reducer节点的映射
    std::unordered_map<MessageNameRef, std::unique_ptr<ReducerNode>> reducers_;

    std::unordered_map<NodeTagRef, TriggerDescription> triggers_;

    // 消息
    std::vector<std::pair<MessageNameRef, MessageShape>> messages_;
    std::unordered_map<MessageNameRef, MessageId> message_id_map_;
    std::vector<InputDescription> input_descriptions_;
    std::vector<OutputDescription> output_descriptions_;

    // 消息转发
    std::unordered_map<MessageId, MessageRoute> message_routes_;            // 原路由
    std::unordered_map<MessageId, MessageRoute> adjusted_message_routes_;   // 增加了reducer的路由

    // 节点消息图
    Graph message_graph_;
    Graph active_graph_;
    Graph trigger_graph_;      //
    std::unordered_map<NodeNameRef, Vertex> vertex_map;

    // 消息队列
    std::unordered_map<MessageId, MessageRoute> active_messages_routes_;
    std::vector<std::unique_ptr<MessageQueue>> message_queues_;

    // 执行顺序
    std::vector<std::vector<NodeId>> execution_order_;
    std::unordered_map<NodeId, int> node_order_;
    int current_execute_order_ = 0;
};

} // namespace core
} // namespace cuda_simulator


#endif // CUDASIM_MESSAGEBUS_HH