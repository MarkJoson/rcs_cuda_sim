#ifndef CUDASIM_MESSAGEBUS_HH
#define CUDASIM_MESSAGEBUS_HH


#include <cassert>
#include <cstdint>
#include <memory>
#include <string>
#include <iostream>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <utility>


#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/topological_sort.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/property_map/property_map.hpp>
#include <boost/graph/properties.hpp>
#include <boost/graph/breadth_first_search.hpp>
#include <boost/graph/depth_first_search.hpp>
#include <boost/graph/visitors.hpp>

#include "core_types.hh"
#include "Component.hh"
#include "storage/ITensor.h"
#include "storage/TensorRegistry.h"

namespace cuda_simulator
{
namespace core
{


class MessageShape {
public:
    MessageShape(std::vector<int64_t> &shape) : shape_(shape.data()) {
        dim_ = shape.size();
    }

    MessageShape(int64_t *shape) : shape_(shape) {
        // 以0为终止符
        for (dim_ = 0; shape[dim_] != 0; dim_++);
    }

    int64_t operator[](int index) const {
        return shape_[index];
    }

    int64_t size() const {
        return dim_;
    }

    bool operator==(const MessageShape &other) const {
        if (dim_ != other.dim_) {
            return false;
        }

        for (int i = 0; i < dim_; i++) {
            if (shape_[i] != other.shape_[i]) {
                return false;
            }
        }

        return true;
    }
private:
    int64_t dim_;
    int64_t *shape_;
};

using MessageQueueId = std::uint32_t;
using DescriptionId = std::int32_t;


// &---------------------------------------------- MessageQueue -------------------------------------------------------
class MessageQueue {
public:
    explicit MessageQueue(NodeId pub_node_id, MessageId message_id, MessageNameRef message_name,
        MessageShape shape, size_t max_history_len, TensorHandle history_padding_val = nullptr)
        : pub_node_id_(pub_node_id)
        ,message_id_(message_id)
        , message_name_(message_name)
        , shape_(shape)
        , max_history_len_(max_history_len)
        , history_padding_val_(history_padding_val)
        , write_index_(0)
        , valid_count_(0) {
            allocate();
        }

    // 申请消息队列空间
    void allocate() {
        history_.resize(max_history_len_);
        // TODO.
    }

    // 获取历史消息
    TensorHandle getHistory(int offset) {
        if (offset >= max_history_len_) {
            throw std::runtime_error("Invalid history offset");
        }

        int index = (write_index_ - offset + max_history_len_) % max_history_len_;
        return history_[index];
    }

    // 获取当前的写入Tensor
    TensorHandle getWriteTensor() {
        auto result = history_[write_index_];
        write_index_ = (write_index_ + 1) % max_history_len_;
        return result;
    }

    void resetEnvData(int env_group_id, int env_id) {
        // TODO.
    }

    void reset() {
        valid_count_ = 0;
        write_index_ = 0;
    }

private:
    NodeId      pub_node_id_;
    MessageId   message_id_;
    MessageNameRef message_name_;
    MessageShape shape_;
    size_t max_history_len_;
    TensorHandle history_padding_val_;

    size_t write_index_ = 0;
    size_t valid_count_ = 0;

    // [History_len, env_group_size, env_cnt, *shape...]
    std::vector<TensorHandle> history_;
};

// &---------------------------------------------- Descriptions -------------------------------------------------------
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
    TensorHandle history_padding_val;
    // 消息队列ID
    MessageQueueId queue_id = INT_MAX;
};


struct NodeDescription {
    NodeId node_id;
    NodeNameRef node_name;
    NodeTagRef node_tag;
    ComponentBase* component;

    std::unordered_map<MessageId, DescriptionId> input_map;
    std::unordered_map<MessageId, DescriptionId> output_map;

    std::unordered_map<MessageId, DescriptionId> active_input_desc;
    std::unordered_map<MessageId, DescriptionId> active_outputs_desc;

    std::unordered_map<MessageNameRef, std::vector<
        std::pair<MessageQueueId, int>>> active_input_map;
    std::unordered_map<MessageNameRef, MessageQueueId> active_output_map;
};


// &---------------------------------------------- ReducerComponent -------------------------------------------------------
class ReducerComponent : public ComponentBase {

public:
    ReducerComponent(const MessageNameRef &message_name, ReduceMethod reduce_method, MessageShape shape)
        : ComponentBase(generateNodeName(message_name, reduce_method)), message_shape_(shape) {

    }

    MessageNameRef getOutputMessageName() {
        return getName();
    }

    static MessageName generateNodeName(const MessageNameRef &message_name, ReduceMethod reduce_method) {
        return std::string(message_name) + "." + std::to_string(static_cast<int>(reduce_method));
    }

    void onRegister(SimulatorContext* context) override {
        // 注册组件

    }

    void onEnvironGroupInit(SimulatorContext* context) override { };

    void onExecute(
        SimulatorContext* context,
        const std::unordered_map<MessageNameRef, TensorHandle> input,
        const std::unordered_map<MessageNameRef, TensorHandle> output
    ) override {

    };

    void onReset(
        TensorHandle reset_flags,
        std::unordered_map<MessageNameRef, TensorHandle> &state
    ) override {
        // TODO. 处理MessageQueue
    }

    // void onInit
private:
    MessageShape message_shape_;
};


// &---------------------------------------------- MessageBus -------------------------------------------------------
class MessageBus
{
public:
    MessageBus(SimulatorContext *context) : context_(context)  {};

    void registerComponent(ComponentBase* component) {
        std::lock_guard<std::mutex> lock(mutex_);

        createNodeId(component->getName(), component->getTag(), component);

        // 调用注册回调函数，初始化输入输出
        component->onRegister(context_);
    }

    void registerInput(
        ComponentBase* component,
        const MessageNameRef &message_name,
        const MessageShape &shape,
        int history_offset = 0,
        ReduceMethod reduce_method = ReduceMethod::STACK
    ) {
        std::lock_guard<std::mutex> lock(mutex_);

        if (history_offset < 0) {
            throw std::runtime_error("Invalid history offset");
        }

        MessageId message_id = lookUpOrCreateMessageId(message_name, shape);

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
            message_name,
            shape,
            history_offset,
            reduce_method,
            {}, {}
            }
        );

        // 更新节点接收消息列表
        node_descriptions_[node_id].input_map[message_id] = input_des_id;

        // 更新路由表
        message_routes_[message_id].second.insert(input_des_id);
    }


    void registerOutput(
        ComponentBase* component,
        const MessageNameRef &message_name,
        const MessageShape &shape,
        TensorHandle history_padding_val = nullptr
    ) {
        std::lock_guard<std::mutex> lock(mutex_);
        MessageId message_id = lookUpOrCreateMessageId(message_name, shape);

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
            message_name,
            shape,
            history_padding_val,
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


    void buildGraph() {
        std::lock_guard<std::mutex> lock(mutex_);

        // 1. 处理多发布者对一个消息的情况
        adjustMessageRoutes();

        // 2. 建立消息发送图
        buildMessageGraph();

        // 3. 构建活动图
        pruneGraph();

        // 4. 检查循环依赖
        checkActiveGraphLoop();

        // 5. 创建消息队列
        createMessageQueues();

        // 6. 更新Node的Input-MessageQueue映射
        updateNodeDescription();
    }

    void clearAll() {
        std::lock_guard<std::mutex> lock(mutex_);

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


private:    // ^---------------------------------------------- 私有定义 -------------------------------------------------------

    struct VertexProperties {
        NodeId node_id;
        NodeNameRef node_name;
        NodeTagRef node_tag;
    };
    // using EdgeProperties = boost::property<boost::edge_index_t, MessageId>;

    struct EdgeProperties {
        MessageId message_id;
        DescriptionId pub_des_id;
        DescriptionId sub_des_id;
    };

    using Graph = boost::adjacency_list<boost::vecS, boost::vecS, boost::directedS, VertexProperties, EdgeProperties>;
    using GraphId = std::string;
    using MessageRoute = std::pair<std::unordered_set<DescriptionId>, std::unordered_set<DescriptionId>>;
    using Vertex = boost::graph_traits<Graph>::vertex_descriptor;
    using VertexIterator = boost::graph_traits<Graph>::vertex_iterator;
    using EdgeIterator = boost::graph_traits<Graph>::edge_iterator;
    using VertexOutEdgeIterator = boost::graph_traits<Graph>::out_edge_iterator;


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

    NodeId createNodeId(const NodeNameRef& node_name, const NodeTagRef& node_tag, ComponentBase* component) {
        if(node_id_map_.find(node_name) != node_id_map_.end()) {
            throw std::runtime_error("Node has been registered!");
        }
        NodeId node_id = node_descriptions_.size();
        node_id_map_[node_name] = node_id;

        node_descriptions_.push_back({
            node_id,
            node_name,
            node_tag,
            component,
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

    ReducerComponent* lookUpOrCreateReducerNode(const MessageNameRef& message_name, const ReduceMethod& reduce_method, const MessageShape &shape) {
        // reducer 组件的命名规则为 messageName.reduceMethod
        // 新的消息id为 messageName.reduceMethod
        NodeName reducer_name = ReducerComponent::generateNodeName(message_name, reduce_method);
        ReducerComponent *p_com;

        auto finder = node_id_map_.find(reducer_name);
        if (finder == node_id_map_.end()) {
            auto new_component = std::make_unique<ReducerComponent>(message_name, reduce_method, shape);
            p_com = new_component.get();
            registerComponent(p_com);
            reducers_.insert(std::move(new_component));
        } else {
            p_com = static_cast<ReducerComponent*>(node_descriptions_[finder->second].component);
        }

        return p_com;
    }

    void adjustMessageRoutes() {
        std::unordered_map<MessageId, MessageRoute> newRoutes = message_routes_;

        // 遍历所有消息, 处理多发布者发布同一个一个消息的情况
        for (auto& [message_id, routes] : message_routes_) {
            auto& [pub_ids, sub_ids] = routes;

            for (auto sub_id : sub_ids) {
                auto& sub = input_descriptions_[sub_id];

                // 仅有一个publisher时，统一将reduce_method设置为stack
                if (pub_ids.size() == 1) {
                    sub.stack_order.clear();
                    sub.reduce_method = ReduceMethod::STACK;
                }

                // 如果不是STACK，则需要创建reducer组件
                if (sub.reduce_method != ReduceMethod::STACK) {
                    auto reducer = lookUpOrCreateReducerNode(sub.message_name, sub.reduce_method, sub.shape);

                    // 更改newRoute中的subscriber的接收消息id，为Reducer发布的消息ID
                    MessageNameRef reducer_pub_msg_name = reducer->getOutputMessageName();
                    sub.message_name = reducer_pub_msg_name;
                    sub.message_id = message_id_map_[reducer_pub_msg_name];

                    newRoutes[message_id_map_[reducer_pub_msg_name]].second.insert(sub_id);
                    // 删除原有message到subscriber的消息路由
                    newRoutes[message_id].second.erase(sub_id);
                }

                // 在最后只允许reduce_method==STACK
                // 为空时，按照默认顺序填充stack_order
                if (sub.stack_order.size() == 0) {
                    for (auto pub_id : pub_ids) {
                        sub.stack_order.push_back( output_descriptions_[pub_id].node_name );
                    }
                }

                assert(sub.stack_order.size() == pub_ids.size());

                // 将stack_order转换为node_id
                for (auto& stack_node : sub.stack_order) {
                    if (node_id_map_.find(stack_node) == node_id_map_.end()) {
                        throw std::runtime_error("Invalid stack order");
                    }

                    // NodeNameRef -> NodeId -> Node Description -> Publisher DescriptionId
                    DescriptionId pub_id = node_descriptions_[ node_id_map_[stack_node] ].output_map[message_id];
                    sub.stack_order_pub_id.push_back(pub_id);
                }
            }
        }
        adjusted_message_routes_ = std::move(newRoutes);
    }

    void buildMessageGraph() {
        /// 我们首先建立一个由Component作为节点，消息作为边的图。检查是否有仅订阅消息但没有发布消息的节点。
        /// 如果有，我们将其连接到一个名为no_pub的辅助节点上。在pruneGraph()中，我们将删除no_pub节点及其后续节点。

        message_graph_.clear();

        // 添加所有组件节点
        for (const auto& node_des  : node_descriptions_) {
            Vertex v = boost::add_vertex({
                node_des.node_id,
                node_des.node_name,
                node_des.node_tag,}, message_graph_);
            vertex_map[node_des.node_name] = v;
        }

        // 添加辅助节点，用于表示无发布者的消息
        Vertex no_pub_vertex = boost::add_vertex({NOPUB_NODE_ID, NOPUB_NODE_NAME, NOPUB_NODE_TAG}, message_graph_);
        vertex_map["no_pub"] = no_pub_vertex;

        // 建立消息依赖关系
        for (const auto& [message_id, routes] : adjusted_message_routes_) {   // 遍历所有消息
            const auto& [pub_ids, sub_ids] = routes;

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

    void pruneGraph() {
        /// 我们首先删除所有没有发布者的消息（no_pub后续节点），然后删除所有没有出度的节点，因为这些节点不会贡献数据。
        /// 最后，我们删除所有需要历史消息的边，因为我们将直接从消息队列中取出历史消息。

        std::vector<Vertex> inactive_nodes;

        // 删除no_pub的后续节点
        boost::breadth_first_search(message_graph_, vertex_map["no_pub"],
            boost::visitor(VertexVisitor(inactive_nodes)));

        // 删除出度为0的节点
        for (auto it = boost::vertices(active_graph_).first; it != boost::vertices(active_graph_).second; ++it) {
            if (boost::out_degree(*it, active_graph_) == 0) {
                inactive_nodes.push_back(*it);
            }
        }

        // 删除no_pub节点
        inactive_nodes.push_back(vertex_map["no_pub"]);

        // 打印所有会被删除的节点，用于调试
        for (auto v : inactive_nodes) {
            std::cout << "Inactive node: " << active_graph_[v].node_name << std::endl;
        }

        // 修剪后的图作为活动图
        active_graph_ = message_graph_;
        for (const Vertex& v : inactive_nodes) {
            boost::remove_vertex(v, active_graph_);
        }

        // 删除所有需要历史消息的边
        auto [ei, eind] = boost::edges(active_graph_);
        while (ei != eind) {
            auto current = ei++;
            const auto &sub = input_descriptions_[ active_graph_[*current].sub_des_id ];
            if (sub.history_offset > 0)
                boost::remove_edge(*current, active_graph_);
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
                const MessageNameRef& message_name = messages_[message_id].first;
                OutputDescription &pub = output_descriptions_[ edge_props.pub_des_id ];

                int mq_history_len = 0;

                // 遍历当前消息的所有接收者，得到最大历史长度
                for(auto sub_id : active_messages_routes_[message_id].second) {
                    const auto &sub = input_descriptions_[sub_id];

                    if(pub.shape != sub.shape) {
                        throw std::runtime_error("Shape mismatch for subscriber of message " + std::string(message_name));
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
        /// 更新节点描述信息，包括active_inputs, active_outputs, active_input_map, active_output_map
        /// 这些信息将在执行时用于消息传递。

        for(auto [vi, vend] = boost::vertices(active_graph_); vi!=vend; vi++) {
            NodeDescription &node_des = node_descriptions_[ active_graph_[*vi].node_id ];

            // 填充active_input, active_output
            VertexOutEdgeIterator ei, eind;
            for(std::tie(ei, eind) = boost::out_edges(*vi, active_graph_); ei!=eind; ei++) {
                const EdgeProperties &edge_props = active_graph_[*ei];
                node_des.active_input_desc.insert({edge_props.message_id, edge_props.sub_des_id});
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

                auto &active_pubs = active_messages_routes_[message_id].first;

                // 所有其他方法都由Reducer节点处理了
                assert (sub.reduce_method == ReduceMethod::STACK);

                std::vector<std::pair<MessageQueueId, int>> mq_ids {};
                for(DescriptionId pub_id : sub.stack_order_pub_id) {
                    // 如果消息不是活跃的，就跳过
                    if (active_pubs.find(pub_id) == active_pubs.end()) continue;

                    const auto &pub = output_descriptions_[pub_id];
                    mq_ids.push_back({pub.queue_id, sub.history_offset});
                }
                node_des.active_input_map[message_name] = mq_ids;
            }

        }
    }

    void addEntryVertex() {
        /// 添加一个入口节点，用于并行执行后续节点。这个节点将是所有其他无依赖节点的入口。

        Vertex entry_vertex = boost::add_vertex({0, ENTRY_NODE_NAME, ENTRY_NODE_TAG}, active_graph_);

        // TODO. 后续还需要根据TAG分离子图
    }

    void executeNode(NodeId node) {
        /// 执行节点，将节点的输出写入消息队列

        auto &node_des = node_descriptions_[node];
        auto &component = node_des.component;

        // 收集所有输入数据
        std::unordered_map<MessageNameRef, TensorHandle> input_data;


        for (const auto& [message_name, sub_mq_ids] : node_des.active_input_map) {
            // 如果只有一个消息发布者，简化操作
            if (sub_mq_ids.size() == 1) {
                auto [mq_id, hist_off] = sub_mq_ids[0];
                input_data[message_name] = message_queues_[mq_id]->getHistory(hist_off);
            } else {
                std::vector<TensorHandle> stack_data;

                for (auto [mq_id, hist_off] : sub_mq_ids) {
                    stack_data.push_back(message_queues_[mq_id]->getHistory(hist_off));
                }
                // TODO. 调用Tensor管理器进行STACK
            }
        }

        std::unordered_map<MessageNameRef, TensorHandle> output_data;

        for (const auto& [message_name, mq_id] : node_des.active_output_map) {
            output_data[message_name] = message_queues_[mq_id]->getWriteTensor();
        }

        // 所有输入就绪时执行组件
        component->onExecute(context_, input_data, output_data);
    }

private:
    static constexpr const char* NOPUB_NODE_NAME = "no_pub";
    static constexpr const char* NOPUB_NODE_TAG = "default";
    static constexpr NodeId NOPUB_NODE_ID = 65536;

    static constexpr const char* ENTRY_NODE_NAME = "entry";
    static constexpr const char* ENTRY_NODE_TAG = "default";
    static constexpr NodeId ENTRY_NODE_ID = 65537;

    SimulatorContext *context_;

    std::mutex mutex_;

    // 节点
    // std::vector<ComponentBase*> nodes_;
    std::vector<NodeDescription> node_descriptions_;
    std::unordered_map<NodeNameRef, NodeId> node_id_map_;

    // Reducer
    std::unordered_set<std::unique_ptr<ReducerComponent>> reducers_;

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
    std::unordered_map<NodeNameRef, Vertex> vertex_map;

    // 消息队列
    std::unordered_map<MessageId, MessageRoute> active_messages_routes_;
    std::vector<std::unique_ptr<MessageQueue>> message_queues_;

    // 执行顺序
    std::vector<std::vector<NodeId>> execution_order_;

};

} // namespace core
} // namespace cuda_simulator


#endif // CUDASIM_MESSAGEBUS_HH