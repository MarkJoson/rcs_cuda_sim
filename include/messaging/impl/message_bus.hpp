#pragma once
#include "messaging/bus.hpp"
#include "core/context.hpp"
#include <unordered_map>
#include <deque>
#include <mutex>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/topological_sort.hpp>

namespace messaging {

class MessageQueue {
public:
    explicit MessageQueue(size_t maxHistoryLen)
        : maxHistoryLen_(maxHistoryLen)
        , validCount_(0)
        , latestUpdateTime_(0) {}

    core::Tensor getHistory(int offset) {
        if (auto it = cache_.find(offset); it != cache_.end()) {
            return it->second;
        }

        auto result = history_[history_.size() - 1 - offset];
        cache_[offset] = result;
        return result;
    }

    void append(const core::Tensor& data) {
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
    std::deque<core::Tensor> history_;
    size_t maxHistoryLen_;
    size_t validCount_;
    size_t latestUpdateTime_;
    std::unordered_map<int, core::Tensor> cache_;
};

class MessageBus : public IMessageBus, public IPublish {
public:
    using Graph = boost::adjacency_list<
        boost::vecS, boost::vecS, boost::directedS,
        boost::property<boost::vertex_name_t, std::string>>;

    void publish(const std::shared_ptr<core::Context>& context,
                const core::EntityId& publishNode,
                const core::MessageId& tensorId,
                const core::Tensor& tensor) override {
        std::lock_guard<std::mutex> lock(mutex_);
        auto queueKey = std::make_pair(publishNode, tensorId);
        auto& queue = messageQueues_[queueKey];

        queue.append(tensor);

        auto& [publishers, subscribers] = messageRoutes_[tensorId];
        for (auto& sub : subscribers) {
            if (queue.getValidCount() >= sub->getHistoryOffset()) {
                triggerComponentExecution(context, sub->getComponentId());
            }
        }
    }

    // ... 其他接口实现 ...

    void registerComponent(const core::EntityId& componentId,
                         const core::GraphId& execGraph) override {
        std::lock_guard<std::mutex> lock(mutex_);
        components_[componentId] = execGraph;
    }

    std::shared_ptr<Publisher> createPublisher(
        const core::EntityId& componentId,
        const core::MessageId& messageId,
        const core::MessageShape& shape) override {
        std::lock_guard<std::mutex> lock(mutex_);

        auto publisher = std::make_shared<Publisher>(
            componentId, messageId, shape,
            std::dynamic_pointer_cast<IPublish>(shared_from_this()));

        messageRoutes_[messageId].first.push_back(publisher);
        return publisher;
    }

    std::shared_ptr<Subscriber> createSubscriber(
        const core::EntityId& componentId,
        const core::MessageId& messageId,
        const core::MessageShape& shape,
        int historyOffset,
        std::optional<core::Tensor> historyPaddingVal,
        core::ReduceMethod reduceMethod) override {
        std::lock_guard<std::mutex> lock(mutex_);

        auto subscriber = std::make_shared<Subscriber>(
            componentId, messageId, shape,
            historyOffset, historyPaddingVal, reduceMethod);

        messageRoutes_[messageId].second.push_back(subscriber);
        return subscriber;
    }

    void buildGraph() {
        std::lock_guard<std::mutex> lock(mutex_);

        // 1. 处理多发布者对一个消息的情况
        adjustMessageRoutes();

        // 2. 建立消息发送图
        buildMessageGraph();

        // 3. 禁用无源组件
        disableNoSourceComponents();

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
    void adjustMessageRoutes() {
        std::unordered_map<core::MessageId,
                          std::pair<std::vector<std::shared_ptr<Publisher>>,
                                  std::vector<std::shared_ptr<Subscriber>>>>
        newRoutes = messageRoutes_;

        for (const auto& [messageId, routes] : messageRoutes_) {
            const auto& [publishers, subscribers] = routes;
            std::set<core::EntityId> reducerNodes;

            for (const auto& sub : subscribers) {
                if (publishers.size() <= 1) continue;

                if (sub->getReduceMethod() == core::ReduceMethod::STACK) {
                    sub->setStackDim(1);
                    std::vector<core::EntityId> stackOrder;
                    for (const auto& pub : publishers) {
                        stackOrder.push_back(pub->getComponentId());
                    }
                    sub->setStackOrder(stackOrder);
                } else {
                    // 创建reducer组件
                    auto [reducerId, newMessageId] = generateReducerIds(messageId,
                                                                      sub->getReduceMethod());

                    if (reducerNodes.find(reducerId) == reducerNodes.end()) {
                        createReducerComponent(reducerId, messageId, newMessageId,
                                            sub->getShape(), sub->getReduceMethod());
                        reducerNodes.insert(reducerId);

                        // 更新路由
                        newRoutes[newMessageId].first = publishers;
                        newRoutes[messageId].first.clear();
                    }

                    // 更改subscriber的接收消息id
                    sub->getMessageId() = newMessageId;
                    newRoutes[newMessageId].second.push_back(sub);
                    auto it = std::find(newRoutes[messageId].second.begin(),
                                      newRoutes[messageId].second.end(), sub);
                    if (it != newRoutes[messageId].second.end()) {
                        newRoutes[messageId].second.erase(it);
                    }
                }
            }
        }

        messageRoutes_ = std::move(newRoutes);
    }

    void buildMessageGraph() {
        messageGraph_.clear();

        // 添加辅助节点
        boost::add_vertex("no_pub", messageGraph_);

        // 添加所有组件节点
        for (const auto& [componentId, _] : components_) {
            boost::add_vertex(componentId, messageGraph_);
        }

        // 建立消息依赖关系
        for (const auto& [messageId, routes] : messageRoutes_) {
            const auto& [publishers, subscribers] = routes;
            for (const auto& sub : subscribers) {
                if (publishers.empty()) {
                    boost::add_edge(boost::vertex("no_pub", messageGraph_),
                                  boost::vertex(sub->getComponentId(), messageGraph_),
                                  messageId, messageGraph_);
                } else {
                    for (const auto& pub : publishers) {
                        boost::add_edge(boost::vertex(pub->getComponentId(), messageGraph_),
                                      boost::vertex(sub->getComponentId(), messageGraph_),
                                      messageId, messageGraph_);
                    }
                }
            }
        }
    }

    void disableNoSourceComponents() {
        std::vector<core::EntityId> noSourceComponents;
        boost::breadth_first_search(
            messageGraph_,
            boost::vertex("no_pub", messageGraph_),
            boost::visitor(boost::make_bfs_visitor(
                boost::record_predecessors(
                    boost::make_vector_property_map<core::EntityId>(noSourceComponents),
                    boost::on_tree_edge()))));

        for (const auto& componentId : noSourceComponents) {
            if (auto it = components_.find(componentId); it != components_.end()) {
                it->second->setEnabled(false);
            }
        }
    }

    void buildActiveGraph() {
        activeGraph_.clear();

        // 复制所有启用的节点
        boost::graph_traits<Graph>::vertex_iterator vi, vi_end;
        for (boost::tie(vi, vi_end) = boost::vertices(messageGraph_); vi != vi_end; ++vi) {
            core::EntityId componentId = boost::get(boost::vertex_name, messageGraph_, *vi);
            if (componentId == "no_pub") continue;

            auto it = components_.find(componentId);
            if (it != components_.end() && it->second->isEnabled()) {
                boost::add_vertex(componentId, activeGraph_);
            }
        }

        // 复制启用节点间的边
        boost::graph_traits<Graph>::edge_iterator ei, ei_end;
        for (boost::tie(ei, ei_end) = boost::edges(messageGraph_); ei != ei_end; ++ei) {
            core::EntityId sourceId = boost::get(boost::vertex_name, messageGraph_,
                                               boost::source(*ei, messageGraph_));
            core::EntityId targetId = boost::get(boost::vertex_name, messageGraph_,
                                               boost::target(*ei, messageGraph_));

            if (sourceId == "no_pub" || targetId == "no_pub") continue;

            auto sourceIt = components_.find(sourceId);
            auto targetIt = components_.find(targetId);

            if (sourceIt != components_.end() && targetIt != components_.end() &&
                sourceIt->second->isEnabled() && targetIt->second->isEnabled()) {
                boost::add_edge(boost::vertex(sourceId, activeGraph_),
                              boost::vertex(targetId, activeGraph_),
                              boost::get(boost::edge_name, messageGraph_, *ei),
                              activeGraph_);
            }
        }
    }

    void checkActiveGraphMessageCompatible() {
        for (const auto& [messageId, routes] : messageRoutes_) {
            const auto& [publishers, subscribers] = routes;
            if (publishers.empty() || subscribers.empty()) continue;

            const auto& pubShape = publishers[0]->getShape();
            // 检查所有发布者的形状
            for (size_t i = 1; i < publishers.size(); ++i) {
                if (publishers[i]->getShape() != pubShape) {
                    throw std::runtime_error(
                        "Inconsistent shapes for publishers of message " + messageId);
                }
            }

            // 检查所有订阅者的形状
            for (const auto& sub : subscribers) {
                if (sub->getShape() != pubShape) {
                    throw std::runtime_error(
                        "Shape mismatch for subscriber of message " + messageId);
                }
            }
        }
    }

    void checkActiveGraphCycles() {
        try {
            std::vector<core::EntityId> order;
            boost::topological_sort(activeGraph_, std::back_inserter(order));
        } catch (const boost::not_a_dag&) {
            throw std::runtime_error("Circular dependency detected in message graph");
        }
    }

    void updateComponentsRouteRef() {
        for (auto& [componentId, component] : components_) {
            // 更新发布者的订阅者引用
            for (auto& pub : component->getPublishers()) {
                pub->getSubscribers().clear();
                const auto& [_, subscribers] = messageRoutes_[pub->getMessageId()];
                for (const auto& sub : subscribers) {
                    if (sub->isEnabled()) {
                        pub->addSubscriber(sub);
                    }
                }
            }

            // 更新订阅者的发布者引用
            for (auto& sub : component->getSubscribers()) {
                sub->getPublishers().clear();
                const auto& [publishers, _] = messageRoutes_[sub->getMessageId()];
                for (const auto& pub : publishers) {
                    if (pub->isEnabled()) {
                        sub->addPublisher(pub);
                    }
                }
            }
        }
    }

    void buildExecutionGraph() {
        executionGraph_.clear();

        // 添加所有执行图ID作为节点
        for (const auto& [componentId, graphId] : components_) {
            if (boost::vertex(graphId, executionGraph_) == boost::null_vertex()) {
                boost::add_vertex(graphId, executionGraph_);
            }
        }

        // 为相同执行图的节点添加边
        boost::graph_traits<Graph>::edge_iterator ei, ei_end;
        for (boost::tie(ei, ei_end) = boost::edges(activeGraph_); ei != ei_end; ++ei) {
            core::EntityId sourceId = boost::get(boost::vertex_name, activeGraph_,
                                               boost::source(*ei, activeGraph_));
            core::EntityId targetId = boost::get(boost::vertex_name, activeGraph_,
                                               boost::target(*ei, activeGraph_));

            auto sourceGraphId = components_[sourceId]->getGraphId();
            auto targetGraphId = components_[targetId]->getGraphId();

            if (sourceGraphId != targetGraphId) {
                boost::add_edge(boost::vertex(sourceGraphId, executionGraph_),
                              boost::vertex(targetGraphId, executionGraph_),
                              executionGraph_);
            }
        }
    }

    void generateExecutionOrder() {
        executionOrder_.clear();

        std::vector<core::GraphId> order;
        boost::topological_sort(executionGraph_, std::back_inserter(order));

        // 将排序结果分组
        std::vector<std::vector<core::EntityId>> groups;
        for (const auto& graphId : order) {
            std::vector<core::EntityId> group;
            for (const auto& [componentId, component] : components_) {
                if (component->getGraphId() == graphId && component->isEnabled()) {
                    group.push_back(componentId);
                }
            }
            if (!group.empty()) {
                executionOrder_.push_back(std::move(group));
            }
        }
    }

    void triggerComponentExecution(const std::shared_ptr<core::Context>& context,
                                 const core::EntityId& componentId) {
        auto component = components_.find(componentId);
        if (component == components_.end() || !component->second->isEnabled()) {
            return;
        }

        // 收集所有输入数据
        std::map<core::MessageId, core::Tensor> inputData;
        bool allInputsReady = true;

        for (const auto& sub : component->second->getSubscribers()) {
            if (!sub->isEnabled()) continue;

            // 获取发布者数据
            assert(sub->getPublishers().size() == 1);
            auto pub = sub->getPublishers()[0];
            auto queueKey = std::make_pair(pub->getComponentId(), sub->getMessageId());
            auto& queue = messageQueues_[queueKey];

            if (sub->getHistoryOffset() == 0) {
                if (queue.getValidCount() == 0) return;
                inputData[sub->getMessageId()] = queue.getHistory(0);
            } else {
                if (queue.getValidCount() < sub->getHistoryOffset() + 1) {
                    if (!sub->acceptsInvalidHistory()) {
                        allInputsReady = false;
                        break;
                    }
                    inputData[sub->getMessageId()] = *sub->getHistoryPaddingVal();
                } else {
                    inputData[sub->getMessageId()] = queue.getHistory(sub->getHistoryOffset());
                }
            }
        }

        // 所有输入就绪时执行组件
        if (allInputsReady) {
            component->second->onExecute(context, inputData);
        }
    }

    std::pair<core::EntityId, core::MessageId> generateReducerIds(
        const core::MessageId& messageId,
        core::ReduceMethod reduceMethod) {
        std::string reducerId = "_" + messageId + "." + std::to_string(static_cast<int>(reduceMethod));
        std::string newMessageId = messageId + "." + std::to_string(static_cast<int>(reduceMethod));
        return {reducerId, newMessageId};
    }

private:
    std::mutex mutex_;
    std::unordered_map<core::EntityId, std::shared_ptr<core::Component>> components_;
    std::unordered_map<core::EntityId, core::GraphId> component_graph_ids_;
    std::unordered_map<std::pair<core::EntityId, core::MessageId>, MessageQueue> messageQueues_;
    std::unordered_map<core::MessageId,
                      std::pair<std::vector<std::shared_ptr<Publisher>>,
                               std::vector<std::shared_ptr<Subscriber>>>> messageRoutes_;
    Graph messageGraph_;
    Graph activeGraph_;
    Graph executionGraph_;
    std::vector<std::vector<core::EntityId>> executionOrder_;
};

} // namespace messaging