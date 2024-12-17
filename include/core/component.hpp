#pragma once
#include "types.hpp"
#include "context.hpp"
#include <memory>
#include <vector>

namespace core {

class Component : public std::enable_shared_from_this<Component> {
public:
    Component(EntityId id, GraphId graphId, ContextId contextId,
             std::shared_ptr<MessageBus> messageBus)
        : id_(id)
        , graphId_(graphId)
        , contextId_(contextId)
        , messageBus_(messageBus)
        , enabled_(true) {}

    virtual ~Component() = default;

    // 纯虚函数
    virtual void onRegister() = 0;
    virtual void onExecute(const std::shared_ptr<Context>& context,
                         const std::map<MessageId, Tensor>& input) = 0;
    virtual void onReset(const std::shared_ptr<Context>& context,
                       const Tensor& resetFlag) = 0;
    virtual void onEnabledChanged(bool enabled) = 0;

    // 虚函数
    virtual std::shared_ptr<Context> onEnvironmentInit(
        const std::shared_ptr<Context>& baseContext) {
        return nullptr;
    }

    virtual void onEnvironmentReset(const std::shared_ptr<Context>& context,
                                  const Tensor& resetFlags) {}

    void setEnabled(bool enabled) {
        if (enabled_ == enabled) return;

        enabled_ = enabled;

        for (auto& pub : publishers_) {
            pub->setEnabled(enabled);
        }

        for (auto& sub : subscribers_) {
            sub->setEnabled(enabled);
        }

        onEnabledChanged(enabled);
    }

    // Getter方法
    const EntityId& getId() const { return id_; }
    const GraphId& getGraphId() const { return graphId_; }
    const ContextId& getContextId() const { return contextId_; }
    bool isEnabled() const { return enabled_; }

protected:
    template<typename T>
    std::shared_ptr<T> createSubscriber(
        const MessageId& messageId,
        const MessageShape& shape,
        int historyOffset = 0,
        std::optional<Tensor> historyPaddingVal = std::nullopt,
        ReduceMethod reduceMethod = ReduceMethod::STACK);

    template<typename T>
    std::shared_ptr<T> createPublisher(
        const MessageId& messageId,
        const MessageShape& shape);

private:
    EntityId id_;
    GraphId graphId_;
    ContextId contextId_;
    std::shared_ptr<MessageBus> messageBus_;
    bool enabled_;

    std::vector<std::shared_ptr<class Publisher>> publishers_;
    std::vector<std::shared_ptr<class Subscriber>> subscribers_;
};

} // namespace core