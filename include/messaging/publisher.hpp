#pragma once
#include "core/types.hpp"
#include <memory>

namespace messaging {

class MessageHandler {
public:
    MessageHandler(const core::EntityId& componentId,
                  const core::MessageId& messageId,
                  const core::MessageShape& shape)
        : componentId_(componentId)
        , messageId_(messageId)
        , shape_(shape)
        , enabled_(true) {}

    virtual ~MessageHandler() = default;

    void setEnabled(bool enabled) { enabled_ = enabled; }
    bool isEnabled() const { return enabled_; }

    const core::EntityId& getComponentId() const { return componentId_; }
    const core::MessageId& getMessageId() const { return messageId_; }
    const core::MessageShape& getShape() const { return shape_; }

protected:
    core::EntityId componentId_;
    core::MessageId messageId_;
    core::MessageShape shape_;
    bool enabled_;
};

class IPublish {
public:
    virtual ~IPublish() = default;
    virtual void publish(const std::shared_ptr<core::Context>& context,
                        const core::EntityId& publishNode,
                        const core::MessageId& tensorId,
                        const core::Tensor& tensor) = 0;
};

class Publisher : public MessageHandler {
public:
    Publisher(const core::EntityId& componentId,
             const core::MessageId& messageId,
             const core::MessageShape& shape,
             std::shared_ptr<IPublish> publishInterface)
        : MessageHandler(componentId, messageId, shape)
        , publishInterface_(publishInterface) {}

    void publish(const std::shared_ptr<core::Context>& context,
                const core::Tensor& tensor) {
        if (!publishInterface_) return;

        publishInterface_->publish(context, componentId_, messageId_, tensor);
    }

    void addSubscriber(std::shared_ptr<class Subscriber> subscriber) {
        subscribers_.push_back(subscriber);
    }

    const std::vector<std::shared_ptr<class Subscriber>>& getSubscribers() const {
        return subscribers_;
    }

private:
    std::shared_ptr<IPublish> publishInterface_;
    std::vector<std::shared_ptr<class Subscriber>> subscribers_;
};

} // namespace messaging