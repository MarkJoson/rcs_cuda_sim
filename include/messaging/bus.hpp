#pragma once
#include "core/types.hpp"
#include <memory>

namespace messaging {

class IMessageBus {
public:
    virtual ~IMessageBus() = default;

    virtual void registerComponent(
        const core::EntityId& componentId,
        const core::GraphId& execGraph) = 0;

    virtual std::shared_ptr<class Publisher> createPublisher(
        const core::EntityId& componentId,
        const core::MessageId& messageId,
        const core::MessageShape& shape) = 0;

    virtual std::shared_ptr<class Subscriber> createSubscriber(
        const core::EntityId& componentId,
        const core::MessageId& messageId,
        const core::MessageShape& shape,
        int historyOffset,
        std::optional<core::Tensor> historyPaddingVal,
        core::ReduceMethod reduceMethod) = 0;
};

} // namespace messaging