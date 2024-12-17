#pragma once
#include "publisher.hpp"
#include <optional>

namespace messaging {

class Subscriber : public MessageHandler {
public:
    Subscriber(const core::EntityId& componentId,
              const core::MessageId& messageId,
              const core::MessageShape& shape,
              int historyOffset = 0,
              std::optional<core::Tensor> historyPaddingVal = std::nullopt,
              core::ReduceMethod reduceMethod = core::ReduceMethod::STACK)
        : MessageHandler(componentId, messageId, shape)
        , historyOffset_(historyOffset)
        , acceptInvalidHistory_(!historyPaddingVal.has_value())
        , historyPaddingVal_(historyPaddingVal)
        , reduceMethod_(reduceMethod)
        , stackDim_(1) {}

    int getHistoryOffset() const { return historyOffset_; }
    bool acceptsInvalidHistory() const { return acceptInvalidHistory_; }
    const std::optional<core::Tensor>& getHistoryPaddingVal() const { return historyPaddingVal_; }
    core::ReduceMethod getReduceMethod() const { return reduceMethod_; }
    int getStackDim() const { return stackDim_; }

    void setStackDim(int dim) { stackDim_ = dim; }
    void setStackOrder(const std::vector<core::EntityId>& order) { stackOrder_ = order; }

    const std::vector<std::shared_ptr<Publisher>>& getPublishers() const {
        return publishers_;
    }

    void addPublisher(std::shared_ptr<Publisher> publisher) {
        publishers_.push_back(publisher);
    }

private:
    int historyOffset_;
    bool acceptInvalidHistory_;
    std::optional<core::Tensor> historyPaddingVal_;
    core::ReduceMethod reduceMethod_;
    int stackDim_;
    std::vector<core::EntityId> stackOrder_;
    std::vector<std::shared_ptr<Publisher>> publishers_;
};

} // namespace messaging