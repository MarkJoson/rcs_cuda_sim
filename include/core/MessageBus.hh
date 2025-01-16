#ifndef CUDASIM_MESSAGEBUS_HH
#define CUDASIM_MESSAGEBUS_HH

#include "core/core_types.hh"
#include "core/ExecuteNode.hh"
#include "core/MessageQueue.hh"

namespace cuda_simulator
{
namespace core
{

class SimulatorContext;

using MessageQueueId = std::uint32_t;
using DescriptionId = std::int32_t;

// &---------------------------------------------- MessageBus -------------------------------------------------------
class MessageBus {
public:
    MessageBus();

    ~MessageBus();

    void registerComponent(ExecuteNode* node);

    const MessageShape& getMessageShape(const MessageNameRef& message_name);

    void registerInput(ExecuteNode* component, const ExecuteNode::NodeInputInfo &info);

    void registerOutput( ExecuteNode* component, const ExecuteNode::NodeOutputInfo &info );

    MessageQueue* getMessageQueue(const NodeNameRef &node_name, const MessageNameRef &message_name);

    void addTrigger(NodeTagRef trigger_tag);

    void buildGraph();

    void clearAll();

    void trigger(NodeTagRef trigger_tag);

    void resetExecuteOrder();

    int getCurrentExecuteOrder() const;

    int getNodeOrder(NodeId node_id) const;
private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};



} // namespace core
} // namespace cuda_simulator


#endif // CUDASIM_MESSAGEBUS_HH