#include "core/ReducerComponent.hh"
#include "core/MessageBus.hh"
#include "core/SimulatorContext.hh"

namespace cuda_simulator
{
namespace core
{

void ReducerComponent::onRegister(SimulatorContext* context) {
    auto* msg_bus = context->getMessageBus();
    msg_bus->registerInput(this, message_name_, message_shape_, history_offset_);
    msg_bus->registerOutput(this, output_message_name_, message_shape_);
}

void ReducerComponent::onExecute( SimulatorContext* context, const NodeExecInputType &input, NodeExecOutputType &output ) {
    switch (reduce_method_) {
        case ReduceMethod::STACK:
            throw std::runtime_error("STACK method not supported in ReducerComponent");
            break;
        case ReduceMethod::SUM:
            output.begin()->second.gatherSum(input.begin()->second);
            break;
        case ReduceMethod::MAX:
            output.begin()->second.gatherMax(input.begin()->second);
            break;
        case ReduceMethod::MIN:
            output.begin()->second.gatherMin(input.begin()->second);
            break;
        case ReduceMethod::AVERAGE:
            output.begin()->second.gatherMean(input.begin()->second);
            break;
        default:
            throw std::runtime_error("Invalid reduce method");
    }
};

void ReducerComponent::onReset( const TensorHandle& reset_flags, NodeExecStateType &state ) {
    // TODO. 处理MessageQueue
}

} // namespace core
} // namespace cuda_simulator

