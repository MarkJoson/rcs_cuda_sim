#include "core/ReducerNode.hh"

namespace cuda_simulator {
namespace core {

ReducerNode::ReducerNode(
                                           const NodeName &reducer_name,
                                           const NodeTag &reducer_tag,
                                           const MessageNameRef &message_name,
                                           const MessageNameRef &message_output_name,
                                           ReduceMethod reduce_method,
                                           int history_offset,
                                           const MessageShape &shape)
    : ExecuteNode(reducer_name, reducer_tag)
    , message_name_(message_name)
    , output_message_name_(message_output_name)
    , reduce_method_(reduce_method)
    , history_offset_(history_offset)
    , message_shape_(shape)
{ }

void ReducerNode::onNodeInit() {
    addInput({message_name_, message_shape_, history_offset_});
    addOutput({output_message_name_, message_shape_});
}

void ReducerNode::onNodeExecute(const NodeExecInputType &input, NodeExecOutputType &output) {
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
}

void ReducerNode::onNodeReset(const TensorHandle& reset_flags, NodeExecStateType &state ) {
    // TODO. 处理MessageQueue
}

}
}
