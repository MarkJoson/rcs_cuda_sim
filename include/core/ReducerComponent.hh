#ifndef CUDASIM_REDUCER_HH
#define CUDASIM_REDUCER_HH

#include <cassert>
#include <optional>

#include "core/ExecuteNode.hh"
#include "core_types.hh"
#include "Component.hh"

namespace cuda_simulator
{
namespace core
{

// &---------------------------------------------- ReducerComponent -------------------------------------------------------
class ReducerNode final : public ExecuteNode {

public:
    ReducerNode(
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


    void onNodeInit(SimulatorContext *context) override;
    void onNodeExecute(SimulatorContext* context, const NodeExecInputType &input, NodeExecOutputType &output) override;
    void onNodeReset(const TensorHandle& reset_flags, NodeExecStateType &state ) override;

    MessageNameRef getMessageName() { return message_name_; }
    MessageNameRef getOutputMessageName() { return output_message_name_; }
    ReduceMethod getReduceMethod() { return reduce_method_; }
    MessageShapeRef getMessageShape() { return message_shape_; }

private:
    MessageName message_name_;
    MessageName output_message_name_;
    ReduceMethod reduce_method_;
    int history_offset_;
    MessageShape message_shape_;
};

} // namespace core
} // namespace cuda_simulator


#endif // CUDASIM_MESSAGEBUS_HH