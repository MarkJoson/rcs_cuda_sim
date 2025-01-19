#ifndef CUDASIM_REDUCER_HH
#define CUDASIM_REDUCER_HH

#include <cassert>

#include "core_types.hh"
#include "ExecuteNode.hh"

namespace cuda_simulator {
namespace core {

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
            const MessageShape &shape);


    void onNodeInit() override;

    void onNodeExecute(const NodeExecInputType &input, NodeExecOutputType &output, NodeExecStateType &state) override;

    void onNodeReset(const TensorHandle& reset_flags, NodeExecStateType &state ) override;

    MessageNameRef getMessageName() { return message_name_; }
    MessageNameRef getOutputMessageName() { return output_message_name_; }
    ReduceMethod getReduceMethod() { return reduce_method_; }
    MessageShapeRef getMessageShape() { return MessageShapeRef(message_shape_); }

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