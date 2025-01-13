#ifndef CUDASIM_EXECUTENODE_HH
#define CUDASIM_EXECUTENODE_HH

#include <optional>
#include "core_types.hh"

namespace cuda_simulator {
namespace core {

class SimulatorContext;

class ExecuteNode {
public:
    struct NodeInputInfo {
        const MessageNameRef &message_name;
        const MessageShape &shape;
        int history_offset;
        ReduceMethod reduce_method = ReduceMethod::STACK;
    };

    struct NodeOutputInfo {
        const MessageNameRef &message_name;
        const MessageShape &shape;
        std::optional<TensorHandle> history_padding_val = std::nullopt;
    };

    ExecuteNode(const NodeName &name, const NodeTag &tag = "default")
        : name_(name), tag_(tag) {}

    virtual ~ExecuteNode() = default;

    // 初始化输入输出
    virtual void onNodeInit(SimulatorContext* context) = 0;
    // 执行
    virtual void onNodeExecute(SimulatorContext* context, const NodeExecInputType &input, NodeExecOutputType &output) = 0;
    // 重置
    virtual void onNodeReset(const TensorHandle& reset_flags, NodeExecStateType &state) = 0;

    // Getters
    const NodeName& getName() const { return name_; }
    const NodeTag& getTag() const { return tag_; }

protected:
    void addInput(const NodeInputInfo &info) {
        input_info_.push_back(info);
    }

    void addOutput(const NodeOutputInfo &info) {
        output_info_.push_back(info);
    }

protected:
    NodeName name_;
    NodeTag tag_;

    std::vector<NodeInputInfo> input_info_;
    std::vector<NodeOutputInfo> output_info_;
};

} // namespace core
} // namespace cuda_simulator

#endif // CUDASIM_EXECUTENODE_HH
