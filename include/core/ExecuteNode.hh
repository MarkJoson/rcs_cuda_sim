#ifndef CUDASIM_EXECUTENODE_HH
#define CUDASIM_EXECUTENODE_HH

#include <optional>
#include <unordered_map>
#include "core/MessageShape.hh"
#include "core/storage/Scalar.hh"
#include "core_types.hh"

namespace cuda_simulator {
namespace core {

class SimulatorContext;

class ExecuteNode {
public:
    // NodeInputInfo保留原始数据（message name字符串， shape数组）
    struct NodeInputInfo {
        const MessageName name;
        const MessageShape shape;
        int history_offset;
        ReduceMethod reduce_method = ReduceMethod::STACK;
    };

    struct NodeOutputInfo {
        const MessageName name;
        const MessageShape shape;
        NumericalDataType dtype = NumericalDataType::kFloat32;
        std::optional<GTensor> history_padding_val = std::nullopt;
    };

    struct NodeStateInfo {
        const StateName name;
        const StateShape shape;
        NumericalDataType dtype = NumericalDataType::kFloat32;
        std::optional<GTensor> init_val = std::nullopt;
    };

    ExecuteNode(const NodeName &name, const NodeTag &tag = "default")
        : name_(name), tag_(tag) {}

    ExecuteNode(const NodeNameRef &name, const NodeTag &tag = "default")
        : name_(name), tag_(tag) {}

    virtual ~ExecuteNode() = default;

    // 初始化输入输出
    virtual void onNodeInit() = 0;
    // 初始化内部对象
    virtual void onNodeStart() {}
    // 执行
    virtual void onNodeExecute(const NodeExecInputType &input, NodeExecOutputType &output, NodeExecStateType &state) = 0;
    // 重置
    virtual void onNodeReset(const GTensor& reset_flags, NodeExecStateType &state) = 0;

    // Getters
    NodeNameRef getName() const { return name_; }
    NodeTagRef getTag() const { return tag_; }

    const NodeInputInfo &getInputInfo(const MessageName &message_name) const {
        return input_info_.at(message_name);
    }

    const NodeOutputInfo &getOutputInfo(const MessageName &message_name) const {
        return output_info_.at(message_name);
    }

    const NodeStateInfo &getStateInfo(const MessageName &state_name) const {
        return state_info_.at(state_name);
    }

    const auto& getInputs() const { return input_info_; }
    const auto& getOutputs() const { return output_info_; }
    const auto& getStates() const { return state_info_; }


protected:
    void addInput(const NodeInputInfo &info) {
        input_info_.insert({info.name, info});
    }

    void addOutput(const NodeOutputInfo &info) {
        output_info_.insert({info.name, info});
    }

    void addState(const NodeStateInfo &info) {
        state_info_.insert({info.name, info});
    }

protected:
    NodeName name_;
    NodeTag tag_;

    std::unordered_map<MessageName, NodeInputInfo> input_info_;
    std::unordered_map<MessageName, NodeOutputInfo> output_info_;
    std::unordered_map<MessageName, NodeStateInfo> state_info_;
};

} // namespace core
} // namespace cuda_simulator

#endif // CUDASIM_EXECUTENODE_HH
