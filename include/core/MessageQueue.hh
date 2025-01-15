#ifndef CUDASIM_MESSAGEQUEUE_HH
#define CUDASIM_MESSAGEQUEUE_HH


#include <cassert>
#include <cstdint>
#include <optional>
#include <string>

#include "core_types.hh"
#include "EnvGroupManager.cuh"
#include "SimulatorContext.hh"

namespace cuda_simulator
{
namespace core
{

using MessageQueueId = std::uint32_t;
using DescriptionId = std::int32_t;


// &---------------------------------------------- MessageQueue -------------------------------------------------------
class MessageQueue {
public:
    explicit MessageQueue(
            NodeId pub_node_id,
            NodeNameRef pub_node_name,
            MessageId message_id,
            MessageNameRef message_name,
            MessageShapeRef shape,
            size_t max_history_len,
            std::optional<TensorHandle> history_padding_val = std::nullopt)
        : pub_node_id_(pub_node_id)
        , pub_node_name_(pub_node_name)
        , message_id_(message_id)
        , message_name_(message_name)
        , shape_(shape)
        , max_history_len_(max_history_len)
        , history_padding_val_(history_padding_val)
        , write_index_(0)
        , valid_count_(0)
        , history_() {
            allocate();
        }

    // 申请消息队列空间
    void allocate() {
        history_.clear();

        for(size_t i = 0; i < max_history_len_; i++) {
            std::string tensor_uri = generateTensorUri(i);

            // 根据shape创建对应的tensor
            std::vector<int64_t> tensor_shape(shape_.begin(), shape_.end());
            history_.push_back(
                core::getEnvGroupMgr()->createTensor<float>(tensor_uri, tensor_shape));

            if(history_padding_val_.has_value()) {
                // 填充历史数据
                history_[i] = *history_padding_val_;
            }
        }
    }

    // 获取历史消息
    const TensorHandle& getHistoryTensorHandle(size_t offset) {
        if (offset >= max_history_len_) {
            throw std::runtime_error("Invalid history offset");
        }

        int index = (write_index_ - offset + max_history_len_) % max_history_len_;
        return history_[index];
    }

    // 获取当前的写入Tensor
    TensorHandle& getWriteTensorPtr() {
        auto& result = history_[write_index_];
        write_index_ = (write_index_ + 1) % max_history_len_;
        return result;
    }

    void resetEnvData(int env_group_id, int env_id) {
        // TODO.
    }

    void reset() {
        valid_count_ = 0;
        write_index_ = 0;
    }

private:
    std::string generateTensorUri(size_t index) const {
        // 生成唯一的tensor URI: message_name/node_id/history_index
        return std::string(message_name_) + "/" +
               std::string(pub_node_name_) + "/" +
               std::to_string(index);
    }

private:
    NodeId      pub_node_id_;
    NodeNameRef pub_node_name_;

    MessageId   message_id_;
    MessageNameRef message_name_;

    // TODO. batchsize

    MessageShapeRef shape_;
    size_t max_history_len_;
    std::optional<TensorHandle> history_padding_val_;

    size_t write_index_ = 0;
    size_t valid_count_ = 0;

    // [History_len, env_group_size, env_cnt, *shape...]
    std::vector<TensorHandle> history_;
};




} // namespace core
} // namespace cuda_simulator


#endif // CUDASIM_MESSAGEBUS_HH