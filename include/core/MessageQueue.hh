#ifndef CUDASIM_MESSAGEQUEUE_HH
#define CUDASIM_MESSAGEQUEUE_HH


#include <cassert>
#include <cstdint>
#include <optional>
#include <string>

#include "core/storage/Scalar.hh"
#include "core_types.hh"


namespace cuda_simulator {
namespace core {

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
            NumericalDataType dtype,
            size_t max_history_len,
            std::optional<TensorHandle> history_padding_val = std::nullopt);

    MessageQueue(const MessageQueue&) = delete;
    MessageQueue& operator=(const MessageQueue&) = delete;

    void allocate();

    const TensorHandle& getHistoryTensorHandle(size_t offset);

    TensorHandle& getWriteTensorRef();

    void resetEnvData(int env_group_id, int env_id);

    void reset();

private:
    std::string generateTensorUri(size_t index) const;

private:
    NodeId      pub_node_id_;
    NodeNameRef pub_node_name_;

    MessageId   message_id_;
    MessageNameRef message_name_;

    MessageShapeRef shape_;
    NumericalDataType dtype_;
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