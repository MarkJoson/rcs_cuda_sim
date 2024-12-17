#pragma once
#include <core/context.hpp>
#include <deque>
#include <memory>
#include <unordered_map>

namespace environment {

class MessageQueue {
public:
    explicit MessageQueue(int maxHistoryLen)
        : maxHistoryLen_(maxHistoryLen)
        , validCount_(0)
        , latestUpdateTime_(0) {
        history_.set_capacity(maxHistoryLen);
    }

    core::Tensor getHistory(int offset) {
        auto cacheKey = offset;
        if (cacheValid_ && cache_.contains(cacheKey)) {
            return cache_[cacheKey];
        }

        auto result = history_[history_.size() - 1 - offset];
        cache_[cacheKey] = result;
        return result;
    }

    void append(const core::Tensor& data) {
        cacheValid_ = false;
        history_.push_back(data);
        validCount_ = std::min(maxHistoryLen_, validCount_ + 1);
        latestUpdateTime_++;
        return validCount_;
    }

    void reset() {
        history_.clear();
        validCount_ = 0;
        latestUpdateTime_ = 0;
        cache_.clear();
        cacheValid_ = false;
    }

    int getValidCount() const { return validCount_; }
    int getLatestUpdateTime() const { return latestUpdateTime_; }

private:
    boost::circular_buffer<core::Tensor> history_;
    int maxHistoryLen_;
    int validCount_;
    int latestUpdateTime_;
    std::unordered_map<int, core::Tensor> cache_;
    bool cacheValid_ = false;
};

class EnvironContext : public core::Context {
public:
    using QueueKey = std::pair<core::EntityId, core::MessageId>;

    struct QueueKeyHash {
        std::size_t operator()(const QueueKey& key) const {
            return std::hash<std::string>()(key.first + key.second);
        }
    };

    explicit EnvironContext(const std::shared_ptr<EnvironConfig>& config)
        : config_(config) {}

    std::shared_ptr<MessageQueue> getQueue(const core::EntityId& entityId,
                                         const core::MessageId& messageId) {
        QueueKey key(entityId, messageId);
        auto it = messageQueues_.find(key);
        return it != messageQueues_.end() ? it->second : nullptr;
    }

    void addQueue(const core::EntityId& entityId,
                 const core::MessageId& messageId,
                 std::shared_ptr<MessageQueue> queue) {
        messageQueues_[QueueKey(entityId, messageId)] = queue;
    }

    const std::shared_ptr<EnvironConfig>& getConfig() const { return config_; }

private:
    std::shared_ptr<EnvironConfig> config_;
    std::unordered_map<QueueKey, std::shared_ptr<MessageQueue>, QueueKeyHash> messageQueues_;
};

} // namespace environment