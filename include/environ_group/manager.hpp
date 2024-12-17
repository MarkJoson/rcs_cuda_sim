#pragma once
#include "context.hpp"
#include "config.hpp"
#include <core/component.hpp>
#include <memory>
#include <string>
#include <unordered_map>

namespace environment {

class EnvironManager : public std::enable_shared_from_this<EnvironManager> {
public:
    EnvironManager(const core::EntityId& groupId,
                  std::shared_ptr<EnvironConfig> config)
        : groupId_(groupId)
        , config_(config) {}

    void initContexts() {
        std::set<std::string> initialized;

        // 递归初始化上下文的lambda函数
        std::function<void(const std::string&)> initContext =
            [&](const std::string& contextId) {
            if (initialized.contains(contextId)) {
                return;
            }

            auto descriptor = core::ContextRegistry::getInstance()
                                                 .getDescriptor(contextId);
            if (!descriptor) {
                throw std::runtime_error(
                    "Context provider " + contextId + " not registered");
            }

            // 初始化依赖的上下文
            for (const auto& depType : descriptor->dependencies) {
                auto depId = findContextIdByType(depType);
                if (depId) {
                    initContext(*depId);
                }
            }

            // 收集依赖的上下文
            std::unordered_map<core::ContextType, std::shared_ptr<core::Context>> deps;
            for (const auto& depType : descriptor->dependencies) {
                auto depCtx = findContextByType(depType);
                if (!depCtx) {
                    throw std::runtime_error(
                        "Required dependency " + std::to_string(static_cast<int>(depType)) +
                        " not found");
                }
                deps[depType] = depCtx;
            }

            // 创建基础上下文
            auto baseContext = std::make_shared<core::Context>();
            baseContext->setDependencies(std::move(deps));
            baseContext->setConfig(config_);
            baseContext->setManager(shared_from_this());

            // 调用提供者的Init方法创建上下文
            auto context = descriptor->provider->onEnvironmentInit(baseContext);
            contexts_[contextId] = context;
            initialized.insert(contextId);
        };

        // 初始化所有已注册的上下文
        auto& registry = core::ContextRegistry::getInstance();
        for (const auto& [contextId, _] : registry.getAllDescriptors()) {
            initContext(contextId);
        }
    }

    void resetContexts(const core::Tensor& resetFlags) {
        auto& registry = core::ContextRegistry::getInstance();
        for (const auto& [contextId, descriptor] : registry.getAllDescriptors()) {
            if (auto context = contexts_[contextId]) {
                descriptor->provider->onEnvironmentReset(context, resetFlags);
            }
        }
    }

    std::shared_ptr<core::Context> getContext(const core::ContextId& contextId) {
        auto it = contexts_.find(contextId);
        return it != contexts_.end() ? it->second : nullptr;
    }

private:
    std::optional<std::string> findContextIdByType(core::ContextType type) {
        auto& registry = core::ContextRegistry::getInstance();
        for (const auto& [contextId, descriptor] : registry.getAllDescriptors()) {
            if (descriptor->contextType == type) {
                return contextId;
            }
        }
        return std::nullopt;
    }

    std::shared_ptr<core::Context> findContextByType(core::ContextType type) {
        auto contextId = findContextIdByType(type);
        return contextId ? contexts_[*contextId] : nullptr;
    }

private:
    core::EntityId groupId_;
    std::shared_ptr<EnvironConfig> config_;
    std::unordered_map<core::ContextId, std::shared_ptr<core::Context>> contexts_;
};

} // namespace environment