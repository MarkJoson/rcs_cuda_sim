#pragma once
#include "types.hpp"
#include <unordered_map>
#include <mutex>

namespace core {

class ContextDescriptor {
public:
    ContextDescriptor(ContextType type,
                     std::type_index contextClass,
                     std::shared_ptr<Component> provider,
                     std::vector<ContextType> deps = {})
        : contextType(type)
        , contextClass(contextClass)
        , provider(provider)
        , dependencies(deps) {}

    ContextType contextType;
    std::type_index contextClass;
    std::shared_ptr<Component> provider;
    std::vector<ContextType> dependencies;
};

class ContextRegistry {
public:
    static ContextRegistry& getInstance() {
        static ContextRegistry instance;
        return instance;
    }

    void registerProvider(const std::string& contextId,
                         std::shared_ptr<ContextDescriptor> descriptor) {
        std::lock_guard<std::mutex> lock(mutex_);
        descriptors_[contextId] = descriptor;
    }

    std::shared_ptr<ContextDescriptor> getDescriptor(const std::string& contextId) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = descriptors_.find(contextId);
        return it != descriptors_.end() ? it->second : nullptr;
    }

    std::unordered_map<std::string, std::shared_ptr<ContextDescriptor>>
    getAllDescriptors() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return descriptors_;
    }

private:
    ContextRegistry() = default;

    mutable std::mutex mutex_;
    std::unordered_map<std::string, std::shared_ptr<ContextDescriptor>> descriptors_;
};

class Context {
public:
    virtual ~Context() = default;

    template<typename T>
    T* getDependency(ContextType type) {
        auto it = dependencies_.find(type);
        return it != dependencies_.end() ?
               static_cast<T*>(it->second.get()) : nullptr;
    }

protected:
    std::unordered_map<ContextType, std::shared_ptr<Context>> dependencies_;
    std::shared_ptr<class EnvironConfig> envConfig_;
    std::weak_ptr<class EnvironManager> contextManager_;
};

} // namespace core