#ifndef __COMPONENT_H__
#define __COMPONENT_H__

#include "tree.h"
#include "storage/TensorRegistry.h"

namespace RSG_SIM
{


class EnvironGroup;


class ExecutionCtx
{
public:
    EnvironGroup *group;
};


class ExecutionNode
{
public:
    virtual ~ExecutionNode() = default;
    virtual void initialize(ExecutionCtx &ctx) = 0;
    virtual void reset(ExecutionCtx &ctx) = 0;
    virtual void compute(ExecutionCtx &ctx) = 0;
};

class EnvironGroup
{
private:
    TensorRegistry registry_;
    int env_count_;
public:
    EnvironGroup(int env_count) : env_count_(env_count) {}

    int getEnvCount() { return env_count_; }

    template<typename T>
    auto addTensor(Component *com, const std::string &name, const std::vector<int64_t> shape) {
        std::string uri = PathUtils::joinPaths(com->getPath(), name);
        return registry_.createTensor<T>(uri, shape);
    }

    template<typename T>
    GTensor<T>* getTensor(const std::string &uri) {
        return static_cast<GTensor<T>*>(registry_.getTensor(uri));
    }
};

class Component : public ExecutionNode, public TreeNode
{
public:
    virtual ~Component() = default;
    virtual void onNewEnvironGroupCreated(EnvironGroup* env_group);
};

} // namespace RSG_SIM




#endif