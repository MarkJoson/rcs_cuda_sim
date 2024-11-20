#ifndef __COMPONENT_H__
#define __COMPONENT_H__

#include "tree.h"
#include "TensorRegistry.h"

namespace RSG_SIM
{
class ExecutionNode
{
public:
    virtual ~ExecutionNode() = default;
    virtual void initialize() = 0;
    virtual void reset() = 0;
    virtual void compute() = 0;
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
};

class Component : public ExecutionNode, public TreeNode
{
public:
    virtual ~Component() = default;
    virtual void onNewEnvironGroupCreated(EnvironGroup* env_group);
};

} // namespace RSG_SIM




#endif