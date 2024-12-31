#ifndef CUDASIM_SIMULATOR_CONTEXT_HH
#define CUDASIM_SIMULATOR_CONTEXT_HH

#include <memory>
#include <MessageBus.hh>
#include <Component.hh>
#include <EnvironGroupConfigItem.hh>


namespace cuda_simulator
{
namespace core
{

class SimulatorContext
{
public:
    SimulatorContext() {}
    virtual ~SimulatorContext() {}

    void registerComponent() {}
    void getMessageBus() {}
    void getEnvironGroupManager() {}

    void loadBasicComponent() {}

private:
    int num_max_exec_envs_;     // 应该是32的倍数
    int num_env_per_group;      // 每个环境组的env数量
    std::unique_ptr<MessageBus> message_bus;
};

} // namespace core
} // namespace cuda_simulator



#endif // CUDASIM_SIMULATOR_CONTEXT_HH