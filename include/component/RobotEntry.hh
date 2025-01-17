#ifndef CUDASIM_ROBOT_ENTRY_HH
#define CUDASIM_ROBOT_ENTRY_HH

#include "core/Component.hh"
#include <cuda_runtime.h>

namespace cuda_simulator {
namespace robot_entry {

class RobotEntry : public core::Component {
public:
    RobotEntry(int robot_per_env)
        : Component("robot_entry"), num_robot_per_env_(robot_per_env) { }

    void onEnvironGroupInit() override;

    void onNodeReset(const core::TensorHandle &reset_flags,
                     core::NodeExecStateType &state) override;

    void onNodeStart() override;

    void onNodeInit() override;

    void onNodeExecute(const core::NodeExecInputType &input,
                       core::NodeExecOutputType &output) override;

    void setRobotPose(float4 robot_pose);

  private:
    uint32_t num_robot_per_env_;
};




} // namespace robot_entry
} // namespace cuda_simulator



#endif // CUDASIM_ROBOT_ENTRY_HH