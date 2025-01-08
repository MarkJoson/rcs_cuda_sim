#ifndef CUDASIM_COMPONENT_TRAJ_PLANNER_CU
#define CUDASIM_COMPONENT_TRAJ_PLANNER_CU

#include "core/Component.hh"

namespace cuda_simulator {
namespace component {

class TrajPlanner : public core::Component {
public:
    TrajPlanner() : Component("LidarSensor") {}

    void onEnvironGroupInit(core::SimulatorContext* context) override {
        // 初始化LidarSensor
    }

    void onReset(core::TensorHandle reset_flags, core::NodeExecStateType &state) override {
        // 重置LidarSensor
    }

    virtual void onRegister(core::SimulatorContext* context) {

    }

    virtual void onExecute( core::SimulatorContext* context, const core::NodeExecInputType &input, const core::NodeExecOutputType &output) {

    }
};



} // namespace component
} // namespace cuda_simulator



#endif // CUDASIM_COMPONENT_LIDAR_HH