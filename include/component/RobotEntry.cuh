#ifndef CUDASIM_ROBOT_ENTRY_HH
#define CUDASIM_ROBOT_ENTRY_HH

#include <cub/cub.cuh>

#include "component/map_generator/mapgen_postprocess.hh"
#include "core/Component.hh"
#include "core/SimulatorContext.hh"
#include "core/EnvGroupManager.cuh"
#include "core/MessageBus.hh"
#include "geometry/GeometryManager.cuh"
#include "core/core_types.hh"
#include "geometry/shapes.hh"

namespace cuda_simulator {
namespace robot_entry {

class RobotEntry : public core::Component {
public:
    RobotEntry(int robot_per_env)
        : Component("robot_entry"), num_robot_per_env_(robot_per_env) { }

    void onEnvironGroupInit() override {
        core::getGeometryManager()->createDynamicPolyObj(
            core::geometry::SimplePolyShapeDef({
                {0,0.5},
                {0.5,0},
                {-0.5,0},
                {0, -0.5}}));
    }

    void onNodeReset(
        const core::TensorHandle& reset_flags,
        core::NodeExecStateType &state) override {

    }

    void onNodeStart() override {

    }

    void onNodeInit() override {
        addOutput({"pose", {num_robot_per_env_, 4}});
    }

    void onNodeExecute( const core::NodeExecInputType &input,
        core::NodeExecOutputType &output) override {

    }
private:
    uint32_t num_robot_per_env_;
};

} // namespace robot_entry
} // namespace cuda_simulator



#endif // CUDASIM_LIDAR_SENSOR_HH