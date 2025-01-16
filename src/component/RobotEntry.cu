#include "geometry/GeometryManager.cuh"
#include "core/SimulatorContext.hh"
#include "component/RobotEntry.hh"

namespace cuda_simulator {
namespace robot_entry {

void RobotEntry::onNodeExecute(const core::NodeExecInputType &input,
                               core::NodeExecOutputType &output) {}

void RobotEntry::onNodeInit() {
    addOutput({"pose", {
        num_robot_per_env_,
        4}});
}

void RobotEntry::onNodeStart() {}

void RobotEntry::onNodeReset(const core::TensorHandle &reset_flags,
                             core::NodeExecStateType &state) {}

void RobotEntry::onEnvironGroupInit() {
  core::getGeometryManager()->createDynamicPolyObj(
      core::geometry::SimplePolyShapeDef(
          {{0, 0.5}, {0.5, 0}, {-0.5, 0}, {0, -0.5}}));
}

} // namespace robot_entry
} // namespace cuda_simulator
