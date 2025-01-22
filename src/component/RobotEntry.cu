#include "component/RobotEntry.hh"
#include "core/MessageBus.hh"
#include "core/SimulatorContext.hh"
#include "core/storage/GTensorConfig.hh"
#include "geometry/GeometryManager.cuh"

using namespace cuda_simulator::core;

namespace cuda_simulator {
namespace robot_entry {

void RobotEntry::onNodeExecute(const core::NodeExecInputType &, core::NodeExecOutputType &, core::NodeExecStateType &) {
}

void RobotEntry::onNodeInit() { addOutput({"pose", {num_robot_per_env_, 4}}); }

void RobotEntry::onNodeStart() {}

void RobotEntry::onNodeReset(const core::GTensor &reset_flags, core::NodeExecStateType &state) {}

void RobotEntry::onEnvironGroupInit() {
  core::getGeometryManager()->createDynamicPolyObj(
      core::geometry::SimplePolyShapeDef({{0, 0.5}, {0.5, 0}, {-0.5, 0}, {0, -0.5}}));
}

void RobotEntry::setRobotPose(float4 robot_pose) {
  GTensor reset = GTensor::fromHostVectorNew<float>({robot_pose.x, robot_pose.y, robot_pose.z, robot_pose.w});
  auto &pose = core::getMessageBus()->getMessageQueue(name_, "pose")->getWriteTensorRef();
  pose.copyFrom(reset);
  GTensor &&random_tensor = GTensor::rands({pose.shape()[1], num_robot_per_env_, 4}) * 0.3f;
  pose += random_tensor;
}

} // namespace robot_entry
} // namespace cuda_simulator
