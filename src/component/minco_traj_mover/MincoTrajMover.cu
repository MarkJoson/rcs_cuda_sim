#include "component/MincoTrajMover.hh"
#include "core/MessageBus.hh"
#include "core/SimulatorContext.hh"
#include "core/storage/GTensorConfig.hh"
#include "geometry/GeometryManager.cuh"

#include "MincoTrajHelper.hh"

using namespace cuda_simulator::core;

namespace cuda_simulator {
namespace minco_traj_mover {

void MincoTrajMover::onNodeExecute(const core::NodeExecInputType &input, core::NodeExecOutputType &output) {}

void MincoTrajMover::onNodeInit() { }

void MincoTrajMover::onNodeStart() {}

void MincoTrajMover::onNodeReset(const core::TensorHandle &reset_flags, core::NodeExecStateType &state) {}

void MincoTrajMover::onEnvironGroupInit() { }


} // namespace robot_entry
} // namespace cuda_simulator
