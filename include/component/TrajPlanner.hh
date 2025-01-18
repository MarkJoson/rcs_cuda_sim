#ifndef CUDASIM_COMPONENT_MINCO_TRAJ_MOVER_HH
#define CUDASIM_COMPONENT_MINCO_TRAJ_MOVER_HH

#include "core/Component.hh"

namespace cuda_simulator {
namespace component {

class MincoTrajMover : public core::Component {
public:
    MincoTrajMover() : Component("minco_traj_mover") { }

    void onEnvironGroupInit() override;

    void onNodeReset(const core::TensorHandle &reset_flags,
                     core::NodeExecStateType &state) override;

    void onNodeStart() override;

    void onNodeInit() override;

    void onNodeExecute(const core::NodeExecInputType &input,
                       core::NodeExecOutputType &output) override;
};


} // namespace component
} // namespace cuda_simulator



#endif // CUDASIM_COMPONENT_MINCO_TRAJ_MOVER_HH