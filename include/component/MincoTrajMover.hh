#ifndef CUDASIM_COMPONENT_MINCOTRAJMOVER_HH
#define CUDASIM_COMPONENT_MINCOTRAJMOVER_HH

#include "core/Component.hh"
#include "core/core_types.hh"

namespace cuda_simulator {
namespace minco_traj_mover {



class MincoTrajMover : public core::Component {
  struct Priv;
public:
  MincoTrajMover();
  ~MincoTrajMover() override;

  void onEnvironGroupInit() override;

  void onNodeReset(const core::GTensor &reset_flags, core::NodeExecStateType &state) override;

  void onNodeStart() override;

  void onNodeInit() override;

  void onNodeExecute(const core::NodeExecInputType &input, core::NodeExecOutputType &output,
                     core::NodeExecStateType &state) override;
private:
  std::unique_ptr<Priv> priv_;
};


} // namespace minco_traj_mover
} // namespace cuda_simulator

#endif // CUDASIM_COMPONENT_MINCOTRAJMOVER_HH