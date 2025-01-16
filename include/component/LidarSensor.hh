#ifndef CUDASIM_LIDAR_SENSOR_HH
#define CUDASIM_LIDAR_SENSOR_HH

#include "core/Component.hh"

namespace cuda_simulator {
namespace lidar_sensor {

class LidarSensor : public core::Component {
public:
  LidarSensor();

  void onEnvironGroupInit() override;
  void onNodeReset(const core::TensorHandle &reset_flags, core::NodeExecStateType &state) override;
  void onNodeStart() override {}
  void onNodeInit() override;
  void onNodeExecute(const core::NodeExecInputType &input, core::NodeExecOutputType &output) override;

private:
    uint32_t num_inst_;
    core::MessageShape output_shape_;
};


} // namespace lidar_sensor
} // namespace cuda_simulator

#endif // CUDASIM_LIDAR_SENSOR_HH