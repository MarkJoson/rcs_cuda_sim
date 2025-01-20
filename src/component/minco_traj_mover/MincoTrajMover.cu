#include "component/MincoTrajMover.hh"
#include "core/EnvGroupManager.cuh"
#include "core/MessageBus.hh"
#include "core/SimulatorContext.hh"
#include "core/storage/GTensorConfig.hh"
#include "cuda_helper.h"
#include "geometry/GeometryManager.cuh"

#include "MincoTrajHelper.hh"

#include <cub/cub.cuh>

// #include "MincoTrajHelper.hh"

using namespace cuda_simulator::core;

namespace cuda_simulator {
namespace minco_traj_mover {

struct MincoMoverConfig {
  float coeff_matF[6][6];  // 轨迹, 状态转移矩阵
  float coeff_matG[6];     // 目标位置, 输入矩阵
  float mat_ckpt[2][4][6]; // 速度, 加速度, 检查矩阵, [{vel/acc}, {ckpt0/1/2/3}, {coeff0/1/2/3/4/5}]
  float2 bound[2][3];      // [速度/加速度, x/y/z方向]
};

static __constant__ MincoMoverConfig d_config;

__global__ void mincoTrajCoeffIterate(int num_total_robot,                    //
                                      const float *__restrict__ target_poses, //
                                      float *__restrict__ coeffs               // [num_total_robot, 6, 3dim]
) {

  int robot_idx = blockIdx.x * blockDim.x + threadIdx.x;

  // 多余的线程退出
  if (robot_idx >= num_total_robot)
    return;

  // 每一个thread对应coeff为6*3, target_pose 为 1*3
  // new_coeff = matF * old_coeff + matG * target_pose

  // 6*6 乘 6*3
  float new_coeff[6][3];
  #pragma unroll
  for (int i = 0; i < 6; i++) {
    #pragma unroll
    for (int j = 0; j < 3; j++) {
      new_coeff[i][j] = 0;
      #pragma unroll
      for (int k = 0; k < 6; k++) {
        new_coeff[i][j] += d_config.coeff_matF[i][k] * coeffs[robot_idx * 6 * 3 + k * 3 + j];
      }
      new_coeff[i][j] += d_config.coeff_matG[i] * target_poses[robot_idx * 3 + j];
    }
  }
  #pragma unroll
  for (int i = 0; i < 6; i++) {
    #pragma unroll
    for (int j = 0; j < 3; j++) {
      coeffs[robot_idx * 6 * 3 + i * 3 + j] = new_coeff[i][j];
    }
  }
}

__global__ void mincoTrajBound(int num_total_robot, const float *__restrict__ coeffs, float *__restrict__ bound) {
  int robot_idx = blockIdx.x * blockDim.x + threadIdx.x;

  // 多余的线程退出
  if (robot_idx >= num_total_robot)
    return;

  // 速度检查点4个，加速度检查点4个，每个检查点3个维度

  // ckpt:[2][4][3] = mat_ckpt:[2][4][6] * coeff[6][3]
  // 所有检查点做min max规约，得到bound_min[2,对应速度/加速度][3,对应维度x,y,z]
  // 将target_pose(float4)中限制在bound内
}

// TODO. 与物理引擎的交互？

void MincoTrajMover::onNodeInit() {
  std::optional<Component::NodeOutputInfo> pose_info = getContext()->getOutputInfo("robot_entry", "pose");
  if (!pose_info.has_value()) {
    throw std::runtime_error("MincoTrajMover: robot_entry::pose not found.");
  }

  int num_robots = pose_info->shape[pose_info->shape.size() - 2];

  // 状态：位置x，位置y，角度z
  addState({"coeff", {num_robots, 6, 3}, NumericalDataType::kFloat32});
  // 输入：目标位置
  addInput({"tgt_pose", {num_robots, 3}, 0, ReduceMethod::STACK});
  // 输出：力
  addOutput({"force", {num_robots, 3}, NumericalDataType::kFloat32});
  // 输出：当前bound下的位置
  addOutput({"target_pos_bound",
             {
                 num_robots,
                 2,
             },
             NumericalDataType::kFloat32});

  MincoMoverConfig config;

  std::vector<float> mat_f, mat_g, mat_ckpt;
  auto system = MincoTrajSystem(0.1, 0.2);
  system.getMatF(&config.coeff_matF[0][0]);
  system.getMatG(config.coeff_matG);
  system.getMatCkpt(4, &(config.mat_ckpt[0][0][0]));
  // 速度
  config.bound[0][0] = {-1, 1};
  config.bound[0][1] = {-1, 1};
  config.bound[0][2] = {-1, 1};

  // 加速度
  config.bound[1][0] = {-1, 1};
  config.bound[1][1] = {-1, 1};
  config.bound[1][2] = {-1, 1};

  cudaMemcpyToSymbol(d_config, &config, sizeof(MincoMoverConfig));
}

void MincoTrajMover::onNodeExecute(const core::NodeExecInputType &, core::NodeExecOutputType &,
                                   core::NodeExecStateType &) {}

void MincoTrajMover::onNodeStart() {}

void MincoTrajMover::onNodeReset(const core::TensorHandle &reset_flags, core::NodeExecStateType &state) {}

void MincoTrajMover::onEnvironGroupInit() {}

} // namespace minco_traj_mover
} // namespace cuda_simulator
