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

// blockDim[x:32,y:6,z:1]
// gridDim[x:num_total_robot/32,y:1,z:1]
__global__ void mincoTrajCoeffIterate(int num_total_robot,                    //
                                      const float *__restrict__ target_poses, //
                                      float *__restrict__ coeff,              // [num_total_robot, 6, 3dim]
                                      float *__restrict__ force_out) {

  int robot_subidx = threadIdx.x;
  int robot_idx = blockIdx.x * blockDim.x + threadIdx.x;

  // 多余的线程退出
  if (robot_idx >= num_total_robot)
    return;

  using BlockStore = cub::BlockStore<float, 32, 3, cub::BLOCK_STORE_TRANSPOSE, 6>;
  __shared__ union {
    float coeff[6][3][32];
    typename BlockStore::TempStorage store;
  } temp_storage;

  // 每个block包含32个机器人，6个对应每机器人的工作线程

  // 加载数据。总共有num_total_robot*18个数据，每次加载1/3，每个机器人6*3dim个float系数
  int block_size = blockDim.x * blockDim.y;
  int tid = threadIdx.y * blockDim.x + threadIdx.x;
  for (int i = tid; i < block_size*3; i+=block_size) {
    // 每次读取本block中，1/3的机器人数据
    int copy_robot_idx = i / 18;
    int copy_coff = (i / 3) % 6;
    int copy_dim = copy_coff % 3;
    temp_storage.coeff[copy_coff][copy_dim][copy_robot_idx] = coeff[robot_idx * 18 + i];
  }

  __syncthreads();

  float4 pose = reinterpret_cast<const float4 *>(target_poses)[robot_idx];

  // 计算新的系数
  float new_coeff[3] = {0, 0, 0};
#pragma unroll
  for (int i = 0; i < 6; i++) {
#pragma unroll
    for (int dim = 0; dim < 3; dim++) {
      new_coeff[dim] += d_config.coeff_matF[threadIdx.y][i] * temp_storage.coeff[dim][i][robot_subidx];
    }
  }

#pragma unroll
  for (int i = 0; i < 3; i++) {
    new_coeff[i] += d_config.coeff_matG[threadIdx.y] * reinterpret_cast<const float *>(&pose)[i];
  }

#pragma unroll
  for (int i = 0; i < 3; i++) {
    temp_storage.coeff[threadIdx.y][i][threadIdx.x] = new_coeff[i];
  }

  __syncthreads();

  for (int i = 0; i < 18 * 32; i += block_size) {
    coeff[robot_idx * 18 + i] = temp_storage.coeff[threadIdx.y][i][threadIdx.x];
  }
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
