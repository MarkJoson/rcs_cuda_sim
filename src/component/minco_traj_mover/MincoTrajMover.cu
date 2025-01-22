#include <cub/cub.cuh>
#include <cublas_v2.h>

#include "MincoTrajHelper.hh"
#include "component/MincoTrajMover.hh"
#include "core/EnvGroupManager.cuh"
#include "core/MessageBus.hh"
#include "core/SimulatorContext.hh"
#include "core/storage/GTensorConfig.hh"
#include "cuda_helper.h"
#include "geometry/GeometryManager.cuh"

// #include "MincoTrajHelper.hh"

using namespace cuda_simulator::core;

namespace cuda_simulator {
namespace minco_traj_mover {

#define CUBLAS_CHECK(err)                                                      \
  do {                                                                         \
    cublasStatus_t err_ = (err);                                               \
    if (err_ != CUBLAS_STATUS_SUCCESS) {                                       \
      std::printf("cublas error %d at %s:%d\n", err_, __FILE__, __LINE__);     \
      throw std::runtime_error("cublas error");                                \
    }                                                                          \
  } while (0)

constexpr int NUM_CKPTS = 4;

struct MincoTrajMover::Priv {
  cublasHandle_t cublas_handle_;
  float *d_matF; // 轨迹, 状态转移矩阵(6x6), col-major
  float *d_matG; // 目标位置, 输入矩阵(6x1), col-major
  float *d_matL; // 速度, 加速度, 检查矩阵(6x6), [{vel/acc}, {ckpt0/1/2}, {coeff0/1/2/3/4/5}]
  TensorHandle matCmGinv; // 检查矩阵乘以输入矩阵(2xNUM_CKPTSx1)
  TensorHandle bound;     // 2xNUM_CKPTSx3 [速度/加速度, ckpts, x/y/z方向]
};

MincoTrajMover::MincoTrajMover()
    : Component("minco_traj_mover"), priv_(std::make_unique<Priv>()) {

std::vector<float> mat_f, mat_g, mat_ckpt;
  auto system = MincoTrajSystem(0.1, 0.2);
  // system.getMatF(&config.coeff_matF[0][0]);
  // system.getMatG(config.coeff_matG);
  // system.getMatCkpt(4, &(priv_->mat_ckpt[0][0][0]));


  // cudaMemcpyToSymbol(d_config, &config, sizeof(MincoMoverConfig));

  priv_->matCmGinv.reshape({2,NUM_CKPTS,1});

  float max_vel_x = 1;
  float max_vel_y = 1;
  float max_vel_z = 1;
  float max_acc_x = 1;
  float max_acc_y = 1;
  float max_acc_z = 1;

  priv_->bound = TensorHandle::fromHostVectorNew<float>(
      {max_vel_x, max_vel_y, max_vel_z, max_acc_x, max_acc_y, max_acc_z});
  priv_->bound.reshape({2,3});
  priv_->bound = priv_->bound.expand({2,NUM_CKPTS,3});

  priv_->bound *= priv_->matCmGinv;
}

MincoTrajMover::~MincoTrajMover() {
  CUBLAS_CHECK(cublasDestroy(priv_->cublas_handle_));
}

// TODO. 与物理引擎的交互？

void MincoTrajMover::onNodeInit() {
  std::optional<Component::NodeOutputInfo> pose_info =
      getContext()->getOutputInfo("robot_entry", "pose");
  if (!pose_info.has_value()) {
    throw std::runtime_error("MincoTrajMover: robot_entry::pose not found.");
  }

  int num_robots = pose_info->shape[pose_info->shape.size() - 2];

  // 状态：位置x，位置y，角度z, [[coeff_x], [coeff_y], [coeff_z]]
  addState({"coeff", {num_robots, 3, 6}, NumericalDataType::kFloat32});

  // 输入：目标位置
  addInput({"posT", {num_robots, 3}, 0, ReduceMethod::STACK});
  // 输出：力
  addOutput({"force", {num_robots, 3}, NumericalDataType::kFloat32});
  // 输出：当前bound下的位置
  addOutput({"target_pos_bound",
             {
                 num_robots,
                 2,
             },
             NumericalDataType::kFloat32});
}

void MincoTrajMover::onNodeExecute(const core::NodeExecInputType &input,
                                   core::NodeExecOutputType &,
                                   core::NodeExecStateType &state) {
  float alpha = 1.0f;
  float beta = 0.0f;

  // 获取输入
  const auto &coeff = state.at("coeff");
  const auto &posT = input.at("posT").front();
  // TensorHandle::zeros()
  // const auto &FmX = state.at("FmX").typed_data<float>();
  // const auto &GmU = state.at("GmU").typed_data<float>();
  TensorHandle FmX = coeff.zerosLike();
  TensorHandle GmU = coeff.zerosLike();

  int batch_count = coeff.elemCount() / 18;

  // 计算F*x, 使用gemm运算; F(6x6), x(6xB3); m=6, n=batch_count * 3, k=6
  CUBLAS_CHECK(cublasSgemm_v2(priv_->cublas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
                              6, batch_count * 3, 6, &alpha, priv_->d_matF, 6,
                              coeff.typed_data<float>(), 6, &beta,
                              FmX.typed_data<float>(), 6));
  // 计算G*u, 使用gemm运算; G(6x1), u(1xB3); m=6, n=batch_count * 3, k=1
  CUBLAS_CHECK(cublasSgemm_v2(priv_->cublas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
                              6, batch_count * 3, 1, &alpha, priv_->d_matF, 6,
                              posT.typed_data<float>(), 1, &beta,
                              GmU.typed_data<float>(), 6));

  // 计算V - Ckpt*FmX, [8,B3]=> [2(vel/acc),4(pts),B(batch_size),3(dim)],
  // 使用gemm运算; Ckpt(8x6), FmX(6xB3); m=8, n=batch_count * 3, k=6
  TensorHandle CkFmX = TensorHandle::zeros({2, NUM_CKPTS, batch_count, 3}); //priv_->bound.expand({2, NUM_CKPTS, batch_count, 3});
  alpha = -1.0f;
  // TensorHandle::zeros(CkFmX_shape, NumericalDataType::kFloat32);
  CUBLAS_CHECK(cublasSgemm_v2(priv_->cublas_handle_, CUBLAS_OP_N,
                              CUBLAS_OP_N, 2*NUM_CKPTS, batch_count * 3, 6, &alpha,
                              priv_->d_matL, 2*NUM_CKPTS, FmX.typed_data<float>(), 6,
                              &beta, CkFmX.typed_data<float>(), 8));


  // 计算点除 CmGinv(2,4,1,1) * VmCkFmX(2,4,B,3)
  // auto CmG_view = priv_->matCmGinv.reshape({2, NUM_CKPTS, 1, 1});
  // VmCkFmX *= CmG_view;

}

void MincoTrajMover::onNodeStart() {
  CUBLAS_CHECK(cublasCreate(&priv_->cublas_handle_));
}

void MincoTrajMover::onNodeReset(const core::TensorHandle &reset_flags,
                                 core::NodeExecStateType &state) {
}

void MincoTrajMover::onEnvironGroupInit() {
}

} // namespace minco_traj_mover
} // namespace cuda_simulator
