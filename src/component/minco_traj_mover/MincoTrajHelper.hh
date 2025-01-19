#ifndef CUDASIM_COMPONENT_MINCOTRAJHELPER_HH
#define CUDASIM_COMPONENT_MINCOTRAJHELPER_HH

#include <Eigen/Dense>
#include <Eigen/src/Core/Matrix.h>
#include <cmath>
#include <cuda_runtime_api.h>
#include <iostream>
#include <vector>

namespace cuda_simulator {
namespace minco_traj_mover {

//-----------------------------------------
//  常量定义
//-----------------------------------------
static const int S = 3;                 // jerk控制
static const int POLY_RANK = 2 * S - 1; // 多项式次数, 即5
static const int NCOFF = 2 * S;         // 轨迹系数个数, 即6
static const int NDIM = 2;              // 轨迹维数

//-----------------------------------------
//  使用using定义常见矩阵类型
//-----------------------------------------
using Mat1x1 = Eigen::Matrix<double, 1, 1>;
using Mat1x6 = Eigen::Matrix<double, 1, NCOFF>;
using Mat3x6 = Eigen::Matrix<double, S, NCOFF>;
using Mat6x6 = Eigen::Matrix<double, NCOFF, NCOFF>;
using Vec6 = Eigen::Matrix<double, NCOFF, 1>;
using Mat3x2 = Eigen::Matrix<double, S, NDIM>;
using MatCoff = Eigen::Matrix<double, NCOFF, NDIM>;

//-----------------------------------------
//  一个简单的阶乘函数
//-----------------------------------------
static inline unsigned long factorial(int n) {
  if (n <= 1)
    return 1UL;
  return static_cast<unsigned long>(n) * factorial(n - 1);
}

//-----------------------------------------
//  工具箱类 (全部为 static 函数演示)
//-----------------------------------------
class Toolbox {
public:
  /// 构造特定时间的 β(t)
  static Vec6 constructBetaT(double t, int rank) {
    Vec6 beta = Vec6::Zero();
    for (int i = rank; i < NCOFF; ++i) {
      if (std::fabs(t) > 1e-12 || (i - rank) == 0) {
        double num_factor = static_cast<double>(factorial(i));
        double den_factor = static_cast<double>(factorial(i - rank));
        double power_val = std::pow(t, i - rank);
        beta(i, 0) = num_factor / den_factor * power_val;
      }
    }
    return beta;
  }

  /// 构造矩阵 Ei(2s*2s), Ei[i, :] = β(T, i-1)
  static Mat6x6 constructEi(double trajT) {
    Mat6x6 Ei = Mat6x6::Zero();
    // 第 0 行是 constructBetaT(trajT, 0)
    Ei.row(0) = constructBetaT(trajT, 0).transpose();
    // 第 1~(NCOFF-1) 行依次为 constructBetaT(trajT, i-1)
    for (int i = 1; i < NCOFF; ++i) {
      Ei.row(i) = constructBetaT(trajT, i - 1).transpose();
    }
    return Ei;
  }

  /// 构造矩阵 Fi(2s*2s), Fi[i, :] = -β(0, i-1), 其中 i>=1
  static Mat6x6 constructFi() {
    Mat6x6 Fi = Mat6x6::Zero();
    for (int i = 1; i < NCOFF; ++i) {
      Fi.row(i) = -constructBetaT(0.0, i - 1).transpose();
    }
    return Fi;
  }

  /// 构造F0(s*2s)=[β(0,0), β(0,1), β(0,2) ... β(0,s-1)]
  static Mat3x6 constructF0() {
    Mat3x6 F0 = Mat3x6::Zero();
    for (int i = 0; i < S; ++i) {
      F0.row(i) = constructBetaT(0.0, i).transpose();
    }
    return F0;
  }

  /// 构造E0(s*2s)=[β(T,0), β(T,1), β(T,2) ... β(T,s-1)]
  static Mat3x6 constructEM(double trajT) {
    Mat3x6 E0 = Mat3x6::Zero();
    for (int i = 0; i < S; ++i) {
      E0.row(i) = constructBetaT(trajT, i).transpose();
    }
    return E0;
  }

  /// 构造大矩阵 M, 维度为 (num_pieces*NCOFF) x (num_pieces*NCOFF).
  /// 参考原 Python 里的填充方法。
  static Eigen::MatrixXd constructM(double pieceT, int num_pieces) {
    // 大小: (num_pieces*NCOFF) x (num_pieces*NCOFF)
    Eigen::MatrixXd M = Eigen::MatrixXd::Zero(num_pieces * NCOFF, num_pieces * NCOFF);

    // 最前面 0 ~ S-1 行 M[0:S, 0:NCOFF] = F0
    Mat3x6 F0 = constructF0();
    M.block(0, 0, S, NCOFF) = F0;

    // 最后面行: M[-S:, -NCOFF:] = E0
    Mat3x6 E0 = constructEM(pieceT);
    M.block(num_pieces * NCOFF - S, num_pieces * NCOFF - NCOFF, S, NCOFF) = E0;

    // 中间块
    for (int i = 1; i < num_pieces; ++i) {
      auto EiMat = constructEi(pieceT);
      auto FiMat = constructFi();

      int row_start = (i - 1) * NCOFF + S;
      int col_start_Ei = (i - 1) * NCOFF;
      int col_start_Fi = i * NCOFF;
      M.block(row_start, col_start_Ei, NCOFF, NCOFF) = EiMat;
      M.block(row_start, col_start_Fi, NCOFF, NCOFF) = FiMat;
    }

    return M;
  }

  /// 构造右端路径点约束B矩阵, 维度 (num_pieces*NCOFF) x NDIM
  static Eigen::MatrixXd constructB(const Eigen::MatrixXd &state0, const Eigen::MatrixXd &stateT,
                                    const Eigen::MatrixXd &mid_pos, int num_pieces) {
    Eigen::MatrixXd B = Eigen::MatrixXd::Zero(num_pieces * NCOFF, NDIM);

    // 起点, B[0:S, :] = state0
    B.block(0, 0, S, NDIM) = state0;

    // 终点, B[-S:, :] = stateT
    B.block(num_pieces * NCOFF - S, 0, S, NDIM) = stateT;

    // 中间点
    for (int i = 1; i < num_pieces; ++i) {
      B.block((i - 1) * NCOFF + S, 0, 1, NDIM) = mid_pos.block(i - 1, 0, 1, NDIM);
    }

    return B;
  }

  static Mat6x6 constructBBTint(double pieceT, int rank) {
    Mat6x6 bbint = Mat6x6::Zero();
    Vec6 beta = constructBetaT(pieceT, rank);

    for (int i = 0; i < NCOFF; ++i) {
      for (int j = 0; j < NCOFF; ++j) {
        if ((i + j - 2 * rank) < 0)
          continue;
        double coff = 1.0 / static_cast<double>(i + j - 2 * rank + 1);
        bbint(i, j) = coff * beta(i, 0) * beta(j, 0) * pieceT;
      }
    }
    return bbint;
  }

  static Mat6x6 consturctMatR(double pieceT) {
    Mat6x6 mat_r = Mat6x6::Zero();
    for (int i = 0; i < NCOFF; ++i) {
      mat_r.row(i) = constructBetaT(pieceT, i).transpose();
    }
    return mat_r;
  }

  /// 构造单个piece的CKPT检查矩阵 (NCOFF x num_ckpt)
  static Eigen::MatrixXd constructCkptMat(double pieceT, int num_ckpt, int rank) {
    Eigen::MatrixXd ckpt_mat = Eigen::MatrixXd::Zero(NCOFF, num_ckpt);

    // 计算若干插值时刻
    std::vector<double> ckpt_ts(num_ckpt, 0.0);
    for (int i = 0; i < num_ckpt; ++i) {
      double frac = double(i + 1) / double(num_ckpt + 1);
      ckpt_ts[i] = frac * pieceT;
    }

    // 循环填充
    for (int i = 0; i < num_ckpt; ++i) {
      auto b = constructBetaT(ckpt_ts[i], rank);
      // 将 b 存到第 i 列
      for (int r = 0; r < NCOFF; ++r) {
        ckpt_mat(r, i) = b(r, 0);
      }
    }

    return ckpt_mat;
  }

  /// 构造整条轨迹的CKPT检查矩阵 (npiece*NCOFF) x (npiece*nckpt)
  /// 每段插值后在对应的块写入
  static Eigen::MatrixXd constructPiecesCkptMat(double pieceT, int rank, int nckpt, int npiece) {
    Eigen::MatrixXd ckpt_mat = Eigen::MatrixXd::Zero(npiece * NCOFF, npiece * nckpt);

    for (int i = 0; i < npiece; ++i) {
      auto single_ckpt = constructCkptMat(pieceT, nckpt, rank);
      // 写入块 [i*NCOFF : (i+1)*NCOFF, i*nckpt : (i+1)*nckpt]
      ckpt_mat.block(i * NCOFF, i * nckpt, NCOFF, nckpt) = single_ckpt;
    }
    return ckpt_mat;
  }

  static MatCoff constructInitCoeff(const Eigen::MatrixXd &init_pos) {
    MatCoff coeff = MatCoff::Zero();

    int a = init_pos.rows();
    for (int i = 0; i < a && i < NCOFF; ++i) {
      coeff(i, 0) = init_pos(i, 0);
      coeff(i, 1) = init_pos(i, 1);
    }
    return coeff;
  }
};

class MincoToolbox : public Toolbox {
public:
  /// 5阶连续
  static Mat6x6 constructMincoM(double pieceT) {
    Mat6x6 mat_m = Mat6x6::Zero();
    for (int i = 0; i < NCOFF - 1; ++i) {
      mat_m.row(i) = constructBetaT(0.0, i).transpose();
    }
    mat_m.row(NCOFF - 1) = constructBetaT(pieceT, 0).transpose();
    return mat_m;
  }

  /// 4阶连续
  static Mat6x6 constructMincoM2(double pieceT) {
    Mat6x6 mat_m = Mat6x6::Zero();
    for (int i = 0; i < NCOFF - 2; ++i) {
      mat_m.row(i) = constructBetaT(0.0, i).transpose();
    }

    // 固定位置和速度 (最后两行)
    mat_m.row(NCOFF - 2) = constructBetaT(pieceT, 0).transpose(); // pos
    mat_m.row(NCOFF - 1) = constructBetaT(pieceT, 1).transpose(); // vel

    // 求逆
    Mat6x6 mat_m_inv = mat_m.inverse();

    // 构造 ∫ββ^T, rank=S
    auto bb_int = constructBBTint(pieceT, S);

    // mat_supp = [[0,0,0,0,0,1]] * mat_m_inv * bb_int
    Eigen::RowVectorXd row_vec(NCOFF);
    row_vec.setZero();
    row_vec(NCOFF - 1) = 1.0; // 最后一列为1

    // mat_supp = row_vec * mat_m_inv * bb_int  => (1 x NCOFF)
    Eigen::RowVectorXd mat_supp = row_vec * mat_m_inv * bb_int;

    // mat_m[-1, :] = mat_supp[-1, :]
    for (int j = 0; j < NCOFF; ++j) {
      mat_m(NCOFF - 1, j) = mat_supp(0, j);
    }

    return mat_m;
  }

  static MatCoff constructMincoQ(const MatCoff &last_coff, const Eigen::Vector2d &tgtPos, double pieceT) {
    Mat6x6 mat_r = consturctMatR(pieceT);
    // mat_q = mat_r * last_coff  (维度: (NCOFF,NCOFF) * (NCOFF,NDIM) = (NCOFF, NDIM))
    MatCoff mat_q = mat_r * last_coff;

    // mat_q[-1, :] = tgtPos
    mat_q.row(NCOFF - 1) = tgtPos.transpose();
    return mat_q;
  }
};

class MincoTrajSystem {
  double pieceT;
  double realT;

  // 矩阵
  Mat6x6 mat_F; // (6x6)
  Vec6 mat_G; // (6x1) or (NCOFFx1)
  Mat6x6 mat_F_stab;
  Vec6 mat_G_stab;
  Mat1x6 K; // LQR增益矩阵

  // 求解离散Riccati方程的LQR
  Mat1x6 solveDLQR(const Mat6x6 &A, const Vec6 &B, const Mat6x6 &Q, const Mat1x1 &R) {
    // 这里使用迭代法求解离散Riccati方程
    Mat6x6 P = Q; // 初始猜测
    Mat6x6 P_next;
    const double epsilon = 1e-7;
    const int max_iter = 1000;

    for (int i = 0; i < max_iter; i++) {
      // P_next = Q + A^T * P * A - A^T * P * B * (R + B^T * P * B)^-1 * B^T * P * A
      Mat1x1 temp = R + B.transpose() * P * B;
      Mat1x6 K_temp = (temp.inverse() * B.transpose() * P * A);
      P_next = Q + A.transpose() * P * A - A.transpose() * P * B * K_temp;

      if ((P_next - P).norm() < epsilon) {
        P = P_next;
        break;
      }
      P = P_next;
    }

    // 计算LQR增益 K = (R + B^T * P * B)^-1 * B^T * P * A
    Mat1x1 temp = R + B.transpose() * P * B;
    return (temp.inverse() * B.transpose() * P * A);
  }

public:
  MincoTrajSystem(double execT=0.1, double RATIO=0.2) {
    pieceT = execT / RATIO;
    realT = execT;

    // 1. construct M2
    Mat6x6 mat_m = MincoToolbox::constructMincoM2(pieceT);
    Mat6x6 mat_m_inv = mat_m.inverse();

    // 2. mat_r
    Mat6x6 mat_r = Toolbox::consturctMatR(realT);

    // 3. mat_s = diag([1,1,1,1,0,0]) => (6x6)
    Mat6x6 mat_s = Mat6x6::Zero();
    for (int i = 0; i < 4; ++i) {
      mat_s(i, i) = 1.0;
    }

    // 4. mat_u = [[0,0,0,0,1,0]]^T => (6x1)
    Vec6 mat_u;
    mat_u.setZero();
    mat_u(4, 0) = 1.0;

    // 计算F和G矩阵
    // mat_F = mat_m_inv @ mat_s @ mat_r
    mat_F = mat_m_inv * mat_s * mat_r;
    // mat_G = (mat_m_inv @ mat_u)
    mat_G = mat_m_inv * mat_u;

    // LQR，状态权重矩阵
    Mat6x6 Q = Toolbox::constructBBTint(pieceT, S);
    // LQR，控制权重矩阵
    Mat1x1 R;
    R << 10.0;

    // 求解LQR
    K = solveDLQR(mat_F, mat_G, Q, R);

    // 计算Kpp
    Mat1x6 Kpp = (mat_G.transpose() * mat_G).inverse() * mat_G.transpose() * (Mat6x6::Identity() - mat_F) + K;

    // 计算闭环系统矩阵
    mat_F_stab = mat_F - mat_G * K;

    // 构造稳定化的输入矩阵
    Vec6 selector = Vec6::Zero();
    selector(0) = 1.0;
    mat_G_stab = mat_G * Kpp * selector;

    // std::cout << "mat_F: " << std::endl << mat_F << std::endl;
    // std::cout << "mat_G: " << std::endl << mat_G << std::endl;
    // printf("K:[[");
    // for(int i=0;i<6;i++) {
    //   printf("%f, ", K(0, i));
    // }
    // printf("]]\n");
    // // std::cout << "K: " << std::endl << K << std::endl;
    // std::cout << "Kpp: " << std::endl << Kpp << std::endl;
    // std::cout << "mat_F_stab: " << std::endl << mat_F_stab << std::endl;
    // std::cout << "mat_G_stab: " << std::endl << mat_G_stab << std::endl;
  }

  template<typename T>
  void getMatF(T* output) {
    // output.resize(6 * 6);
    for (int i = 0; i < 6; ++i) {
      for (int j = 0; j < 6; ++j) {
        output[i * 6 + j] = mat_F_stab(i, j);
      }
    }
  }

  template<typename T>
  void getMatG(T* output) {
    // output.resize(6);
    for (int i = 0; i < 6; ++i) {
      output[i] = mat_G_stab(i, 0);
    }
  }

  template<typename T>
  void getMatCkpt(int num_ckpts, T* output) {

    Eigen::MatrixXd ckpt_mat = Eigen::MatrixXd::Zero(num_ckpts * 2, NCOFF);
    ckpt_mat.block(0, 0, num_ckpts, NCOFF) = MincoToolbox::constructCkptMat(pieceT, num_ckpts, 1);
    ckpt_mat.block(num_ckpts, 0, num_ckpts, NCOFF) = MincoToolbox::constructCkptMat(pieceT, num_ckpts, 2);

    // output.resize(num_ckpts * 2 * NCOFF);
    for (int i = 0; i < num_ckpts * 2; ++i) {
      for (int j = 0; j < NCOFF; ++j) {
        output[i * NCOFF + j] = ckpt_mat(i, j);
      }
    }
  }
};

} // namespace minco_traj_mover
} // namespace cuda_simulator

#endif