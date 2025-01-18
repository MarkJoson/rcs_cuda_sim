#ifndef __MINCO_TRAJ_FACTORY_H__
#define __MINCO_TRAJ_FACTORY_H__

#pragma once

#include <Eigen/Dense>
#include <vector>

namespace RSG_SIM
{

constexpr int POLY_DIM = 2;                             // 维度
constexpr int POLY_CTRL_EFFORT = 3;                     // Control Effort 次数，2->Accelerate, 3->Jerk
constexpr int POLY_DEGREE = POLY_CTRL_EFFORT * 2 - 1;   // 轨迹多项式次数

typedef Eigen::Matrix<double, POLY_DEGREE+1, POLY_DIM>       CoffMat6x2d;
typedef Eigen::Matrix<double, POLY_CTRL_EFFORT*2, POLY_DIM>  QMat6x2d;
typedef Eigen::Vector<double, POLY_DEGREE+1>                 CoffVec6d;
typedef Eigen::Vector<float, POLY_DEGREE+1>                 CoffVec6f;
typedef Eigen::Vector<double, 2>                             TimeVec2d;
typedef Eigen::Matrix<double, POLY_DEGREE+1, POLY_DEGREE+1>  MMat6x6d;
typedef Eigen::Matrix<float, POLY_DEGREE+1, POLY_DEGREE+1>  MMat6x6f;
typedef Eigen::Matrix<double, POLY_DEGREE+1, POLY_DEGREE+1, Eigen::RowMajor>  MMat6x6dRowMajor;
typedef Eigen::Matrix<float, POLY_DEGREE+1, POLY_DEGREE+1, Eigen::RowMajor>  MMat6x6fRowMajor;

typedef Eigen::Vector<double, POLY_DIM>                      StateVec2;
typedef Eigen::Matrix<double, POLY_CTRL_EFFORT, POLY_DIM>    StateMat3x2;

class PolyTraj {
public:

    PolyTraj(CoffMat6x2d coff_ = CoffMat6x2d::Zero()) : coff(coff_) { }

    static CoffVec6d construct_beta(double t, int rank)  {
    CoffVec6d betaT = CoffVec6d::Zero();
    CoffVec6d beta_coff = CoffVec6d::Zero();

    // beta:
    // [1 T T^2 T^3 ...] (rank=0)
    // [0 1 2*T 3*T^2 4*T^3 ...] (rank=1)
    // [0 0 2   2*3*T 3*4*T^2 ...] (rank=2)

    // 计算T有关的部分 [1 T T^2 T^3 ...]
    betaT(rank) = 1;
    for (int i = rank+1; i <= POLY_DEGREE; i++) {
        betaT(i) = betaT(i-1) * t;
    }
    // 计算多项式系数 [1 1 1 1 1 1](rank=0)     [0 1 2 3 4 ...](rank=1)     [0 0 2 2*3 3*4 ...](rank=2)
    for (int i = rank; i <= POLY_DEGREE; i++) {
        double coff = 1;
        for (int j = 0; j < rank; j++) {
            coff *= i-j;
        }
        beta_coff(i) = coff;
    }

    return betaT.cwiseProduct(beta_coff);
}

    static MMat6x6d construct_M(TimeVec2d ts) {
    MMat6x6d M;
    int row_idx = 0;
    for (int i = 0; i < ts.size(); i++) {
        for (int s = 0; s < POLY_CTRL_EFFORT; s++) {
            M.row(row_idx) = construct_beta(ts(i), s).transpose();
            row_idx ++;
        }
    }
    return M;
}

    static PolyTraj init_by_qT(QMat6x2d q0, QMat6x2d qT, TimeVec2d ts) {

    MMat6x6d M = construct_M(ts);
    MMat6x6d M_inv = M.inverse();

    QMat6x2d q;
    q << q0, qT;
    CoffMat6x2d c = M_inv * q;
    return PolyTraj(c);
}

    StateVec2 get_pos(double t) {
        CoffVec6d beta = construct_beta(t, 0);
        return beta.transpose() * coff;
    }

    StateVec2 get_vel(double t) {
        CoffVec6d beta = construct_beta(t, 1);
        return beta.transpose() * coff;
    }

    StateVec2 get_acc(double t) {
        CoffVec6d beta = construct_beta(t, 2);
        return beta.transpose() * coff;
    }

    CoffMat6x2d get_poly_coff() {
        return coff;
    }

private:
    CoffMat6x2d coff;
};

class MincoTrajFactory {
public:
    MincoTrajFactory() : pJpC_(MMat6x6d::Zero()) {}

    void clearPJpC() {
        pJpC_ = MMat6x6d::Zero();
    }

    MMat6x6d getMatrixMInv(TimeVec2d ts, int dof);

    PolyTraj solveWithCostJ(StateMat3x2 q0, StateMat3x2 qT, TimeVec2d ts, int dof) {
    // 如果Minv已经有缓存，就用缓存过的
    // if (Minv_.size() != 0) {
    //     return Minv_;
    // }

    MMat6x6d M = PolyTraj::construct_M(ts);
    MMat6x6d M_inv = M.inverse();

    // ~q是约束中的自由变量，如当不限制速度和加速度时 ~q=[0,0,0,0,vT,aT]
    // 对末端a（即[0,0,0,0,0,a]^T向量）求偏导:∂J/∂(~q) = ∂q^T/∂(~q) * ∂c^T/∂q * ∂J/∂c
    // pQptQ = ∂q^T/∂(~q) = diag([0,0,0, ... ,1])
    // pCpQ = ∂c^T/∂(q) = M^(-T)
    // pJpC = ∂J/∂c = ∫(ββ^T)dt * c + ...
    // pQptQ 是q对tilt q的偏导数

    MMat6x6d pQptQ = MMat6x6d::Zero();
    pQptQ.bottomRightCorner(dof, dof) = Eigen::MatrixXd::Identity(dof, dof);

    MMat6x6d pCpQ = M_inv.transpose();
    MMat6x6d pJptQ = pQptQ * pCpQ * pJpC_;

    MMat6x6d newM = M;
    newM.bottomRows(dof) = pJptQ.bottomRows(dof);
    // std::cout << FG_YELLOW "pQptQ:\n" FG_DEFAULT << pQptQ << std::endl << std::endl;
    // std::cout << FG_YELLOW "pCpQ:\n" FG_DEFAULT << pCpQ << std::endl << std::endl;
    // std::cout << FG_YELLOW "pJpC_:\n" FG_DEFAULT << pJpC_ << std::endl << std::endl;

    MMat6x6d newM_inv = newM.inverse();

    return newM_inv;
}

    std::pair<StateVec2, StateVec2> solveEndPosEqu(
        TimeVec2d ts,
        int dof,
        double t,
        int rank,
        StateMat3x2 q0,
        StateVec2 velT,
        double max_min_equ_vel);

    std::pair<StateVec2, StateVec2> solveEndPosBound(TimeVec2d ts, StateMat3x2 q0, StateVec2 velT, int rank, double maxVel);

    /***
     * @brief 计算得到∫ββ^T
     */
    static MMat6x6d betabetaT_int(double t, int rank);
    static MMat6x6d betabetaT(double t, int rank);

    void add_control_effort_cost(double T);
    void add_acc_cost(double T);

private:
    MMat6x6d pJpC_;
    // MMat6x6d Minv_;
};

} // namespace RSG_SIM

#endif