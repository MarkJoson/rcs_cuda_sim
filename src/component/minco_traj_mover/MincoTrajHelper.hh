#ifndef CUDASIM_COMPONENT_MINCOTRAJHELPER_HH
#define CUDASIM_COMPONENT_MINCOTRAJHELPER_HH

#include <Eigen/Dense>
#include <cmath>

namespace cuda_simulator {
namespace minco_traj_mover {


// Constants
constexpr int S = 3;          // jerk control
constexpr int POLY_RANK = 2*S-1;  // polynomial degree
constexpr int NCOFF = 2*S;    // trajectory coefficients count
constexpr int NDIM = 2;       // trajectory dimensions

// Matrix type definitions
using MatrixX = Eigen::MatrixXd;
using VectorX = Eigen::VectorXd;
using Matrix6 = Eigen::Matrix<double, NCOFF, NCOFF>;
using Vector6 = Eigen::Matrix<double, NCOFF, NDIM>;
using Matrix3_6 = Eigen::Matrix<double, S, NCOFF>;
using Matrix2 = Eigen::Matrix<double, NDIM, NDIM>;
using Vector2 = Eigen::Matrix<double, NDIM, NDIM>;


class TrajToolbox {
public:
    static Vector6 constructBetaT(double t, int rank) {
        Vector6 beta = Vector6::Zero();
        for (int i = rank; i < NCOFF; i++) {
            if (t != 0 || i-rank == 0) {
                beta(i) = factorial(i)/factorial(i-rank) * std::pow(t, i-rank);
            }
        }
        return beta;
    }

    static Matrix6 constructEi(double trajT) {
        Matrix6 Ei = Matrix6::Zero();
        Ei.row(0) = constructBetaT(trajT, 0).transpose();
        for (int i = 1; i < NCOFF; i++) {
            Ei.row(i) = constructBetaT(trajT, i-1).transpose();
        }
        return Ei;
    }

    static Matrix6 constructFi() {
        Matrix6 Fi = Matrix6::Zero();
        for (int i = 1; i < NCOFF; i++) {
            Fi.row(i) = -constructBetaT(0, i-1).transpose();
        }
        return Fi;
    }

    static Matrix3_6 constructF0() {
        Matrix3_6 F0 = Matrix3_6::Zero();
        for (int i = 0; i < S; i++) {
            F0.row(i) = constructBetaT(0, i).transpose();
        }
        return F0;
    }

    static Matrix3_6 constructEM(double trajT) {
        Matrix3_6 E0 = Matrix3_6::Zero();
        for (int i = 0; i < S; i++) {
            E0.row(i) = constructBetaT(trajT, i).transpose();
        }
        return E0;
    }

    static MatrixX constructM(double pieceT, int num_pieces) {
        MatrixX M = MatrixX::Zero(num_pieces*NCOFF, num_pieces*NCOFF);

        M.block(0, 0, S, NCOFF) = constructF0();
        M.block(M.rows()-S, M.cols()-NCOFF, S, NCOFF) = constructEM(pieceT);

        for (int i = 1; i < num_pieces; i++) {
            M.block((i-1)*NCOFF+S, (i-1)*NCOFF, NCOFF, NCOFF) = constructEi(pieceT);
            M.block((i-1)*NCOFF+S, i*NCOFF, NCOFF, NCOFF) = constructFi();
        }
        return M;
    }

    static MatrixX constructB(const MatrixX& state0, const MatrixX& stateT,
                            const MatrixX& mid_pos, int num_pieces) {
        MatrixX B = MatrixX::Zero(num_pieces*NCOFF, NDIM);
        B.block(0, 0, S, NDIM) = state0;
        B.block(B.rows()-S, 0, S, NDIM) = stateT;

        for (int i = 1; i < num_pieces; i++) {
            B.block((i-1)*NCOFF+S, 0, 1, NDIM) = mid_pos.row(i-1);
        }
        return B;
    }

    static Matrix6 constructBBTint(double pieceT, int rank) {
        Matrix6 bbint = Matrix6::Zero();
        Vector6 beta = constructBetaT(pieceT, rank);

        for (int i = 0; i < NCOFF; i++) {
            for (int j = 0; j < NCOFF; j++) {
                if (i+j-2*rank < 0) continue;
                double coff = 1.0 / (i+j-2*rank+1);
                bbint(i,j) = coff * beta(i) * beta(j) * pieceT;
            }
        }
        return bbint;
    }

    static Matrix6 consturctMatR(double pieceT) {
        Matrix6 mat_r = Matrix6::Zero();
        for (int i = 0; i < NCOFF; i++) {
            mat_r.row(i) = constructBetaT(pieceT, i).transpose();
        }
        return mat_r;
    }

    static MatrixX constructCkptMat(double pieceT, int num_ckpt, int rank) {
        MatrixX ckpt_mat = MatrixX::Zero(NCOFF, num_ckpt);

        for (int i = 0; i < num_ckpt; i++) {
            double frac = static_cast<double>(i+1)/(num_ckpt+1);
            double ckpt_t = frac * pieceT;
            ckpt_mat.col(i) = constructBetaT(ckpt_t, rank);
        }
        return ckpt_mat;
    }

    static MatrixX constructPiecesCkptMat(double pieceT, int rank, int nckpt, int npiece) {
        MatrixX ckpt_mat = MatrixX::Zero(npiece*NCOFF, npiece*nckpt);
        for (int i = 0; i < npiece; i++) {
            ckpt_mat.block(i*NCOFF, i*nckpt, NCOFF, nckpt) =
                constructCkptMat(pieceT, nckpt, rank);
        }
        return ckpt_mat;
    }

private:
    static int factorial(int n) {
        if (n <= 1) return 1;
        return n * factorial(n-1);
    }
};

} // namespace minco_traj_mover
} // namespace cuda_simulator



#endif