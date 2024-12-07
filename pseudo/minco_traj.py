# import numpy as np
# from numpy.typing import NDArray
# import jax.numpy as np
import numpy as np
from numpy import ndarray as Array
# from jax import Array
from typing import Tuple, Optional
from functools import lru_cache

class Poly_Traj:
    N = 2           # 维度
    S = 3           # Control Effort 次数，2->Accelerate, 3->Jerk
    M = S * 2 - 1   # 轨迹多项式次数
    def __init__(self, period:float, coff:Array|None = None) -> None:
        if coff is None:
            self.coff = np.zeros((Poly_Traj.M+1, Poly_Traj.N), dtype=np.float32)
        else:
            self.coff = coff
        self.period = period

    @staticmethod
    @lru_cache
    def construct_beta(t:float, rank:int) -> Array:
        '''
        构造beta矩阵:
        | 1 t t^2 t^3 ... |^T
        | 1 t t^2 t^3 ... |
        N*M维度
        '''
        beta = np.zeros(Poly_Traj.M+1, dtype=np.float32)
        beta_coff = np.zeros(Poly_Traj.M+1, dtype=np.float32)

        # 计算T有关的部分
        beta[rank] = 1
        for i in range(rank+1, Poly_Traj.M+1):
            beta[i] = beta[i-1] * t
        # 计算多项式系数
        for i in range(rank, Poly_Traj.M+1):
            coff = 1
            for j in range(0, rank):
                coff *= i-j
            beta_coff[i] = coff

        # return beta * beta_coff
        beta = beta * beta_coff
        return beta[:,None]


    @staticmethod
    def construct_M(ts:Array) -> Array:
        if ts.shape[0] != 2:
            raise Exception
        beta_s = [Poly_Traj.construct_beta(t.item(), s).transpose() for t in ts for s in range(Poly_Traj.S) ]
        M = np.vstack(beta_s)
        return M

    @staticmethod
    def init_by_qT(q0:Array, qT:Array, ts:Array):
        '''
        q0, qT: [
            pos^T: [Dim x, Dim y, Dim z],
            vel^T: [Dim x, Dim y, Dim z],
            acc^T: [Dim x, Dim y, Dim z],
            ...]
        ts: [Point 0 time, Point 1 time]
        '''

        if q0.shape[0] != Poly_Traj.S or qT.shape[0] != Poly_Traj.S:
            raise Exception
        if q0.shape[1] != Poly_Traj.N or qT.shape[1] != Poly_Traj.N:
            raise Exception

        M = Poly_Traj.construct_M(ts)
        M_inv = np.linalg.inv(M)

        q = np.vstack([q0, qT])
        c = M_inv @ q
        return Poly_Traj(ts[1].item(), c)

    def get_derivative(self, t:float, rank:int) -> Array:
        beta = Poly_Traj.construct_beta(t = t, rank = rank)
        return (beta.T @ self.coff)[0]

    def get_pos(self, t:float) -> Array:
        return self.get_derivative(t=t, rank=0)

    def get_vel(self, t:float) -> Array:
        return self.get_derivative(t=t, rank=1)

    def get_acc(self, t:float) -> Array:
        return self.get_derivative(t=t, rank=2)

    def get_curvatures(self, t:float) -> float:
        vel = self.get_vel(t)
        acc = self.get_acc(t)
        epsilon = 1e-6
        denom = (vel[0]**2 + vel[1]**2)**1.5
        if denom <= epsilon:
            return 0
        else:
            return ((vel[0] * acc[1] - vel[1] * acc[0]) / denom).item()


    def get_jerk(self, t:float) -> Array:
        beta = Poly_Traj.construct_beta(t = t, rank = 3)
        # print(beta)
        return beta.T @ self.coff

    def get_end_pos(self) -> Array:
        return self.get_pos(self.period)

    def _compute_trajectory(self, ts:Array):

        positions = []
        velocities = []
        accelerations = []
        curvatures = []

        for t in ts:
            positions.append(self.get_pos(t))
            velocities.append(self.get_vel(t))
            accelerations.append(self.get_acc(t))
            curvatures.append(self.get_curvatures(t))

        # 转换为数组
        positions = np.array(positions)
        velocities = np.array(velocities)
        accelerations = np.array(accelerations)
        curvatures = np.array(curvatures)

        return positions, velocities, accelerations, curvatures


class MincoPolyTrajFactory:
    def __init__(self, ts:Array, dof:int):
        self.pJpC = np.zeros((Poly_Traj.M+1, Poly_Traj.M+1), dtype=np.float32)
        self.boundCoffA = None
        self.ts = ts
        self.dof = dof
        self.Minv : Optional[Array] = None

    def clearPJpC(self):
        self.pJpC = np.zeros((Poly_Traj.M+1, Poly_Traj.M+1), dtype=np.float32)

    def calcMatrixMInv(self):
        '''
        获得当前用于求解多项式系数的M^-1矩阵
        '''

        M = Poly_Traj.construct_M(self.ts)
        M_inv = np.linalg.inv(M)

        pQptQ = np.zeros_like(M)
        pQptQ[-1-self.dof+1:,-1-self.dof+1:] = np.eye(self.dof)
        pCpQ = M_inv.T
        # pJpC 是代价函数对多项式系数的偏导数（∂J/∂c），由其他函数累计得到
        pJptQ = pQptQ @ pCpQ @ self.pJpC

        newM = M.copy()
        newM[-1-self.dof+1:,:] = pJptQ[-1-self.dof+1:,:]
        newM_inv = np.linalg.inv(newM)

        self.Minv = np.array(newM_inv)
        # return newM_inv

    def solveWithCostJ(self, q0:Array, qT:Array) -> "Poly_Traj":
        '''
        q0: [
            pos^T: [Dim x, Dim y, Dim z],
            vel^T: [Dim x, Dim y, Dim z],
            acc^T: [Dim x, Dim y, Dim z],
            ...]

        qT: [
            pos^T: [Dim x, Dim y, Dim z],
            vel^T: [Dim x, Dim y, Dim z],
            比q0缺少一维
            ...]

        ts: [Point 0 time, Point 1 time]

        dof: 自由度
        '''
        assert self.Minv is not None

        if q0.shape[0] != Poly_Traj.S or qT.shape[0] != Poly_Traj.S-self.dof:
            raise Exception
        if q0.shape[1] != Poly_Traj.N or qT.shape[1] != Poly_Traj.N:
            raise Exception

        qT = np.concatenate((qT, np.zeros((self.dof, Poly_Traj.N), dtype=np.float32)), axis=0)
        q = np.vstack([q0, qT])
        c = self.Minv @ q

        return Poly_Traj(self.ts[1].item(), c)

    def getPSigmaPQ(self, t:float, rank:int):

        assert self.Minv is not None

        beta = Poly_Traj.construct_beta(t=t, rank=rank)
        S = np.zeros(Poly_Traj.M+1)
        S[Poly_Traj.S:2*Poly_Traj.S-self.dof] = 1

        jac = S @ self.Minv.T @ beta

        return jac


    def solveEndPosEqu(self, t:float, rank:int, q0:Array, maxMinEquRVal:float, velT:Optional[Array]) -> Tuple[Array, Array]:
        '''
        求解末端位置的约束等式。
        返回值:
        Tuple[pos_u, pos_d]
            pos <= pos_u: 代表位置的上界
            pos >= pos_d: 代表位置的下界
        '''
        # 求解不等式 (Phi'(t)^T) * (M^-1) * q （<= v_max 或 a_max，取第二行或第三行）或（>= -v_max 或 -a_max）
        # c0*p0+c1*v0+c2*a0+c3*pT+c4*vT <= v_max    (Minco求解时，末端aT是线性相关量)
        # aT 是线性相关（可被其他表出）
        # (v_max - c0*p0+c1*v0+c2*a0+c4*vT) / c3

        # if self.boundCoffA is None:

        # 计算(Phi'(t)^T) * (M^-1)
        assert self.Minv is not None

        phi_rank = Poly_Traj.construct_beta(t=t, rank=rank)
        self.boundCoffA = phi_rank.T @ self.Minv

        # 构造系数向量[p0, v0, a0, 0, vT, 0]
        q = np.vstack([q0, np.zeros((Poly_Traj.S, Poly_Traj.N))])
        # 末端速度受约束
        if self.dof==1:
            assert isinstance(velT, np.ndarray)
            q[Poly_Traj.S+1:Poly_Traj.S+2,:] = velT

        # [c0, c1, c2, c3, c4, c5]，取出pT对应的系数, 这里可能会变号，当系数为负时，会使不等式改变方向
        posT_coff_in_equ = self.boundCoffA[Poly_Traj.S:2*Poly_Traj.S-2]
        if posT_coff_in_equ < 0:
            maxMinEquRVal = - maxMinEquRVal

        # DOF==2时不用设置vT, 对应位置为空
        const_val_max =  + maxMinEquRVal - self.boundCoffA@q
        const_val_min =  - maxMinEquRVal - self.boundCoffA@q

        pos_u = const_val_max / posT_coff_in_equ
        pos_d = const_val_min / posT_coff_in_equ

        return pos_u, pos_d

    def solveEndPosBound(self, ts:Array, dof:int, q0:Array, velT:Optional[Array], rank:int, maxVel:float) \
            -> Tuple[Array, Array]:
        '''
        求解最大速度/加速度区间
        '''
        assert rank==1 or rank==2
        # 区间划分的段数
        seg_count = 5
        den = ts[1] / (2*seg_count)
        bound_u_results = np.zeros((seg_count,Poly_Traj.N))
        bound_d_results = np.zeros((seg_count,Poly_Traj.N))
        for i in range(seg_count):
            t = (2*i+1) * den
            pos_u, pos_d = self.solveEndPosEqu(t=t.item(), rank=rank, q0=q0, velT=velT, maxMinEquRVal=maxVel)
            bound_u_results[i, :] = pos_u
            bound_d_results[i, :] = pos_d
        return np.min(bound_u_results, axis=0), np.max(bound_d_results, axis=0)

    @staticmethod
    def betabetaT_int(t:float, rank:int) -> Array:
        b = Poly_Traj.construct_beta(t=t, rank=rank).reshape(-1,1)
        bbT = b@b.T
        int_coff = np.array([[1/(i+j-2*rank+1) if i+j-2*rank+1 > 0 else 0 for j in range(Poly_Traj.M+1)] for i in range(Poly_Traj.M+1)])
        return t * bbT * int_coff

    def add_control_effort_cost(self, T:float):
        # pJpC = ∂J/∂c = ∫(ββ^T)dt * c
        # 构造末端时间的代价矩阵∫(ββ^T)dt，维度(M+1)*(M+1)
        bbT_int = MincoPolyTrajFactory.betabetaT_int(T, Poly_Traj.S)
        self.pJpC += bbT_int
        self.Minv = None

    def add_acc_cost(self, T:float):
        # pJpC = ∂J/∂c = ∫(ββ^T)dt * c
        # 构造末端时间的代价矩阵∫(ββ^T)dt，维度(M+1)*(M+1)
        bbT_int = MincoPolyTrajFactory.betabetaT_int(T, Poly_Traj.S-1)
        self.pJpC += bbT_int
        self.Minv = None

    @staticmethod
    def betabetaT(t:float, rank:int) -> Array:
        b = Poly_Traj.construct_beta(t=t, rank=rank).reshape(-1,1)
        bbT = b@b.T
        return bbT

    def add_max_vel_cost(self, weight:float, traj:Poly_Traj, T:float, max_vel:float, cps_num:int):
        # pJpC = ∂J/∂c = ∂(v*v)/∂c = β*β^T * c
        # 构造末端时间的代价矩阵∫(ββ^T)dt，维度(M+1)*(M+1)
        for t in np.linspace(0, T, cps_num):
            if abs(traj.get_vel(t)[0]) > max_vel:
                # print("add cost!")
                self.pJpC += weight * MincoPolyTrajFactory.betabetaT(t, 1)

    def add_max_acc_cost(self, weight:float, traj:Poly_Traj, T:float, max_acc:float, cps_num:int):
        # pJpC = ∂J/∂c = ∂(v*v)/∂c = β*β^T * c
        # 构造末端时间的代价矩阵∫(ββ^T)dt，维度(M+1)*(M+1)
        for t in np.linspace(0, T, cps_num):
            if abs(traj.get_acc(t)[0]) > max_acc:
                # print("add cost!")
                self.pJpC += weight * MincoPolyTrajFactory.betabetaT(t, 2)
