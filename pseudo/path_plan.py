import numpy as np
from scipy import optimize
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
import matplotlib.pyplot as plt

import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Optional

class Poly_Traj:
    N = 2           # 维度
    S = 3           # Control Effort 次数，2->Accelerate, 3->Jerk
    M = S * 2 - 1   # 轨迹多项式次数
    def __init__(self, period:float, coff:NDArray[np.float32]|None = None) -> None:
        if coff is None:
            self.coff = np.zeros((Poly_Traj.M+1, Poly_Traj.N), dtype=np.float32)
        else:
            self.coff = coff
        self.period = period

    @staticmethod
    def construct_beta(t:float, rank:int) -> NDArray[np.float32]:
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

        return beta * beta_coff

    @staticmethod
    def construct_M(ts:NDArray[np.float32]) -> NDArray[np.float32]:
        if ts.shape[0] != 2:
            raise Exception
        beta_s = [Poly_Traj.construct_beta(t, s).transpose() for t in ts for s in range(Poly_Traj.S) ]
        M = np.vstack(beta_s)
        return M

    @staticmethod
    def init_by_qT(q0:NDArray[np.float32], qT:NDArray[np.float32], ts:NDArray[np.float32]):
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
        return Poly_Traj(ts[1], c)

    def get_pos(self, t:float) -> NDArray[np.float32]:
        beta = Poly_Traj.construct_beta(t = t, rank = 0)
        return beta.T @ self.coff

    def get_vel(self, t:float) -> NDArray[np.float32]:
        beta = Poly_Traj.construct_beta(t = t, rank = 1)
        # print(f"****Requrie Traj @ Time {t}")
        return beta.T @ self.coff

    def get_acc(self, t:float) -> NDArray[np.float32]:
        beta = Poly_Traj.construct_beta(t = t, rank = 2)
        # print(beta)
        return beta.T @ self.coff

    def get_curvatures(self, t:float) -> float:
        vel = self.get_vel(t)
        acc = self.get_acc(t)
        epsilon = 1e-6
        denom = (vel[0]**2 + vel[1]**2)**1.5 + epsilon
        kappa = (vel[0] * acc[1] - vel[1] * acc[0]) / denom

        return kappa

    def get_jerk(self, t:float) -> NDArray[np.float32]:
        beta = Poly_Traj.construct_beta(t = t, rank = 3)
        # print(beta)
        return beta.T @ self.coff

    def get_end_pos(self) -> NDArray[np.float32]:
        return self.get_pos(self.period)


class MincoPolyTrajFactory:
    def __init__(self):
        self.pJpC = np.zeros((Poly_Traj.M+1, Poly_Traj.M+1), dtype=np.float32)
        self.Minv = None
        self.boundCoffA = None

    def clearPJpC(self):
        self.pJpC = np.zeros((Poly_Traj.M+1, Poly_Traj.M+1), dtype=np.float32)

    def solveWithCostJ(self, q0:NDArray[np.float32], qT:NDArray[np.float32], ts:NDArray[np.float32], dof:int) -> "Poly_Traj":
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

        if q0.shape[0] != Poly_Traj.S or qT.shape[0] != Poly_Traj.S-dof:
            raise Exception
        if q0.shape[1] != Poly_Traj.N or qT.shape[1] != Poly_Traj.N:
            raise Exception

        qT = np.concatenate((qT, np.zeros((dof, Poly_Traj.N), dtype=np.float32)), axis=0)
        q = np.vstack([q0, qT])
        c = self.getMatrixMInv(ts=ts, dof=dof) @ q

        return Poly_Traj(ts[1], c)

    def getMatrixMInv(self, ts:NDArray[np.float32], dof:int) -> NDArray[np.float32]:
        '''
        获得当前用于求解多项式系数的M^-1矩阵
        '''
        if self.Minv is not None:
            return self.Minv

        M = Poly_Traj.construct_M(ts)
        M_inv = np.linalg.inv(M)


        pQptQ = np.zeros_like(M)
        pQptQ[-1-dof+1:,-1-dof+1:] = np.eye(dof)
        pCpQ = M_inv.T
        # pJpC 是代价函数对多项式系数的偏导数（∂J/∂c），由其他函数累计得到
        pJptQ = pQptQ @ pCpQ @ self.pJpC

        newM = M.copy()
        newM[-1-dof+1:,:] = pJptQ[-1-dof+1:,:]
        newM_inv = np.linalg.inv(newM)

        return newM_inv

    def getMatrixM(self, ts:NDArray[np.float32], dof:int) -> NDArray[np.float32]:
        '''
        获得当前用于求解多项式系数的M^-1矩阵
        '''
        if self.Minv is not None:
            return self.Minv

        M = Poly_Traj.construct_M(ts)
        M_inv = np.linalg.inv(M)


        pQptQ = np.zeros_like(M)
        pQptQ[-1-dof+1:,-1-dof+1:] = np.eye(dof)
        pCpQ = M_inv.T
        # pJpC 是代价函数对多项式系数的偏导数（∂J/∂c），由其他函数累计得到
        pJptQ = pQptQ @ pCpQ @ self.pJpC

        newM = M.copy()
        newM[-1-dof+1:,:] = pJptQ[-1-dof+1:,:]
        return newM

    def solveEndPosEqu(self, ts:NDArray[np.float32], dof:int, t:float, rank:int, q0:NDArray[np.float32], maxMinEquRVal:float, velT:Optional[NDArray[np.float32]]) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
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
        phi_rank = Poly_Traj.construct_beta(t=t, rank=rank)
        Minv =  self.getMatrixMInv(ts=ts, dof=dof)
        self.boundCoffA = phi_rank.T @ Minv

        # 构造系数向量[p0, v0, a0, 0, vT, 0]
        q = np.vstack([q0, np.zeros((Poly_Traj.S, Poly_Traj.N))])
        # 末端速度受约束
        if dof==1:
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

    def solveEndPosBound(self, ts:NDArray[np.float32], dof:int, q0:NDArray[np.float32], velT:Optional[NDArray[np.float32]], rank:int, maxVel:float) \
            -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
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
            pos_u, pos_d = self.solveEndPosEqu(ts=ts, dof=dof, t=t, rank=rank, q0=q0, velT=velT, maxMinEquRVal=maxVel)
            bound_u_results[i, :] = pos_u
            bound_d_results[i, :] = pos_d
        return np.min(bound_u_results, axis=0), np.max(bound_d_results, axis=0)

    @staticmethod
    def betabetaT_int(t:float, rank:int) -> NDArray[np.float32]:
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
    def betabetaT(t:float, rank:int) -> NDArray[np.float32]:
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


class TrajectoryOptimizer:
    def __init__(self, T, N, v_max, a_max, k_max, q0, qd):
        """
        初始化轨迹优化器。

        参数：
        - T: 总时间
        - N: 离散时间点数
        - v_max: 最大速度
        - a_max: 最大加速度
        - k_max: 最大曲率
        - q0: 初始状态，包含 'pos'、'vel'、'acc'，每个都是二维向量
        - qd: 期望的目标位置，二维向量
        """
        self.T = T
        self.N = N
        self.dt = T / (N - 1)
        self.v_max = v_max
        self.a_max = a_max
        self.k_max = k_max
        self.q0 = q0  # 初始状态：位置、速度、加速度
        self.qd = qd  # 期望的终点位置

        self.factory = MincoPolyTrajFactory()
        self.factory.add_control_effort_cost(self.T)

        # 时间离散化
        self.time_points = np.linspace(0, T, N)

        # 构建五次多项式的 M 矩阵
        self.M = self._create_M_matrix(T)

    def _create_M_matrix(self, T):
        """
        创建五次多项式的 M 矩阵，将系数 c 与边界条件 q 关联起来。
        """
        factory = MincoPolyTrajFactory()
        factory.add_control_effort_cost(T)

        return factory.getMatrixM(np.array([0,T]), 2)

    def _compute_trajectory(self, q):
        """
        计算给定终点状态 q 下的轨迹：位置、速度、加速度和曲率。

        参数：
        - q: 字典，包含 'pos'、'vel'、'acc'，每个都是二维向量
        """

        ndq = np.array([q['pos']])
        minco_traj = self.factory.solveWithCostJ(q0=self.q0, qT=ndq, ts=np.array([0, self.T]), dof=2)

        # 初始化轨迹存储
        positions = []
        velocities = []
        accelerations = []
        curvatures = []

        for t in self.time_points:
            positions.append(minco_traj.get_pos(t))
            velocities.append(minco_traj.get_vel(t))
            accelerations.append(minco_traj.get_acc(t))
            curvatures.append(minco_traj.get_acc(t))

        # 转换为数组
        positions = np.array(positions)
        velocities = np.array(velocities)
        accelerations = np.array(accelerations)
        curvatures = np.array(curvatures)

        return positions, velocities, accelerations, curvatures

    def constraints(self, q_flat):
        """
        约束函数：速度、加速度和曲率不得超过指定的最大值。

        返回：
        - 约束值的数组（应该满足 <= 0）
        """
        # 从展平的向量中提取 q
        q = {
            'pos': np.array([q_flat[0], q_flat[3]]),
            'vel': np.array([q_flat[1], q_flat[4]]),
            'acc': np.array([q_flat[2], q_flat[5]])
        }

        # 计算轨迹
        positions, velocities, accelerations, curvatures = self._compute_trajectory(q)

        # 最大速度平方
        v_sq = np.sum(velocities**2, axis=1)
        max_v_sq = np.max(v_sq)
        v_constraint = max_v_sq - self.v_max**2

        # 最大加速度平方
        a_sq = np.sum(accelerations**2, axis=1)
        max_a_sq = np.max(a_sq)
        a_constraint = max_a_sq - self.a_max**2

        # 最大曲率
        kappa_abs = np.abs(curvatures)
        max_kappa = np.max(kappa_abs)
        k_constraint = max_kappa - self.k_max

        # 返回约束数组
        return np.array([v_constraint, a_constraint, k_constraint])

    def objective(self, q_flat):
        """
        目标函数：最小化终点状态与期望终点状态的差异。

        参数：
        - q_flat: 展平的终点状态向量，长度为6：[xT, vxT, axT, yT, vyT, ayT]
        """
        # 从展平的向量中提取 q
        q = {
            'pos': np.array([q_flat[0], q_flat[3]]),
            'vel': np.array([q_flat[1], q_flat[4]]),
            'acc': np.array([q_flat[2], q_flat[5]])
        }

        # 权重（可根据需要调整）
        weight_pos = 1.0

        pos_error = q['pos'] - self.qd
        objective = (weight_pos * np.linalg.norm(pos_error))**2

        return objective

    def optimize(self):
        """
        执行优化过程。
        """
        # 初始猜测：期望的终点位置，速度和加速度为零
        q0_flat = np.array([self.qd[0], 0.0, 0.0, self.qd[1], 0.0, 0.0])

        # 定义约束（非线性不等式约束）
        cons = ({
            'type': 'ineq',
            'fun': lambda q_flat: -self.constraints(q_flat)  # 注意取负号，使得约束为 <= 0
        })

        # 执行优化
        result = optimize.minimize(
            self.objective,
            q0_flat,
            method='SLSQP',
            constraints=cons,
            options={'maxiter': 1000}
        )

        if not result.success:
            print("优化失败:", result.message)

        # 提取优化后的终点状态
        q_opt_flat = result.x
        q_opt = {
            'pos': np.array([q_opt_flat[0], q_opt_flat[3]]),
            'vel': np.array([q_opt_flat[1], q_opt_flat[4]]),
            'acc': np.array([q_opt_flat[2], q_opt_flat[5]])
        }

        return q_opt

    def animate_trajectory(self, q_opt):
        """
        创建轨迹动画，包含机器人位置、实时速度、加速度显示，以及速度和加速度矢量箭头。
        """
        positions, velocities, accelerations, curvatures = self._compute_trajectory(q_opt)
        time = self.time_points

        # 创建图形和子图
        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(2, 2)

        # 轨迹子图
        ax_traj = fig.add_subplot(gs[0, 0])
        # 速度子图
        ax_vel = fig.add_subplot(gs[0, 1])
        # 加速度子图
        ax_acc = fig.add_subplot(gs[1, :])

        # 设置轨迹图的范围
        margin = 1.0
        x_min, x_max = min(positions[:, 0].min(), self.q0[0][0]) - margin, max(positions[:, 0].max(), self.qd[0]) + margin
        y_min, y_max = min(positions[:, 1].min(), self.q0[0][1]) - margin, max(positions[:, 1].max(), self.qd[1]) + margin

        # 初始化机器人点（用圆表示）
        robot = Circle((positions[0, 0], positions[0, 1]), 0.2, color='blue', fill=True)
        ax_traj.add_patch(robot)

        # 初始化速度和加速度箭头
        scale_v = 1.0  # 速度箭头缩放因子
        scale_a = 1.0  # 加速度箭头缩放因子
        vel_arrow = ax_traj.quiver(positions[0, 0], positions[0, 1],
                                velocities[0, 0], velocities[0, 1],
                                color='green', scale=10, width=0.005,
                                label='Velocity')
        acc_arrow = ax_traj.quiver(positions[0, 0], positions[0, 1],
                                accelerations[0, 0], accelerations[0, 1],
                                color='red', scale=10, width=0.005,
                                label='Acceleration')

        # 初始化速度和加速度图的线
        vel_line, = ax_vel.plot([], [], 'b-', label='Velocity')
        acc_line, = ax_acc.plot([], [], 'r-', label='Acceleration')

        # 速度和加速度的时间历史数据
        vel_data = {'t': [], 'v': []}
        acc_data = {'t': [], 'a': []}

        def init():
            # 设置轨迹图
            ax_traj.plot(positions[:, 0], positions[:, 1], 'g--', alpha=0.5, label='Planned Path')
            ax_traj.scatter(self.q0[0][0], self.q0[0][1], color='green', label='Start')
            ax_traj.scatter(self.qd[0], self.qd[1], color='red', label='Goal')
            ax_traj.set_xlim(x_min, x_max)
            ax_traj.set_ylim(y_min, y_max)
            ax_traj.set_aspect('equal')
            ax_traj.grid(True)
            ax_traj.set_title('Robot Trajectory')
            ax_traj.legend()

            # 设置速度图
            ax_vel.set_xlim(0, self.T)
            ax_vel.set_ylim(0, self.v_max * 1.2)
            ax_vel.axhline(y=self.v_max, color='r', linestyle='--', label='v_max')
            ax_vel.grid(True)
            ax_vel.set_title('Velocity')
            ax_vel.legend()

            # 设置加速度图
            ax_acc.set_xlim(0, self.T)
            ax_acc.set_ylim(0, self.a_max * 1.2)
            ax_acc.axhline(y=self.a_max, color='r', linestyle='--', label='a_max')
            ax_acc.grid(True)
            ax_acc.set_title('Acceleration')
            ax_acc.legend()

            return robot, vel_arrow, acc_arrow, vel_line, acc_line

        def animate(frame):
            # 更新机器人位置
            robot.center = (positions[frame, 0], positions[frame, 1])

            # 更新速度箭头
            vel_norm = np.linalg.norm(velocities[frame])
            if vel_norm > 0:
                vel_arrow.set_offsets([positions[frame, 0], positions[frame, 1]])
                vel_arrow.set_UVC(velocities[frame, 0] * scale_v / vel_norm,
                                velocities[frame, 1] * scale_v / vel_norm)

            # 更新加速度箭头
            acc_norm = np.linalg.norm(accelerations[frame])
            if acc_norm > 0:
                acc_arrow.set_offsets([positions[frame, 0], positions[frame, 1]])
                acc_arrow.set_UVC(accelerations[frame, 0] * scale_a / acc_norm,
                                accelerations[frame, 1] * scale_a / acc_norm)

            # 更新速度和加速度数据
            current_time = time[frame]
            current_vel = np.linalg.norm(velocities[frame])
            current_acc = np.linalg.norm(accelerations[frame])

            vel_data['t'].append(current_time)
            vel_data['v'].append(current_vel)
            acc_data['t'].append(current_time)
            acc_data['a'].append(current_acc)

            # 更新速度和加速度图
            vel_line.set_data(vel_data['t'], vel_data['v'])
            acc_line.set_data(acc_data['t'], acc_data['a'])

            return robot, vel_arrow, acc_arrow, vel_line, acc_line

        # 创建动画
        anim = FuncAnimation(
            fig, animate, init_func=init,
            frames=len(time), interval=50,
            blit=True, repeat=False
        )

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # 参数设置
    T = 5.0    # 总时间
    N = 100    # 离散时间点数
    v_max = 2.0  # 最大速度
    a_max = 1.0  # 最大加速度
    k_max = 0.5  # 最大曲率

    # 初始状态
    q0 = np.array([[0.5,0.5],[0.5,0],[0,0]])
    # 期望的终点位置
    qd = np.array([10.0, 10.0])  # 目标位置 (xT, yT)

    # 创建优化器实例
    optimizer = TrajectoryOptimizer(T, N, v_max, a_max, k_max, q0, qd)

    # 执行优化
    q_opt = optimizer.optimize()
    print("优化后的终点状态：")
    print("位置:", q_opt['pos'])
    print("速度:", q_opt['vel'])
    print("加速度:", q_opt['acc'])

    # 绘制结果
    optimizer.animate_trajectory(q_opt)