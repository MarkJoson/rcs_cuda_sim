import numpy as np
import jax
from jax import Array
import jax.numpy as jnp

from numpy.typing import NDArray
from typing import Tuple, Optional
from scipy import optimize

from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
import matplotlib.pyplot as plt
from minco_traj import MincoPolyTrajFactory, Poly_Traj


def sigmoid(x: jnp.ndarray) -> jnp.ndarray:
    """实现 sigmoid 函数"""
    return 1 / (1 + jnp.exp(-x))


def object_fn(qT_opting:Array, qT_expect:Array, q0:Array, Minv:Array, trajT:float, dof:int, vmax:float, amax:float):
    diff = (qT_expect.flatten()[:2] - qT_opting[:2])
    obj = diff.T @ diff

    qT = qT_opting.reshape(1,2)
    qT = jnp.concatenate((qT, jnp.zeros((dof, Poly_Traj.N), dtype=jnp.float32)), axis=0)
    q = jnp.vstack([q0, qT])

    vel_penalty = 0
    acc_penalty = 0
    for t in np.linspace(0,trajT, 5):
        beta1 = Poly_Traj.construct_beta(t=t, rank=1)
        beta2 = Poly_Traj.construct_beta(t=t, rank=2)
        vel_t = beta1.T @ Minv @ q
        acc_t = beta2.T @ Minv @ q
        v_sq = (vel_t @ vel_t.T - vmax**2)
        a_sq = (acc_t @ acc_t.T - amax**2)
        vel_penalty += jnp.maximum(0, v_sq)
        acc_penalty += jnp.maximum(0, a_sq)

        # vel_penalty += v_sq*sigmoid(v_sq*10)
        # acc_penalty += a_sq*sigmoid(a_sq*10)

    final_obj = obj + 5 * vel_penalty + 5 * acc_penalty
    return final_obj[0,0]


class TrajectoryOptimizer:
    def __init__(self, T, v_max, a_max, k_max):
        self.T = T
        self.ckpt = 5
        self.v_max = v_max
        self.a_max = a_max
        self.k_max = k_max

        self.factory = MincoPolyTrajFactory(ts=np.array([0, self.T]), dof=2)
        self.factory.add_control_effort_cost(self.T)
        self.factory.calcMatrixMInv()

        self.jac = jax.grad(object_fn)
        self.hess = jax.hessian(object_fn)
        # print(hess(jnp.array([[1.0, 1.0, 0.0, 0.0]]), qd, q0, optimizer.factory.Minv, optimizer.T, optimizer.factory.dof, 2, 2))

    def optimize(self, q0:NDArray, expect_qT:NDArray):
        result = optimize.minimize(
            fun=lambda qT_opting: object_fn(qT_opting, expect_qT, q0, self.factory.Minv, self.T, self.factory.dof, self.v_max, self.a_max),
            x0=q0[0,:].flatten(),
            jac=lambda qT_opting: self.jac(qT_opting, expect_qT, q0, self.factory.Minv, self.T, self.factory.dof, self.v_max, self.a_max),
            # hess=lambda qT_opting: self.hess(qT_opting, qd, q0, self.factory.Minv, self.T, self.factory.dof, self.v_max, self.a_max),
            method='trust-constr',
            options={'maxiter': 1000}
        )

        if not result.success:
            print("优化失败:", result.message)

        # 提取优化后的终点状态
        qT_res:NDArray = result.x
        # qT_res:NDArray[np.float32] = expect_qT
        minco_traj = self.factory.solveWithCostJ(q0=q0, qT=qT_res.reshape(1,2))

        qT = np.array([
            minco_traj.get_pos(t=self.T),
            minco_traj.get_vel(t=self.T),
            minco_traj.get_acc(t=self.T),
        ])

        return qT


if __name__ == "__main__":

    q0 = np.array([[0.5,0.5],[0,0],[0,0]])
    qd = np.array([[10.0, 10.0]])  # 目标位置 (xT, yT)

    optimizer = TrajectoryOptimizer(T=0.5, v_max=2.0, a_max=5.0, k_max=1.0)

    for i in range(50):
        print(f"第 {i} 次迭代")
        q_opt = optimizer.optimize(q0, qd)
        print("优化后的终点状态：")
        print("位置:", q_opt[0])
        print("速度:", q_opt[1])
        print("加速度:", q_opt[2])
        q0 = q_opt
