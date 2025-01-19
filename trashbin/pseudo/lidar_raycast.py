import numpy as np
import numba
from numba import cuda
import os

os.environ["CUDA_COMPUTE_CAPABILITY"] = "75"
cuda.config.CUDA_USE_NVIDIA_BINDING = True
# numba.config.CUDA_DEFAULT_PTX = '84'

FREE = 255
OBSTACLE = 0
UNKNOWN = 127


def parallel_raycast_kernel(bx, by, ty, tx, origins_x, origins_y, angles, ranges, grid_map, resolution, P):
    """CUDA kernel for parallel raycasting

    Args:
        origins_x, origins_y: 激光雷达原点坐标数组 shape=(num_robots,)
        angles: 每个激光雷达的射线角度数组 shape=(num_robots, num_rays)
        ranges: 每条射线的距离读数 shape=(num_robots, num_rays)
        grid_map: 栅格地图 shape=(height, width)
        resolution: 栅格分辨率
        P: 每条射线的并行处理器数量
    """
    # 获取当前线程ID
    robot_id = bx # 机器人ID
    ray_id = by
    processor_id = tx

    if robot_id >= origins_x.shape[0] or ray_id >= angles.shape[1]:
        return

    # 获取当前射线信息
    start_x = origins_x[robot_id]
    start_y = origins_y[robot_id]
    angle = angles[robot_id, ray_id]
    r = ranges[robot_id, ray_id]

    # 计算终点坐标
    end_x = start_x + r * np.cos(angle)
    end_y = start_y + r * np.sin(angle)

    # 将物理坐标转换为栅格坐标
    grid_start_x = int(start_x / resolution)
    grid_start_y = int(start_y / resolution)
    grid_end_x = int(end_x / resolution)
    grid_end_y = int(end_y / resolution)

    # 计算dx和dy
    dx = abs(grid_end_x - grid_start_x)
    dy = abs(grid_end_y - grid_start_y)

    # 计算每个处理器负责的区间
    w = (dx + P - 1) // P  # 向上取整
    x_start = processor_id * w
    x_end = min(x_start + w, dx)

    # 计算初始y坐标(参考论文公式)
    t = dy * x_start
    y = (2 * t + dx) // (2 * dx)

    # Bresenham算法更新栅格
    x = x_start
    while x < x_end:
        # 更新当前栅格为FREE
        current_x = grid_start_x + x
        current_y = grid_start_y + y
        if 0 <= current_x < grid_map.shape[1] and 0 <= current_y < grid_map.shape[0]:
            grid_map[current_y, current_x] = FREE

        # Bresenham迭代
        error = 2 * (dy * x - dx * y)
        if error < 0:
            x += 1
        else:
            x += 1
            y += 1

    # 更新终点栅格为OBSTACLE
    if processor_id == P-1:
        if 0 <= grid_end_x < grid_map.shape[1] and 0 <= grid_end_y < grid_map.shape[0]:
            grid_map[grid_end_y, grid_end_x] = OBSTACLE

def parallel_raycast(robot_poses, num_rays, laser_data, grid_map, resolution):
    """
    主函数:并行处理所有机器人的激光数据
    """
    # 准备数据
    num_robots = len(robot_poses)
    P = 32  # 每条射线的并行数

    # 配置CUDA线程块
    threadsperblock = (P, 1)
    blockspergrid = (num_robots, num_rays)

    # 分配GPU内存
    d_origins_x = cuda.to_device(np.ascontiguousarray(robot_poses[:, 0]))
    d_origins_y = cuda.to_device(np.ascontiguousarray(robot_poses[:, 1]))
    d_angles = cuda.to_device(laser_data['angles'])
    d_ranges = cuda.to_device(laser_data['ranges'])
    d_grid_map = cuda.to_device(grid_map)

    # 启动kernel
    for by in range(blockspergrid[0]):
        for bx in range(blockspergrid[1]):
            for ty in range(threadsperblock[0]):
                for tx in range(threadsperblock[1]):
                    parallel_raycast_kernel(by, bx, ty, tx,
                        robot_poses[:, 0], robot_poses[:, 1], laser_data['angles'], laser_data['ranges'],
                        grid_map, resolution, P
                    )
    # parallel_raycast_kernel[blockspergrid, threadsperblock](
    #     d_origins_x, d_origins_y, d_angles, d_ranges,
    #     d_grid_map, resolution, P
    # )

    # 将结果拷贝回主机
    # grid_map = d_grid_map.copy_to_host()

    return grid_map


import numpy as np
from numba import cuda

class TestParallelRaycast():

    def setUp(self):
        # Initialize test data
        self.robot_poses = np.array([[0.0, 0.0], [1.0, 1.0]])
        self.laser_data = {
            'angles': np.array([[0, np.pi/4], [np.pi/2, np.pi]]),
            'ranges': np.array([[1.0, 1.0], [1.0, 1.0]])
        }
        self.grid_map = np.full((100, 100), 127, dtype=np.uint8)
        self.resolution = 0.1

    def test_parallel_raycast(self):
        result = parallel_raycast(self.robot_poses, 16, self.laser_data, self.grid_map, self.resolution)
        pass

if __name__ == '__main__':
    test = TestParallelRaycast()
    test.setUp()
    test.test_parallel_raycast()