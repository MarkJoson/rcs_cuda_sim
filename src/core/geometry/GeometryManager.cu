#include <cstdint>
#include <cstddef>

/**
 * @brief 依据动态形状的位置，将动态形状的多边形边，转换到世界坐标系下
 block_dim: <2, min(128, num_env_lo), 1>
 grid_dim: <num_env_hi, num_dyn_lines_, 1>
 * @param num_dynobj_lines 动态物体边的数量总和
 * @param dyn_obj_lines 动态物体边外形
 * @param dyn_obj_line_ids 动态物体边对应的object id
 * @param dyn_obj_poses 动态物体位置[obj_id, group, env, 4]
 * @param dyn_lines 输出，动态物体的边 [group, env, lines, 4]
 */
__global__ void transformDynamicLinesKernel(
        const uint32_t num_dynobj_lines,
        const uint32_t num_envs,
        const float     * __restrict__ dyn_obj_lines,       // 多边形边集合
        const uint32_t  * __restrict__ dyn_obj_line_ids,    // 多边形边集合对应的物体ID
        const float     * __restrict__ dyn_obj_poses,       // 多边形形状位姿，[obj, group, env]
        float           * __restrict__ dyn_lines) {         // 当前多边形外形线段在坐标系的端点 [group, env, lines, 4]

    // 每个block负责dyn_obj的一个line(y), 每个线程负责一个点(x)
    // const uint32_t point_id = threadIdx.x;          // 线段的点Id, 0起点，1终点
    // const uint32_t sub_batch_id = threadIdx.y;      // 子batch id, 32~256
    // const uint32_t prim_batch_id = blockIdx.x;      // 主batch id, 0~1024
    const uint32_t batch_id = blockIdx.x * blockDim.y + threadIdx.y;

    const uint32_t line_id = blockIdx.y;
    const uint32_t dyn_obj_id = dyn_obj_line_ids[line_id] >> 16;

    if (line_id >= num_dynobj_lines || batch_id >= num_envs) return;

    const uint32_t global_point_id = line_id * 2 + threadIdx.x;
    const float2 vertex = reinterpret_cast<const float2 *>(dyn_obj_lines)[global_point_id];

    // [obj_id, batch, 4]
    const float4 pose = reinterpret_cast<const float4 *>(dyn_obj_poses)[dyn_obj_id * num_envs + batch_id];
    // float sint = pose.z, cost = pose.w;
    float sint, cost;
    sincosf(pose.z, &sint, &cost);

    float2 new_vertex = {
            vertex.x * cost - vertex.y * sint + pose.x,
            vertex.x * sint + vertex.y * cost + pose.y
    };

    const uint32_t output_indice = (num_dynobj_lines * batch_id + line_id) * 2 + threadIdx.x;
    reinterpret_cast<float2 *>(dyn_lines)[output_indice] = new_vertex;
}