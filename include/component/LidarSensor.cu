#ifndef CUDASIM_COMPONENT_LIDAR_HH
#define CUDASIM_COMPONENT_LIDAR_HH

#include <cub/cub.cuh>
#include "core/Component.hh"


namespace cuda_simulator {
namespace component {


#define CTA_SIZE            (128)
#define RASTER_WARPS        (CTA_SIZE/32)

#define LINE_BUF_SIZE       (CTA_SIZE*2)
#define FR_BUF_SIZE         (CTA_SIZE*2)
#define FRAG_BUF_SIZE       (CTA_SIZE*6)

#define EMIT_PER_THREAD     (2)
#define TOTAL_EMITION       (EMIT_PER_THREAD * CTA_SIZE)

#define LIDAR_LINES_LOG2    (10)         // 128lines
#define LIDAR_LINES         (1<<LIDAR_LINES_LOG2)

#define LIDAR_MAX_RANGE     (8.0f)
#define LIDAR_RESOLU        (M_PI/LIDAR_LINES)
#define LIDAR_RESOLU_INV    (LIDAR_LINES/M_PI)


__host__ __device__ bool prepareLine(float2 vs, float2 ve, float max_range)
{
    float cdot = vs.x*ve.y - vs.y*ve.x;
    float dx = ve.x-vs.x;
    float dy = ve.y-vs.y;
    // float len = sqrtf(dx*dx+dy*dy);
    float invsqrt = rsqrtf(dx*dx+dy*dy);
    float dist = fabs(cdot)*invsqrt;

    return dist < max_range && cdot>=0;
}


__forceinline__ __host__ __device__ float vec_atan2_0_360(float2 vec)
{
    float angle = atan2f(vec.y, vec.x);
    return angle < 0 ? angle + 2*M_PI : angle;
}

__forceinline__ __host__ __device__ float getR(float2 lb, float2 le, float theta)
{
    float sin,cos;
    sincosf(theta, &sin, &cos);
    float2 dvec = make_float2(le.x - lb.x, le.y - lb.y);
    return (lb.y*dvec.x-lb.x*dvec.y)/(dvec.x*sin-dvec.y*cos);
}


__forceinline__ __host__ __device__ uint32_t genMask(int start, int end)
{
    return ((1<<end)-1) & ~((1<<start)-1);
    // int pop = __builtin_popcount(mask);
}


__global__ void rasterKernel(
    int                         num_lines,           // 场景中的线段数量
    const float2 * __restrict__ line_begins,        // 线段的起点
    const float2 * __restrict__ line_ends,          // 线段的终点
    const float3 * __restrict__ poses,              // 机器人的位姿
    int          * __restrict__ lidar_response      // 激光雷达的响应
)
{
    using BlockScan = cub::BlockScan<uint32_t, CTA_SIZE>;
    using BlockRunLengthDecodeT = cub::BlockRunLengthDecode<uint32_t, CTA_SIZE, 1, EMIT_PER_THREAD>;

    volatile __shared__ uint32_t s_lineBuf[LINE_BUF_SIZE];           // 1K
    volatile __shared__ uint32_t s_frLineIdxBuf[FR_BUF_SIZE];        // 1K
    volatile __shared__ uint32_t s_frLineSGridFragsBuf[FR_BUF_SIZE]; // 1K
    __shared__ uint32_t s_lidarResponse[LIDAR_LINES];            // 1K
    __shared__ union {
        typename BlockScan::TempStorage scan_temp_storage;
        typename BlockRunLengthDecodeT::TempStorage decode_temp_storage;
    } temp_storage;

    uint32_t tid = threadIdx.x;
    uint32_t totalLineRead = 0;
    uint32_t lineBufRead=0, lineBufWrite=0;
    uint32_t frLineBufRead=0, frLineBufWrite=0;
    float3 pose = poses[blockIdx.x];

    /********* 初始化lidar数据 *********/
    for(int i=tid; i<LIDAR_LINES; i+=CTA_SIZE)
        s_lidarResponse[i] = LIDAR_MAX_RANGE*100;

    // if(tid == 0) printf("---------##### Task RECEIVED\n");

    for(;;)
    {
        /********* Load Lines to Buffer *********/

        while(lineBufWrite-lineBufRead < CTA_SIZE && totalLineRead < num_lines)
        {
            // if(tid == 0) printf("[READ] LINE RANGE: %d~%d.\n", totalLineRead, totalLineRead+CTA_SIZE);
            uint32_t visibility = false;
            int lineIdx = totalLineRead+tid;
            if(lineIdx < num_lines)
            {
                float2 lb = make_float2(line_begins[lineIdx].x-pose.x, line_begins[lineIdx].y-pose.y);
                float2 le = make_float2(line_ends[lineIdx].x-pose.x, line_ends[lineIdx].y-pose.y);
                visibility = prepareLine(lb, le, LIDAR_MAX_RANGE);
                // if(visibility)
                // {
                //     const auto &vs=lb, &ve=le;
                //     float cdot = vs.x*ve.y - vs.y*ve.x;
                //     float dx = ve.x-vs.x;
                //     float dy = ve.y-vs.y;
                //     float len = sqrtf(dx*dx+dy*dy);
                //     float dist = fabs(cdot)/len;
                //     printf("[READ] PASS:%03d, \t[%.2f,%.2f]->[%.2f,%.2f], \t[%.2f,%.2f]->[%.2f,%.2f], \tarea:%.2f, \tdist%.2f\n", lineIdx,
                //         line_begins[lineIdx].x, line_begins[lineIdx].y, line_ends[lineIdx].x, line_ends[lineIdx].y,
                //         lb.x, lb.y, le.x, le.y, cdot, dist);
                // }
            }

            uint32_t scan, scan_reduce;
            BlockScan(temp_storage.scan_temp_storage).ExclusiveSum(visibility, scan, scan_reduce);

            if(visibility) {
                s_lineBuf[lineBufWrite+scan] = lineIdx;
            }

            lineBufWrite += scan_reduce;

            totalLineRead += CTA_SIZE;
            __syncthreads();
        }

        // 第二部分继续的条件：已经读取了128个，或没有读取128个，但没有新的线段

        // if(tid == 0) printf("[READ] FINISHED! TOTAL READ: %d, LINE BUF:%d\n", totalLineRead, lineBufWrite);

        /********* 计算终止栅格，细光栅化，存入待发射线段缓冲区 *********/
        // do
        {
            int lineIdx = -1;
            int frag = 0;
            int s_grid = -1;
            if(lineBufRead + tid < lineBufWrite)
            {
                lineIdx = s_lineBuf[(lineBufRead + tid) % LINE_BUF_SIZE];
                float2 lb = make_float2(line_begins[lineIdx].x-pose.x, line_begins[lineIdx].y-pose.y);
                float2 le = make_float2(line_ends[lineIdx].x-pose.x, line_ends[lineIdx].y-pose.y);

                auto s_angle = vec_atan2_0_360(lb);
                auto e_angle = vec_atan2_0_360(le);
                s_grid = s_angle * LIDAR_RESOLU_INV;
                int e_grid = e_angle * LIDAR_RESOLU_INV;

                frag = (e_grid-s_grid) + ((e_grid < s_grid) ? LIDAR_LINES : 0);
                // printf("[RASTER] THREAD:%d, LINE_ID:%d, SGRID:%d, EGRID:%d, FRAG:%d\n", tid, lineIdx, s_grid, e_grid, frag);
            }

            // 压缩到FR_BUF队列中
            uint32_t scan, scan_sum;
            BlockScan(temp_storage.scan_temp_storage).ExclusiveSum(frag>0, scan, scan_sum);
            if(frag > 0)
            {
                uint32_t idx = (frLineBufWrite+scan) % FR_BUF_SIZE;
                s_frLineIdxBuf[idx] = lineIdx;
                s_frLineSGridFragsBuf[idx] = (s_grid << 16) | (frag & 0xffff);

            }
            frLineBufWrite += scan_sum;
            __syncthreads();

            //
            lineBufRead = min(lineBufRead+CTA_SIZE, lineBufWrite);
        }

        // 此时要么 读取了128个，要么 lineBuf处理完了
        // if(tid == 0) printf("[RASTER] FRBuf:[R:%d,W:%d] VALID: %d\n", frLineBufRead, frLineBufWrite, frLineBufWrite - frLineBufRead);

        // 第三部分继续的条件：读取到128个，或未读取到128个，但是已经无法再读取新的线段；
        if(frLineBufWrite-frLineBufRead < CTA_SIZE && (lineBufRead < lineBufWrite || totalLineRead < num_lines)) {
            // if(tid == 0) printf("[RASTER] CONTINUE LOOP!\n");
            continue;
        }

        // if(tid == 0) printf("[RASTER] FINISHED!\n");

        /********* Count and Emit *********/
        do
        {
            // if(tid == 0) printf("[EMIT] LOAD LINE [%d-%d]\n", frLineBufRead, frLineBufWrite);
            // 加载CTA_SIZE个到缓冲区，准备进行Decode
            uint32_t runValue[1] = {0}, runLength[1] = {0};
            int frLineBufIdx = frLineBufRead + tid;
            if(frLineBufIdx < frLineBufWrite)
            {
                frLineBufIdx = frLineBufIdx % FR_BUF_SIZE;
                runValue[0] = frLineBufIdx;
                runLength[0] = s_frLineSGridFragsBuf[frLineBufIdx] & 0xffff;        // 取低16位的frag
            }
            frLineBufRead = min(frLineBufRead+CTA_SIZE, frLineBufWrite);
            __syncthreads();

            uint32_t total_decoded_size = 0;
            BlockRunLengthDecodeT blk_rld(temp_storage.decode_temp_storage, runValue, runLength, total_decoded_size);

            // 将本次读取的 CTA_SIZE*EMIT_PER_LINE 个frag全部发射
            uint32_t decoded_window_offset = 0;
            while(decoded_window_offset < total_decoded_size)
            {
                uint32_t relative_offsets[2];
                uint32_t decoded_items[2];
                uint32_t num_valid_items = min(total_decoded_size - decoded_window_offset, CTA_SIZE * EMIT_PER_THREAD);
                blk_rld.RunLengthDecode(decoded_items, relative_offsets, decoded_window_offset);
                decoded_window_offset += num_valid_items;

                #pragma unroll
                for(int i=0; i<2; i++)
                {
                    if(tid*EMIT_PER_THREAD + i >= num_valid_items)
                        break;

                    int fragIdx = relative_offsets[i];
                    uint32_t frLineBufIdx = decoded_items[i];
                    uint32_t lineIdx = s_frLineIdxBuf[frLineBufIdx];
                    int s_grid = s_frLineSGridFragsBuf[frLineBufIdx] >> 16;

                    float2 lb = make_float2(line_begins[lineIdx].x-pose.x, line_begins[lineIdx].y-pose.y);
                    float2 le = make_float2(line_ends[lineIdx].x-pose.x, line_ends[lineIdx].y-pose.y);
                    int grid = (s_grid + fragIdx + 1) % LIDAR_LINES;
                    uint16_t response = getR(lb, le, grid*LIDAR_RESOLU) * 1024;        // 10位定点小数表示，最大距离64m
                    uint32_t resp_idx = response << 16 | lineIdx & 0xffff;
                    atomicMin_block(&s_lidarResponse[grid], resp_idx);
                }
            }
            __syncthreads();
        } while(frLineBufWrite != frLineBufRead && totalLineRead >= num_lines);       // 继续的条件：已经没有办法读取更多的frag线段，则需要将剩余的frlineBufWrite处理完

        // if(tid == 0) printf("[EMIT] FINISHED!\n");

        // 全部线段已经处理完
        if(totalLineRead >= num_lines) break;
    }

    for(int i=tid; i<LIDAR_LINES; i+=CTA_SIZE)
        lidar_response[i] = s_lidarResponse[i];
}


class LidarSensor : public core::Component {
public:
    LidarSensor() : Component("LidarSensor") {}

    void onEnvironGroupInit(core::SimulatorContext* context) override {
        // 初始化LidarSensor
    }

    void onReset(core::TensorHandle reset_flags, core::NodeExecStateType &state) override {
        // 重置LidarSensor
    }

    virtual void onRegister(core::SimulatorContext* context) {

    }

    virtual void onExecute( core::SimulatorContext* context, const core::NodeExecInputType &input, const core::NodeExecOutputType &output) {
        //
        // rasterKernel, block大小 == 128, 1个block处理1个机器人. grid大小 == (环境组数,环境数,机器人数)
        // rasterKernel: Input:[poses], Output:[lidar_response]
        // 线段数据：储存在全局内存，numLines, 储存在 constant 内存
        // 1. 从场景管理器中获得所有线段的数据：line_begins, line_ends
        // 2. 计算机器人的位姿poses
        // 3. 发布激光雷达的响应lidar_response
    }
};



} // namespace component
} // namespace cuda_simulator



#endif // CUDASIM_COMPONENT_LIDAR_HH