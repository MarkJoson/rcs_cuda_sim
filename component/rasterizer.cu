#include <vector>
#include <glog/logging.h>
#include <ranges>
#include <algorithm>
#include <iostream>
#include <cub/cub.cuh>
#include <thrust/device_vector.h>
#include <boost/range/join.hpp>
#include <SFML/Graphics.hpp>
#include <glm/glm.hpp>
#include "mapgen/mapgen_generate.h"
#include "mapgen/mapgen_postprocess.h"

union Line{
    struct {
        float sx;
        float sy;
        float ex;
        float ey;
    };
    float4 data;
};


std::vector<Line> lines;

constexpr int MAP_WIDTH = 80;
constexpr int MAP_HEIGHT = 60;
constexpr int GRID_SIZE = 1;

constexpr float scaling = 8;
constexpr float offset = 20;

constexpr float PI = M_PI;

#define LIDAR_LINES_LOG2    (8)         // 128lines
#define LIDAR_LINES         (1<<LIDAR_LINES_LOG2)

#define TILE_SIZE_LOG2      (5)         // 32lines
#define TILE_SIZE           (1<<TILE_SIZE_LOG2)

#define TILE_NUM            (1 << (LIDAR_LINES_LOG2-TILE_SIZE_LOG2))

#define CTA_SIZE            (128)

#define RASTER_WARPS        (CTA_SIZE/32)

#define LINE_BUF_SIZE       (CTA_SIZE*2)

#define FR_BUF_SIZE         (CTA_SIZE*2)

#define FRAG_BUF_SIZE       (CTA_SIZE*6)

#define EMIT_PER_THREAD     (2)


// thrust::device_vector<bool> lidar_inst_state;
// thrust::device_vector<int> line_count;

struct RCSParam
{
    int             *line_cnt;
    float2          *line_s;
    float2          *line_e;

    float2          *pos;
    bool            *inst_enabled;
    float           max_range;
    float           resolu_inv;
    float           resolu;
    int             ray_num;
};

__constant__ __device__ RCSParam g_params;


__host__ __device__ bool prepareLine(float2 vs, float2 ve, float max_range)
{
    float cdot = vs.x*ve.y - vs.y*ve.x;
    float dx = ve.x-vs.x;
    float dy = ve.y-vs.y;
    float len = sqrtf(dx*dx+dy*dy);
    float dist = fabs(cdot)/len;

    return dist < max_range && cdot>=0;
}

__forceinline__ __host__ __device__ float vec_atan2_0_360(float2 vec)
{
    float angle = atan2f(vec.y, vec.x);
    return angle < 0 ? angle + 2*PI : angle;
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

__global__ void raster(int numLines, float3 *poses, float2 *line_begins, float2 *line_ends)
{
    __shared__ uint32_t s_lineBuf[LINE_BUF_SIZE];        // 1K
    __shared__ uint32_t s_frLineIdxBuf[FR_BUF_SIZE];     // 1K
    __shared__ int s_frLineSGridBuf[FR_BUF_SIZE];        // 1K
    __shared__ int s_frLineFragsBuf[FR_BUF_SIZE];        // 1K
    __shared__ int s_lidarResponse[LIDAR_LINES];         // 1K

    using BlockScan = cub::BlockScan<uint32_t, CTA_SIZE>;
    using BlockRunLengthDecodeT = cub::BlockRunLengthDecode<uint32_t, CTA_SIZE, 1, EMIT_PER_THREAD>;
    union {
        typename BlockScan::TempStorage scan_temp_storage;
        typename BlockRunLengthDecodeT::TempStorage decode_temp_storage;
    } temp_storage;

    uint32_t tid = threadIdx.x;
    uint32_t lineBufRead=0, lineBufWrite=0;
    uint32_t frLineBufRead=0, frLineBufWrite=0;
    uint32_t workerIdx = blockIdx.x;
    float3 pose = poses[workerIdx];

    /********* 初始化lidar数据 *********/
    for(int i=tid; i<g_params.ray_num; i+=CTA_SIZE)
        s_lidarResponse[i] = g_params.max_range;

    for(;;)
    {
        /********* 加载线段到Buffer *********/
        while(lineBufWrite != numLines)
        {
            int lineIdx = lineBufWrite+tid;
            if(lineBufWrite-lineBufRead >= CTA_SIZE && lineIdx >= numLines) break;

            float2 lb = make_float2(line_begins[lineIdx].x-pose.x, line_begins[lineIdx].y-pose.y);
            float2 le = make_float2(line_ends[lineIdx].x-pose.x, line_ends[lineIdx].y-pose.y);

            bool visibility = prepareLine(lb, le, g_params.max_range);
            uint32_t scan, scan_sum;
            BlockScan(temp_storage.scan_temp_storage).ExclusiveSum(visibility, scan, scan_sum);

            if(visibility)
                s_lineBuf[lineBufWrite+scan] = lineIdx;
            lineBufWrite += scan_sum;
        }
        __syncthreads();

        /********* 计算终止栅格，细光栅化，存入待发射线段缓冲区 *********/
        if(lineBufRead + tid >= lineBufWrite)
        {
            int lineIdx = s_lineBuf[(lineBufRead + tid) % LINE_BUF_SIZE];
            float2 lb = make_float2(line_begins[lineIdx].x-pose.x, line_begins[lineIdx].y-pose.y);
            float2 le = make_float2(line_ends[lineIdx].x-pose.x, line_ends[lineIdx].y-pose.y);

            auto s_angle = vec_atan2_0_360(lb);
            auto e_angle = vec_atan2_0_360(le);
            int s_grid = s_angle * g_params.resolu_inv;
            int e_grid = e_angle * g_params.resolu_inv;

            int frag = e_grid-s_grid + (e_grid < s_grid) ? g_params.ray_num : 0;

            bool valid = frag > 0;
            uint32_t scan, scan_sum;
            BlockScan(temp_storage.scan_temp_storage).ExclusiveSum(valid, scan, scan_sum);
            if(valid)
            {
                uint32_t idx = frLineBufWrite+scan;
                s_frLineIdxBuf[idx] = lineIdx;
                s_frLineFragsBuf[idx] = frag;
                s_frLineSGridBuf[idx] = s_grid;
            }
            frLineBufWrite += scan_sum;
        }
        lineBufRead = min(lineBufRead+CTA_SIZE, lineBufWrite);

        __syncthreads();

        /********* 计算并发射 *********/
        // 如果frLine的缓冲区没有到 CTA_SIZE 继续读取线段
        if(frLineBufWrite-frLineBufRead < CTA_SIZE && lineBufWrite != numLines)
            continue;

        // 加载CTA_SIZE个到缓冲区，准备进行Decode
        uint32_t runValue[1] = {0}, runLength[1] = {0};
        int frLineBufIdx = frLineBufRead + tid;
        if(frLineBufIdx < frLineBufWrite)
        {
            frLineBufIdx = frLineBufIdx % FR_BUF_SIZE;
            runValue[0] = frLineBufIdx;
            runLength[0] = s_frLineSGridBuf[frLineBufIdx];
        }
        frLineBufRead = min(frLineBufRead+CTA_SIZE, frLineBufWrite);

        // TODO. Deocde 太费劲
        uint32_t total_decoded_size = 0;
        BlockRunLengthDecodeT blk_rld(temp_storage.decode_temp_storage, runValue, runLength, total_decoded_size);

        uint32_t decoded_window_offset = 0;
        while(decoded_window_offset < total_decoded_size)
        {
            uint32_t relative_offsets[2];
            uint32_t decoded_items[2];
            uint32_t num_valid_items = total_decoded_size - decoded_window_offset;
            blk_rld.RunLengthDecode(decoded_items, decoded_window_offset);
            decoded_window_offset += CTA_SIZE * EMIT_PER_THREAD;

            #pragma unroll
            for(int i=0; i<2; i++)
            {
                if(tid*EMIT_PER_THREAD + i>num_valid_items)
                    break;
                uint32_t frLineBufIdx = decoded_items[i];
                uint32_t lineIdx = s_frLineIdxBuf[frLineBufIdx];
                int frag = s_frLineFragsBuf[frLineBufIdx];
                int s_grid = s_frLineSGridBuf[frLineBufIdx];

                float2 lb = make_float2(line_begins[lineIdx].x-pose.x, line_begins[lineIdx].y-pose.y);
                float2 le = make_float2(line_ends[lineIdx].x-pose.x, line_ends[lineIdx].y-pose.y);
                int grid = (s_grid + frag + 1) % g_params.ray_num;
                int response = getR(lb, le, grid*g_params.resolu) * 100;
                s_lidarResponse[grid] = atomicMin_block(&s_lidarResponse[grid], response);
            }
        }
        __syncthreads();
    }
}


RCSParam g_hparams {
    .max_range = 20,
    .resolu_inv = LIDAR_LINES / (2*M_PI),
    .resolu = (2*M_PI) / LIDAR_LINES,
    .ray_num = LIDAR_LINES
};


// bool prepareLine(glm::vec2 vs, glm::vec2 ve)
// {
//     float cdot = vs.x*ve.y - vs.y*ve.x;

//     float len = glm::length(ve-vs);
//     float dist = fabs(cdot)/len;

//     return dist < g_hparams.max_range && cdot>=0;
// }




std::vector<float> cpuTest2(int numLines, float3 pose, const std::vector<float2> &line_begins, const std::vector<float2> &line_ends)
{
    // glm::vec2 pos = glm::vec2(inst_pos.x, inst_pos.y);

    // std::vector<int> lineFrag();
    // 初筛队列：背面剔除+距离剔除
    std::vector<int> lineBuf(LINE_BUF_SIZE);
    int lineBufRead=0, lineBufWrite=0;          // Read==Write空，(Write+1)%All==Read满

    // 交射线剔除
    std::vector<std::tuple<int, int, int>> frLineBuf(FR_BUF_SIZE);
    std::vector<int> frLineFrag(FR_BUF_SIZE);
    int frBufRead=0, frBufWrite=0;

    // 片段
    std::vector<std::pair<int,int>> fragBuf(CTA_SIZE*4+LIDAR_LINES);
    int fragRead=0, fragWrite=0;

    std::vector<float> response(LIDAR_LINES);
    // int totalFrag = 0;

    // TODO. 按线段长度排序

    for(int i=0; i<CTA_SIZE; i++)
        for(int j=i; j<g_hparams.ray_num; j+=CTA_SIZE)
            response[j] = g_hparams.max_range;

    // printf("---------##### Task RECEIVED\n");
    // 每次处理CTASize个线条
    for(int batchIdx=0; batchIdx<numLines; batchIdx+=CTA_SIZE)
    {
        // printf("[---★ NEW LOOP ★---] READ LINE FROM %d~%d.\n", batchIdx, batchIdx+CTA_SIZE);
        for(int i=0; i<CTA_SIZE; i++) {
            if(batchIdx + i >= numLines) {
                // printf("[READ] REACH TO THE END: %d.\n", batchIdx+i);
                break;
            }
            // 读取CTASize个线条，计算可见性，存入buf
            float2 lb = make_float2(line_begins[batchIdx+i].x-pose.x, line_begins[batchIdx+i].y-pose.y);
            float2 le = make_float2(line_ends[batchIdx+i].x-pose.x, line_ends[batchIdx+i].y-pose.y);
            bool visibility = prepareLine(lb, le, g_hparams.max_range);
            if(visibility) {        // TODO. compression 用Atomic还是用Exclusive Scan?
                lineBuf[lineBufWrite++ % LINE_BUF_SIZE] = batchIdx + i;
            }
        }

        // printf("[READ] TOTAL VISIBILITY: %d\n", lineBufWrite-lineBufRead);

        // 累计足够256个，或者剩余不足256个
        if((lineBufWrite - lineBufRead < CTA_SIZE) && (numLines - batchIdx >= CTA_SIZE))
        {
            // printf("[READ] Insufficient LINE: %d, Remain: %d, GO ON READING!\n", lineBufWrite-lineBufRead, numLines-batchIdx);
            continue;
        }

        // 计算TILE线段覆盖，每个线程负责一条线段
        // uint32_t bitMat[ctaSize];                       // LINE-TILE位矩阵
        // uint32_t tileTriggerMask=0;                     // 连续或标志位，记录哪些tile出现过
        // uint32_t tileRepMask=0;                         // 与标志位，检查哪些tile的直线数量>1

        // std::vector<int> tileLineCnt(ctaSize);
        // std::vector<int> numLineFrags(ctaSize);
        // int totalFragments = 0;
        //     // blockWide-Sum, block Scan 取最后一个
        //     totalFragments += numLineFrags[i];

        //     int s_tile = s_grid / TILE_SIZE;
        //     int e_tile = e_grid / TILE_SIZE;

        //     // TODO. 处理e_tile < s_tile的情况
        //     uint32_t mask = genMask(s_tile, e_tile);
        //     bitMat[i] = mask;
        //     tileLineCnt[i] = __builtin_popcount(mask);

        //     // block-wide操作
        //     tileTriggerMask |= mask;
        //     tileRepMask |= mask & tileTriggerMask;
        // }

        // ---------------- 方案一：不分tile，针对每一条线段的占用格数计算前缀和发射数

        for(int i=0; i<CTA_SIZE; i++)
        {
            if (lineBufRead + i >= lineBufWrite)
                break;

            // 计算线程的数据索引
            int lineBufIdx = (lineBufRead + i) % LINE_BUF_SIZE;
            int lineIdx = lineBuf[lineBufIdx];
            float2 lb = make_float2(line_begins[lineIdx].x-pose.x, line_begins[lineIdx].y-pose.y);
            float2 le = make_float2(line_ends[lineIdx].x-pose.x, line_ends[lineIdx].y-pose.y);

            // 计算起始和终止栅格
            auto s_angle = vec_atan2_0_360(lb);
            auto e_angle = vec_atan2_0_360(le);
            int s_grid = s_angle * g_hparams.resolu_inv;
            int e_grid = e_angle * g_hparams.resolu_inv;

            // 计算相交射线数
            int frag = e_grid-s_grid;
            frag += (e_grid < s_grid) ? g_hparams.ray_num : 0;

            // 相交射线数 >0 时存入frLineFrag
            if(frag > 0) {
                frLineFrag[frBufWrite % FR_BUF_SIZE] = frag;
                frLineBuf[(frBufWrite++) % FR_BUF_SIZE] =
                    std::make_tuple(lineIdx, s_grid, e_grid);
            }
        }

        // printf("[SEPERATE] FRLINE VALID: %d\n", frBufWrite-frBufRead);

        // 发射
        while(frBufWrite-frBufRead > 0)
        {
            // decompress，加载512个frag
            int oldfragRead = frBufRead;
            while(frBufRead != frBufWrite && fragWrite - fragRead < 512)
            {
                for(int j=0; j<frLineFrag[frBufRead]; j++)
                    fragBuf[fragWrite++ % FRAG_BUF_SIZE] = {frBufRead, j};
                frBufRead++;
            }
            // printf("[EMIT] LOAD LINE {%d-%d}, TOTAL_FRAG: %d \n", oldfragRead, frBufRead, fragWrite-fragRead);

            // 填充
            for(int i=0; i<CTA_SIZE*4; i+=4)
            {
                if(fragRead + i >= fragWrite) break;
                for(int j=0; j<4; j++)
                {
                    int fragBufIdx = fragRead + i + j;
                    if(fragBufIdx >= fragWrite)
                        break;
                    const auto& [frLineIdx, frag] = fragBuf[fragBufIdx % FRAG_BUF_SIZE];
                    const auto& [lineIdx, s_grid, e_grid] = frLineBuf[frLineIdx];
                    float2 lb = make_float2(line_begins[lineIdx].x-pose.x, line_begins[lineIdx].y-pose.y);
                    float2 le = make_float2(line_ends[lineIdx].x-pose.x, line_ends[lineIdx].y-pose.y);
                    int grid = (s_grid + frag + 1) % g_hparams.ray_num;
                    response[grid] = min(response[grid], getR(lb, le, grid*g_hparams.resolu));
                }
            }

            fragRead = min(fragRead+CTA_SIZE*4, fragWrite);
        }

        lineBufRead = min(lineBufRead+CTA_SIZE, lineBufWrite);


    // scan一遍line，累计512个写入任务，而后CTA中每个线程分配4个写入任务

    // 问题在于TILE可能一般情况下比较少，32~128线：约1~4 Tiles
    // 每个Warp分配的工作可能无法达到很高

    //
    // TODO. 深度预裁剪


    //-------------- 每Warp负责一个Tile --------------
    // for(int warpId=0; warpId<warpCnt; warpId++) {
    //     for(int i=0; i<32; i++) {
    //         // warpId / TILE_NUM
    //         // tiles[];
    //     }
    // }

    // 得到每TILE的工作负载：按TILE划分的队列

    // inclusive scan 得到每个线段的写入Index范围

    // 计算当前warp的累计frag数量，如果大于32，进入发射阶段

    // 取一部分线段的写入范围frag，该值是在之前scan过程中保存的

    // for 每个线程取一个线段的fragidx，小于32的线段进行标记

    // 发射阶段，计算当前线程应该写入哪条线段的哪个射线交点
    // 发射标志位[0 0 0 0 0 1 0 0 0 1 0 0 0 1]
    // 线段Id = popc(mask & lane mask)
    // fragId = (inWarpIdx - fragRead) - 上一个三角形的frag数量
    // frag = fragId + 线段start的索引

    }

    return response;
}

template<typename T>
sf::Vector2<T> convertPoint(sf::RenderWindow &window, sf::Vector2<T> v)
{
    auto size = window.getView().getSize();
    return sf::Vector2<T>(v.x * scaling + offset, size.y - (v.y*scaling + offset));
}

template<typename T>
sf::Vector2<T> invConvertPoint(sf::RenderWindow &window, sf::Vector2<T> v)
{
    auto size = window.getSize();

    printf("View.x:%d, View.y:%d\n", size.x, size.y);

    return sf::Vector2<T>((v.x-offset)/(size.x/800.f) / scaling, ((size.y-v.y)-offset)/(size.y/600.f) /scaling);
}


void draw(const std::vector<float2>& lbegins, const std::vector<float2>& lends) {
    sf::RenderWindow window(sf::VideoMode(800, 600), "SFML Draw Lines");

    float2 startPoint(2, 2); // 起点，初始为 {2, 2}
    std::vector<std::pair<float2, float2>> ray_shape;

    // 主循环
    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed) {
                window.close();
            }

            // 鼠标点击事件
            if (event.type == sf::Event::MouseButtonPressed && event.mouseButton.button == sf::Mouse::Left) {
                // 获取鼠标点击位置
                sf::Vector2i mousePos = invConvertPoint(window, sf::Mouse::getPosition(window));

                // 转换为逻辑坐标
                startPoint.x = mousePos.x;
                startPoint.y = mousePos.y;

                // 打印鼠标点击位置（调试用）
                std::cout << "Mouse clicked at: (" << startPoint.x << ", " << startPoint.y << ")\n";

                ray_shape.clear();
                std::vector<float> rays = cpuTest2(
                    lbegins.size(),
                    make_float3(startPoint.x, startPoint.y, 0),
                    lbegins,
                    lends);
                for (int i = 0; i < rays.size(); i++) {
                    float angle = i * g_hparams.resolu;
                    float2 endPoint = make_float2(rays[i] * cosf(angle) + startPoint.x, rays[i] * sinf(angle) + startPoint.y);
                    ray_shape.push_back({ startPoint, endPoint});
                }

            }
        }

        // 清空窗口
        window.clear(sf::Color::Black);

        // 绘制所有直线
        for (int i=0; i<lbegins.size(); i++) {
            // 创建一个 sf::VertexArray 用于绘制线段
            sf::VertexArray lineShape(sf::Lines, 2);

            // 设置第一个点
            lineShape[0].position = convertPoint(window, sf::Vector2f(lbegins[i].x, lbegins[i].y));
            lineShape[0].color = sf::Color::Red;

            // 设置第二个点
            lineShape[1].position = convertPoint(window, sf::Vector2f(lends[i].x, lends[i].y));
            lineShape[1].color = sf::Color::White;

            // 绘制线段
            window.draw(lineShape);
        }

        for (const auto& line : ray_shape) {
            // 创建一个 sf::VertexArray 用于绘制线段
            sf::VertexArray lineShape(sf::Lines, 2);

            // 设置第一个点
            lineShape[0].position = convertPoint(window, sf::Vector2f(line.first.x, line.first.y));
            lineShape[0].color = sf::Color::Red;

            // 设置第二个点
            lineShape[1].position = convertPoint(window, sf::Vector2f(line.second.x, line.second.y));
            lineShape[1].color = sf::Color::White;

            // 绘制线段
            window.draw(lineShape);
        }

        // 显示窗口内容
        window.display();
    }
}



int main()
{
    google::InstallFailureSignalHandler();

    auto map_generator = std::make_unique<map_gen::CellularAutomataGenerator>(MAP_WIDTH, MAP_HEIGHT);
    // auto map_generator = std::make_unique<MessyBSPGenerator>(MAP_WIDTH, MAP_HEIGHT);
    map_generator->generate();
    auto map = map_generator->getMap();
    auto shapes = map_gen::processGridmap(map, GRID_SIZE);

    std::vector<float2> lbegins, lends;
    std::for_each(shapes.begin(), shapes.end(), [&lbegins, &lends](const auto& polygons){
        for(int pgi=0; pgi<polygons.size(); pgi++) {
            auto pg = polygons[pgi];
            pg.push_back(pg.front());
            if(polygons.size() != 2) continue;
            for(int i=0; i<pg.size()-1; i++) {
                auto lb = make_float2(pg[i].x, pg[i].y);
                auto le = make_float2(pg[i+1].x, pg[i+1].y);
                if(polygons.size() == 2) std::swap(lb, le);
                lbegins.push_back(lb);
                lends.push_back(le);
            }
        }

    });

    // lines =
    //     {
    //         {{1,1},{1,11}},
    //         {{1,11},{11,11}},
    //         {{11,11},{11,1}},
    //         {{11,1},{1,1}},
    //     };

    draw(lbegins, lends);

    // std::copy(rays.begin(), rays.end(), std::ostream_iterator<float>(std::cout, ", "));
}