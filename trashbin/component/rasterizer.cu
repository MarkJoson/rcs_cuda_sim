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

constexpr float SCALING = 8;
constexpr float OFFSET = 20;

constexpr float PI = M_PI;

#define LIDAR_LINES_LOG2    (10)         // 128lines
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
#define TOTAL_EMITION       (EMIT_PER_THREAD * CTA_SIZE)


#ifdef __DRIVER_TYPES_H__
static inline const char *_cudaGetErrorEnum(cudaError_t error) {
  return cudaGetErrorName(error);
}
#endif

template <typename T>
void check(T result, char const *const func, const char *const file,
           int const line) {
  if (result) {
    fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
            static_cast<unsigned int>(result), _cudaGetErrorEnum(result), func);
    throw std::exception();
    // exit(EXIT_FAILURE);
  }
}

#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)


// thrust::device_vector<bool> lidar_inst_state;
// thrust::device_vector<int> line_count;

struct RCSParam
{
    // int             *line_cnt;
    // float2          *line_s;
    // float2          *line_e;

    // float2          *pos;
    // bool            *inst_enabled;
    float           max_range;
    float           resolu_inv;
    float           resolu;
    int             ray_num;
};

RCSParam g_hparams {
    .max_range = 20,
    .resolu_inv = LIDAR_LINES / (2*M_PI),
    .resolu = (2*M_PI) / LIDAR_LINES,
    .ray_num = LIDAR_LINES
};


__constant__ __device__ RCSParam g_params;


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

// TODO. Texture 1D
__global__ void rasterKernel(
    int                         numLines,
    const float3 __restrict__*  poses,
    const float2 __restrict__*  line_begins,
    const float2 __restrict__*  line_ends,
             int             *  lidar_response
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
    for(int i=tid; i<g_params.ray_num; i+=CTA_SIZE)
        s_lidarResponse[i] = g_params.max_range*100;

    // if(tid == 0) printf("---------##### Task RECEIVED\n");

    for(;;)
    {
        /********* Load Lines to Buffer *********/

        while(lineBufWrite-lineBufRead < CTA_SIZE && totalLineRead < numLines)
        {
            // if(tid == 0) printf("[READ] LINE RANGE: %d~%d.\n", totalLineRead, totalLineRead+CTA_SIZE);
            uint32_t visibility = false;
            int lineIdx = totalLineRead+tid;
            if(lineIdx < numLines)
            {
                float2 lb = make_float2(line_begins[lineIdx].x-pose.x, line_begins[lineIdx].y-pose.y);
                float2 le = make_float2(line_ends[lineIdx].x-pose.x, line_ends[lineIdx].y-pose.y);
                visibility = prepareLine(lb, le, g_params.max_range);
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
                s_grid = s_angle * g_params.resolu_inv;
                int e_grid = e_angle * g_params.resolu_inv;

                frag = (e_grid-s_grid) + ((e_grid < s_grid) ? g_params.ray_num : 0);
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
        if(frLineBufWrite-frLineBufRead < CTA_SIZE && (lineBufRead < lineBufWrite || totalLineRead < numLines)) {
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
                    int grid = (s_grid + fragIdx + 1) % g_params.ray_num;
                    uint16_t response = getR(lb, le, grid*g_params.resolu) * 1024;        // 10位定点小数表示，最大距离64m
                    uint32_t resp_idx = response << 16 | lineIdx & 0xffff;
                    atomicMin_block(&s_lidarResponse[grid], resp_idx);
                }
            }
            __syncthreads();
        } while(frLineBufWrite != frLineBufRead && totalLineRead >= numLines);       // 继续的条件：已经没有办法读取更多的frag线段，则需要将剩余的frlineBufWrite处理完

        // if(tid == 0) printf("[EMIT] FINISHED!\n");

        // 全部线段已经处理完
        if(totalLineRead >= numLines) break;
    }

    for(int i=tid; i<g_params.ray_num; i+=CTA_SIZE)
        lidar_response[i] = s_lidarResponse[i];
}




std::vector<float> rasterGPU(int numLines, float3 pose, const std::vector<float2> &line_begins, const std::vector<float2> &line_ends)
{
    // float3 poses[1] {pose};
    thrust::device_vector<float3> poses(1);
    poses[0] = pose;

    thrust::device_vector<int> lidar_response(LIDAR_LINES);
    thrust::device_vector<float2> lbs = line_begins;
    thrust::device_vector<float2> les = line_ends;

    rasterKernel<<<1, CTA_SIZE>>>(numLines, poses.data().get(), lbs.data().get(), les.data().get(), lidar_response.data().get());
    checkCudaErrors(cudaDeviceSynchronize());

    return std::vector<float>(lidar_response.begin(), lidar_response.end());
}

std::vector<float> rasterCPU(int numLines, float3 pose, const std::vector<float2> &line_begins, const std::vector<float2> &line_ends)
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
            response[j] = g_hparams.max_range*100;

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
            // int oldfragRead = frBufRead;
            while(frBufRead != frBufWrite && fragWrite - fragRead < TOTAL_EMITION)
            {
                for(int j=0; j<frLineFrag[frBufRead]; j++)
                    fragBuf[fragWrite++ % FRAG_BUF_SIZE] = {frBufRead, j};
                frBufRead++;
            }
            // printf("[EMIT] LOAD LINE {%d-%d}, TOTAL_FRAG: %d \n", oldfragRead, frBufRead, fragWrite-fragRead);

            // 填充
            for(int i=0; i<TOTAL_EMITION; i+=4)
            {
                if(fragRead + i >= fragWrite) break;
                for(int j=0; j<EMIT_PER_THREAD; j++)
                {
                    int fragBufIdx = fragRead + i + j;
                    if(fragBufIdx >= fragWrite)
                        break;
                    const auto& [frLineIdx, frag] = fragBuf[fragBufIdx % FRAG_BUF_SIZE];
                    const auto& [lineIdx, s_grid, e_grid] = frLineBuf[frLineIdx];
                    float2 lb = make_float2(line_begins[lineIdx].x-pose.x, line_begins[lineIdx].y-pose.y);
                    float2 le = make_float2(line_ends[lineIdx].x-pose.x, line_ends[lineIdx].y-pose.y);
                    int grid = (s_grid + frag + 1) % g_hparams.ray_num;
                    response[grid] = min(response[grid], getR(lb, le, grid*g_hparams.resolu)*100);
                }
            }

            fragRead = min(fragRead+TOTAL_EMITION, fragWrite);
        }

        lineBufRead = min(lineBufRead+CTA_SIZE, lineBufWrite);

    // TODO. 深度预裁剪
    }

    return response;
}

template<typename T>
sf::Vector2<T> convertPoint(sf::RenderWindow &window, sf::Vector2<T> v)
{
    auto size = window.getView().getSize();
    return sf::Vector2<T>(v.x * SCALING + OFFSET, size.y - (v.y*SCALING + OFFSET));
}

template<typename T>
sf::Vector2<T> invConvertPoint(sf::RenderWindow &window, sf::Vector2<T> v)
{
    auto size = window.getSize();
    return sf::Vector2<T>((v.x-OFFSET)/(size.x/800.f) / SCALING, ((size.y-v.y)-OFFSET)/(size.y/600.f) /SCALING);
}


void draw(const std::vector<float2>& lbegins, const std::vector<float2>& lends) {
    sf::RenderWindow window(sf::VideoMode(800, 600), "SFML Draw Lines");
    window.setVerticalSyncEnabled(true); // call it once, after creating the window

    float2 startPoint = make_float2(2, 2); // 起点，初始为 {2, 2}
    std::vector<std::pair<float2, float2>> ray_shape;

    sf::Font font;
    if (!font.loadFromFile("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf")) { // 确保有合适的字体文件
        std::cerr << "Failed to load font!" << std::endl;
        return;
    }

    // 文本对象
    sf::Text mousePositionText;
    mousePositionText.setFont(font);
    mousePositionText.setCharacterSize(20); // 设置字体大小
    mousePositionText.setFillColor(sf::Color::Green); // 设置文字颜色
    mousePositionText.setPosition(10, 10); // 设置文字位置

    // 主循环
    while (window.isOpen()) {
        sf::Event event;

        sf::Vector2i mousePos = sf::Mouse::getPosition(window);
        sf::Vector2f mouseWorldPos = invConvertPoint<float>(window, {mousePos.x, mousePos.y});

        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed) {
                window.close();
            }

            // 鼠标点击事件
            if (event.type == sf::Event::MouseButtonPressed && event.mouseButton.button == sf::Mouse::Left) {
                // 获取鼠标点击位置


                // 转换为逻辑坐标
                startPoint.x = mouseWorldPos.x;
                startPoint.y = mouseWorldPos.y;

                // 打印鼠标点击位置（调试用）
                std::cout << "Mouse clicked at: (" << startPoint.x << ", " << startPoint.y << ")\n";

                ray_shape.clear();
                std::vector<float> rays = rasterGPU(
                    lbegins.size(),
                    make_float3(startPoint.x, startPoint.y, 0),
                    lbegins,
                    lends);
                for (size_t i = 0; i < rays.size(); i++) {
                    float angle = i * g_hparams.resolu;
                    float r = rays[i] / 100.f;
                    float2 endPoint = make_float2(r * cosf(angle) + startPoint.x, r * sinf(angle) + startPoint.y);
                    ray_shape.push_back({ startPoint, endPoint});
                }

            }
        }

        // 获取鼠标位置并更新文本内容
        mousePositionText.setString("Mouse: (" + std::to_string(mouseWorldPos.x) + ", " + std::to_string(mouseWorldPos.y) + ")");

        // 清空窗口
        window.clear(sf::Color::Black);

        // 绘制所有直线
        for (size_t i=0; i<lbegins.size(); i++) {
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
            lineShape[0].color = sf::Color::Blue;

            // 设置第二个点
            lineShape[1].position = convertPoint(window, sf::Vector2f(line.second.x, line.second.y));
            lineShape[1].color = sf::Color::White;

            // 绘制线段
            window.draw(lineShape);
        }

        // 绘制鼠标位置文本
        window.draw(mousePositionText);
        // 显示窗口内容
        window.display();
    }
}



int main()
{
    google::InstallFailureSignalHandler();

    checkCudaErrors(cudaMemcpyToSymbol(g_params, &g_hparams, sizeof(g_hparams)));

    auto map_generator = std::make_unique<map_gen::CellularAutomataGenerator>(MAP_WIDTH, MAP_HEIGHT);
    // auto map_generator = std::make_unique<MessyBSPGenerator>(MAP_WIDTH, MAP_HEIGHT);
    map_generator->generate();
    auto map = map_generator->getMap();
    auto shapes = map_gen::processGridmap(map, GRID_SIZE);

    std::vector<float2> lbegins, lends;
    std::for_each(shapes.begin(), shapes.end(), [&lbegins, &lends](const auto& polygons){
        for(size_t pgi=0; pgi<polygons.size(); pgi++) {
            auto pg = polygons[pgi];
            pg.push_back(pg.front());
            // if(polygons.size() != 2 || (polygons.size() == 2 && pgi == 1)) continue;
            for(size_t i=0; i<pg.size()-1; i++) {
                auto lb = make_float2(pg[i].x, pg[i].y);
                auto le = make_float2(pg[i+1].x, pg[i+1].y);
                // if(polygons.size() == 2)
                std::swap(lb, le);
                lbegins.push_back(lb);
                lends.push_back(le);
            }
        }

    });

    // lbegins =
    //     {
    //         {1,1}  ,
    //         {1,11} ,
    //         {11,11},
    //         {11,1} ,
    //     };

    // lends =
    //     {
    //         {1,11},
    //         {11,11},
    //         {11,1},
    //         {1,1},
    //     };

    draw(lbegins, lends);

    // std::copy(rays.begin(), rays.end(), std::ostream_iterator<float>(std::cout, ", "));
}