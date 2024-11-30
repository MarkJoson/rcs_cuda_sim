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

constexpr float PI = M_PI;

#define LIDAR_LINES_LOG2    (8)         // 256lines
#define LIDAR_LINES         (1<<LIDAR_LINES_LOG2)

#define TILE_SIZE_LOG2      (5)         // 32lines
#define TILE_SIZE           (1<<TILE_SIZE_LOG2)

#define TILE_NUM            (1 << (LIDAR_LINES_LOG2-TILE_SIZE_LOG2))

#define CTA_SIZE            (128)


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
    float           ray_num;
};

__constant__ __device__ RCSParam g_params;

#define RASTER_WARPS            (4)
// #define MAX_ENV_GRP


__global__ void raster(int line_count)
{
    // 判断可见性
    int mThrIdx = threadIdx.x;
    int warpIdx = threadIdx.y;

    constexpr int ctaSize = RASTER_WARPS * 32;

    // 环境中的实例Id
    int instIdx = blockIdx.x;
    // 环境组中的环境Id
    int envIdx = blockIdx.y;
    // 环境组Id
    int eGrpIdx = blockIdx.z;

    /**********************
     ****** 填充初始值 ******
     **********************/

    // 检查激光雷达是否启用，判断 inst_enabled 是否为true
    if(g_params.inst_enabled[instIdx] == false)
        return;

    /**************************************************
     ****** 计算可见性，累计一定数量后进入下一个阶段 ******
     **************************************************/
    __shared__ volatile uint32_t s_buf[RASTER_WARPS];
    __shared__ volatile uint32_t s_visList[RASTER_WARPS];

    int numLines = g_params.line_cnt[envIdx];
    auto instPos = g_params.pos[instIdx];

    for(int i=mThrIdx + 32 * warpIdx; i<numLines; i+=ctaSize)       // 一次读取32*numWarps条直线
    {
        auto ls = g_params.line_s[i];
        auto le = g_params.line_e[i];
        auto osx = ls.x - instPos.x;
        auto osy = ls.y - instPos.y;
        auto oex = le.x - instPos.x;
        auto oey = le.y - instPos.y;
        bool cdot = osx*oey - osy*oex;

        float len = sqrtf((oex-osx)*(oex-osx)+(oey-osy)*(oey-osy));
        float dist = fabs(cdot)/len;

        bool result = dist < g_params.max_range && cdot<0;
        // s_visList[warpIdx] = __ballot(result);                          // 当前warp的投票结果

        auto s_angle = atan2f(osy, osx) + osy > 0 ? 0:PI;
        auto e_angle = atan2f(oey, oex) + oey > 0 ? 0:PI;
        int s_grid = s_angle * g_params.resolu_inv;
        int e_grid = e_angle * g_params.resolu_inv;

        // 生成mask
        int mask = ((1<<e_grid)-1) & ((1<<s_grid)-1);
        int pop = __popc(mask);

        // int len = e_grid-s_grid;
        // len = len<0?len+g_params.ray_num:len;

        // 发射

    }

    // 读取thrIdx条直线，变换，计算可见性
    using BlockRunLengthDecodeT = cub::BlockRunLengthDecode<int, CTA_SIZE, 1, 1>;
    __shared__ typename BlockRunLengthDecodeT::TempStorage temp_storage;
    // 流压缩到紧密数组，计算视角，过滤没有与射线相交的线段，二次存入队列

    // decompress队列，发射，深度测试



    /**************************************
     ******   分箱，分成32ray一组？   *******
     **************************************/


    /**********************
     *****   光栅化   ******
     **********************/

}


RCSParam g_hparams {
    .max_range = 6,
    .resolu_inv = 128 / (2*M_PI),
    .ray_num = 128
};



void cpuLineTest(float2 line_s, float2 line_e, float2 instPos)
{

    // inclusive scan 得到每个线段的写入Index范围

    // 计算当前warp的累计frag数量，如果大于32，进入发射阶段

    // 取一部分线段的写入范围frag，该值是在之前scan过程中保存的
/**
 * for 每个线程取一个线段的fragidx，小于32的线段进行标记
 */

    // 发射阶段，计算当前线程应该写入哪条线段的哪个射线交点
    // 发射标志位[0 0 0 0 0 1 0 0 0 1 0 0 0 1]
    // 线段Id = popc(mask & lane mask)
    // fragId = (inWarpIdx - fragRead) - 上一个三角形的frag数量
    // frag = fragId + 线段start的索引
}


bool prepareLine(glm::vec2 vs, glm::vec2 ve)
{
    bool cdot = vs.x*ve.y - vs.y*ve.x;

    float len = glm::length(ve-vs);
    float dist = fabs(cdot)/len;

    return dist < g_hparams.max_range && cdot<0;
}

__forceinline__ float vec_atan2_0_360(glm::vec2 vec)
{
    return atan2f(vec.y, vec.x) + (vec.y > 0 ? 0:PI);
}

__forceinline__ uint32_t genMask(int start, int end)
{
    return ((1<<end)-1) & ~((1<<start)-1);
    // int pop = __builtin_popcount(mask);
}

__forceinline__ float getR(glm::vec2 lb, glm::vec2 le, float theta)
{
    float sin,cos;
    sincosf(theta, &sin, &cos);
    auto dvec = le - lb;
    return (lb.y*dvec.x-lb.x*dvec.y)/(dvec.x*sin-dvec.y*cos);
}

#define FR_BUF_SIZE             (128*2)

void cpuTest2(int numLines, std::pair<float2, float2> lines[], float2 inst_pos)
{
    int ctaSize = 128;
    int warpCnt = ctaSize / 32;
    glm::vec2 pos = glm::vec2(inst_pos.x, inst_pos.y);

    // std::vector<int> lineFrag();
    std::vector<int> lineBuf(ctaSize*2);
    int lineBufRead=0, lineBufWrite=0;          // Read==Write空，(Write+1)%All==Read满

    std::vector<std::tuple<int, int, int>> frLineBuf(FR_BUF_SIZE);
    std::vector<int> frLineFrag(FR_BUF_SIZE);
    int frBufRead=0, frBufWrite=0;

    std::vector<float> response(LIDAR_LINES);

    // TODO. 按线段长度排序

    // 每次处理CTASize个线条
    for(int batchIdx=0; batchIdx<numLines; batchIdx+=ctaSize)
    {
        // 读取CTASize个线条
        std::vector<glm::vec2> lbegin(ctaSize), lend(ctaSize);

        for(int i=0; i<ctaSize; i++) {
            if(batchIdx + i >= numLines) break;
            lbegin[i] = glm::vec2(lines[batchIdx+i].first.x, lines[batchIdx+i].first.y) - pos;
            lend[i] = glm::vec2(lines[batchIdx+i].second.x, lines[batchIdx+i].second.y) - pos;
            bool visibility = prepareLine(lbegin[i], lend[i]);
            if(visibility) {        // TODO. compression 用Atomic还是用Exclusive Scan?
                lineBuf[lineBufWrite++] = i;
            }
        }

        // 累计足够256个，或者剩余不足256个
        if((lineBufWrite - lineBufRead < ctaSize) && (numLines - batchIdx >= ctaSize))
            continue;

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

        // ---------------- 方案一：不分tile，针对每一条线段的占用格数计算前缀和，并发射
        for(int i=0; i<ctaSize; i++)
        {
            if (lineBufRead + i >= lineBufWrite) {
                break;
            }

            // 计算线程的数据索引
            int lineIdx = (lineBufRead + i) % ctaSize;
            glm::vec2 vs = lbegin[lineIdx], ve = lend[lineIdx];

            auto s_angle = vec_atan2_0_360(vs);
            auto e_angle = vec_atan2_0_360(ve);
            int s_grid = s_angle * g_hparams.resolu_inv;
            int e_grid = e_angle * g_hparams.resolu_inv;

            int frag = e_grid-s_grid;
            frag += frag < 0 ? g_hparams.ray_num : 0;

            if(frag > 0) {
                frLineFrag[frBufWrite] = frag;
                frLineBuf[frBufWrite++] =
                    std::make_tuple(lineIdx, s_grid, e_grid);
            }
        }



        for(;;)
        {
            if(frBufWrite-frBufRead == 0)
                break;

            // 加载512个
            std::vector<std::pair<int,int>> fragBuf(512);
            int totalFrag = 0;
            while(frBufRead != frBufWrite)
            {
                // decompress
                for(int j=0; j<frLineFrag[frBufRead]; j++)
                    fragBuf[totalFrag] = {frBufRead,j};
                frBufRead++;
                if(totalFrag >= 512) break;
            }

            for(int i=0; i<ctaSize; i++)
            {
                if(i >= totalFrag) break;
                const auto& [frLineIdx, frag] = fragBuf[i];
                const auto& [lineDataIdx, s_grid, e_grid] = frLineBuf[frLineIdx];
                int grid = s_grid + frag;
                response[grid] = getR(lbegin[lineDataIdx], lend[lineDataIdx], grid*g_hparams.resolu);
            }
        }

        lineBufRead= min(lineBufRead+ctaSize, lineBufWrite);


        // scan一遍line，累计512个写入任务，而后CTA中每个线程分配4个写入任务



        // std::vector<int> scanFrags;     // TODO. 依然是使用队列
        // std::inclusive_scan(numLineFrags.begin(), numLineFrags.end(), scanFrags.begin(), std::plus<int>());




        // 问题在于TILE可能一般情况下比较少，32~128线：约1~4 Tiles
        // 每个Warp分配的工作可能无法达到很高

        //
        // TODO. 深度预裁剪


        //-------------- 每Warp负责一个Tile --------------
        for(int warpId=0; warpId<warpCnt; warpId++) {
            for(int i=0; i<32; i++) {
                // warpId / TILE_NUM
                // tiles[];
            }
        }

        // 得到每TILE的工作负载：按TILE划分的队列

    }
}


void draw(const std::vector<std::pair<float2, float2>> &lines)
{
    sf::RenderWindow window(sf::VideoMode(800, 600), "SFML Draw Lines");

    // 主循环
    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed) {
                window.close();
            }
        }

        // 清空窗口
        window.clear(sf::Color::Black);

        // 绘制所有直线
        for (const auto& line : lines) {
            // 创建一个 sf::VertexArray 用于绘制线段
            sf::VertexArray lineShape(sf::Lines, 2);

            // 设置第一个点
            lineShape[0].position = sf::Vector2f(line.first.x*5+10, line.first.y*5+10);
            lineShape[0].color = sf::Color::Red;

            // 设置第二个点
            lineShape[1].position = sf::Vector2f(line.second.x*5+10, line.second.y*5+10);
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
    auto polygons = shapes | std::views::join;

    std::vector<std::pair<float2, float2>> lines;
    std::for_each(polygons.begin(), polygons.end(), [&lines](const auto &pg){
        for(int i=0; i<pg.size()-1; i++)
            lines.push_back({make_float2(pg[i].x, pg[i].y), make_float2(pg[i+1].x, pg[i+1].y)});
        lines.push_back({make_float2(pg.rbegin()->x, pg.rbegin()->y), make_float2(pg[0].x, pg[0].y)});
    });

    printf("total lines:%ld\n", lines.size());

    // draw(lines);

    cpuLineTest(lines[0].first, lines[0].second, make_float2(0,0));


    // 使用 C++20 的 std::ranges 和 std::views 展开到单层
    // auto flat_view = polygons | std::views::join; // 再展开第二层（vector<int>）




    // cpuLineTest()
    // std::for_each(flat_view.begin(), flat_view.end(), [](const auto &p){
    //     std::cout << p.first << ", " << p.second << std::endl;
    // });
}