
#include "mapgen_generate.h"
#include "mapgen_postprocess.h"

#include <iostream>
#include <numeric>
#include <algorithm>
#include <ranges>
#include <string_view>
#include <deque>
#include <ranges>
#include <list>
#include <unordered_map>
#include <unordered_set>
#include <boost/range/join.hpp>

using namespace std::literals;

// TODO. 目前的Point是int类型
namespace map_gen
{

// 计算单点Hash值
template<typename T>
uint64_t ptHash(const Point<T> &pt) { return std::hash<Point<T>>()(pt); }

// 提取所有的独立联通区域
void extractAllRegions(const Array2d &map, int width, int height, ArrayPoly<int> &regions)
{
    std::vector<std::vector<bool>> visited(width, std::vector<bool>(height, 0));

    // 标记空旷的区域为visited=1，无需遍历
    for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++)
            visited[x][y] = 1 - map[x][y];

    int vx = -1, vy = 0;
    while (1)
    {
        int ckpt_x = 0, ckpt_y = 0;
        // 找到visited_list中一个非0的点
        for (ckpt_y = vy; ckpt_y < height; ckpt_y++)
        {
            // 从上一次的vx继续开始遍历
            for (ckpt_x = vx + 1; ckpt_x < width; ckpt_x++)
                if (visited[ckpt_x][ckpt_y] == 0)
                    break;
            if (ckpt_x != width)
                break;
            vx = -1;
        }
        // 记录当前找到非零点的坐标
        vx = ckpt_x, vy = ckpt_y;

        // 检查完所有的点就退出
        if (ckpt_y == height)
            break;

        // 当前区域的点集
        ArrayPt<int> region_pts;
        // 从当前点出发，进行深度优先遍历，使用栈表
        ArrayPt<int> ptstack;
        // 标记初始点
        visited[ckpt_x][ckpt_y] = 1;
        ptstack.push_back({ckpt_x, ckpt_y});

        while (!ptstack.empty())
        {
            auto [x, y] = ptstack.back();
            region_pts.push_back({x, y});
            ptstack.pop_back();

            // 将邻居节点加入候选列表
            static constexpr std::array<std::pair<int, int>, 4> neighbors4{{{-1, 0}, {0, -1}, {0, 1}, {1, 0}}};
            for (const auto &[dx, dy] : neighbors4)
            {
                if (x + dx < 0 || x + dx >= width || y + dy < 0 || y + dy >= height)
                    continue;
                if (!visited[x + dx][y + dy] && map[x + dx][y + dy] == 1)
                {
                    visited[x + dx][y + dy] = 1;
                    ptstack.push_back({x + dx, y + dy});
                }
            }
        }
        // 加入regions清单
        regions.push_back(region_pts);
    }
}

// 提取region的边
ArrayEdge<float> extractRegionEdge(const Array2d &map, const ArrayPt<int> &region_pts, int width, int height, float grid_size)
{
    static constexpr std::array<Point<int>, 4> neighbors4{{{-1, 0}, {0, -1}, {1, 0}, {0, 1}}};
    const std::array<Edge<float>, 4> neighbor_edges{{
        {{0, grid_size}, {0, 0}},
        {{0, 0}, {grid_size, 0}},
        {{grid_size, 0}, {grid_size, grid_size}},
        {{grid_size, grid_size}, {0, grid_size}},
    }};

    ArrayEdge<float> edge_out;
    int edge_pixel = 0, edge_cnt = 0;
    for (auto &[x, y] : region_pts)
    {
        int n_degree = 0, n_flag = 0;

        for (int i = 0; i < neighbors4.size(); i++)
        {
            const auto &[dx, dy] = neighbors4[i];
            if (x + dx < 0 || x + dx >= width || y + dy < 0 || y + dy >= height)
                continue;
            if (map[x + dx][y + dy] == 1)
            {
                n_degree += 1;
                n_flag |= 1 << i;
                edge_pixel++;
            }
        }

        if (n_degree == 4)
            continue;

        for (int i = 0; i < 4; i++)
        {
            if (!(n_flag & 1 << i))
            {
                edge_cnt++;
                auto &[p1, p2] = neighbor_edges[i];
                edge_out.push_back({{x * grid_size + p1.x, y * grid_size + p1.y}, {x * grid_size + p2.x, y * grid_size + p2.y}});
            }
        }
    }

    return edge_out;
}

// 将边集合拆分成顺时针的多边形
ArrayPoly<float> mergeEdgeToShape(const ArrayEdge<float> &edges)
{
    ArrayPoly<float> polys;

    std::unordered_map<PtHash, std::list<int>> map_spt_idx;
    // std::unordered_map<PtHash, std::list<int>> map_ept_idx;

    for (int i = 0; i < edges.size(); i++)
    {
        const auto &[spt, ept] = edges[i];
        PtHash hash_spt = ptHash(spt), hash_ept = ptHash(ept);

        if (map_spt_idx.find(hash_spt) == map_spt_idx.end())
            map_spt_idx[hash_spt] = std::list<int>();
        // if(map_ept_idx.find(hash_ept)==map_ept_idx.end()) map_ept_idx[hash_ept] = std::list<int>();

        map_spt_idx[hash_spt].push_back(i);
        // map_ept_idx[hash_ept].push_back(i);

        // {
        //     auto old = map_spt_idx.find(hash_spt)->second;
        //     printf("collide!! Edge %d:(%d,%d)->(%d,%d)\n", i, spt.x, spt.y, ept.x, ept.y);
        //     printf("\t with Edge %d:(%d,%d)->(%d,%d)\n", old, edges[old].start.x, edges[old].start.y, edges[old].end.x, edges[old].end.y);
        //     // throw std::exception();
        // }
        // map_spt_idx[hash_spt] = i;
        // map_ept_idx[hash_ept] = i;
    }

    std::unordered_set<Edge<float>> edge_set(edges.begin(), edges.end());

    while (!edge_set.empty())
    {
        Poly<float> poly;

        // 从随机一个边的起点开始
        Pointf pt = edge_set.begin()->start;
        while (true)
        {
            // 寻找起始点spt点对应的边，获得线段的end
            auto spt_vec_iter = map_spt_idx.find(ptHash(pt));
            if (spt_vec_iter == map_spt_idx.end())
            {
                // 如果当前的曲线未闭合，连接head形成多边形闭合。
                if (!(pt == poly.front()))
                {
                    printf("Polygon is not closed!!!\n");
                    poly.push_back(poly[0]);
                }
                break;
            }

            poly.push_back(pt);

            auto &spt_vec = spt_vec_iter->second;
            int edge_id = spt_vec.front();
            spt_vec.pop_front();
            if (spt_vec.empty())
                map_spt_idx.erase(ptHash(pt));

            const Edge<float> &edge = edges[edge_id];
            edge_set.erase(edge);

            // 如果点已经存在于点集
            pt = edge.end;
        }
        polys.push_back(poly);
    }
    return polys;
}

// 点到直线函数
float dist_to_line(const Pointf &l1, const Pointf &l2, const Pointf &op)
{
    if (abs(l2.x - l1.x) == 0 && abs(l2.y - l1.y) == 0)
        throw std::exception();
    float dist = l1.x * l2.y + l2.x * op.y + op.x * l1.y - l1.x * op.y - op.x * l2.y - l2.x * l1.y;
    dist /= sqrtf((l2.x - l1.x) * (l2.x - l1.x) + (l2.y - l1.y) * (l2.y - l1.y));
    return fabs(dist);
}

// 将点集精简
ArrayPt<float> douglasPeukcer(const ArrayPt<float> &pts, float grid_size)
{
    std::list<Pointf> ptlst; //(pts.begin(), pts.end());

    // !----- 粗精简，合并同一条直线上的点
    // 拼接首尾tail, head, ..., tail, head
    auto tht_arr = boost::join(std::array<Pointf, 1>({*std::prev(pts.end())}), boost::join(pts, std::array<Pointf, 1>{pts[0]}));
    for (auto pt_iter = tht_arr.begin() + 1; pt_iter != std::prev(tht_arr.end()); pt_iter++)
    {
        auto p1i = std::prev(pt_iter);
        auto p2i = std::next(pt_iter);
        // if(p1i==p2i) {p2i = pt_iter; continue;}
        // 如果不在直线上，则设置p1和p2到拐点，
        const auto &p1 = *p1i, &p2 = *p2i, &p3 = *pt_iter;
        if ((p2.y - p1.y) * (p3.x - p2.x) != (p3.y - p2.y) * (p2.x - p1.x))
        {
            ptlst.push_back(*pt_iter);
        }
    }

    // TODO. 计算每个点的曲率而后过滤

    // std::vector<std::pair<std::list<Pt>::iterator, std::list<Pt>::iterator>> stack;
    // // 将第一个和最后一个存起来
    // stack.push_back({ptlst.begin(), std::prev(ptlst.end())});
    // while(!stack.empty())
    // {
    //     auto [spi, epi] = stack.back();
    //     stack.pop_back();
    //     // 如果线段之间就差1个差距，可以直接退出
    //     if(std::next(spi)==epi) continue;
    //     // std::cout << std::endl << "**********************************************" << std::endl;

    //     // 检查线段上的每个点，选择最大距离的点进行分割，此外，删除距离较小的点
    //     std::pair<float, std::list<Pt>::iterator> max_pt = {-FLT_MAX, std::next(spi)};
    //     auto sp = *spi, ep = *epi;
    //     for(auto cki=std::next(spi); cki!=epi;) {
    //         auto maybe_erase = cki;
    //         cki++;

    //         float d = dist_to_line(sp, ep, *maybe_erase);
    //         // std::cout << "sp: " << sp << ", ep: " << ep << ", mbe: " << *maybe_erase << ", d: " << d <<std::endl;
    //         if(d <= grid_size*0.05) {
    //             ptlst.erase(maybe_erase);
    //         }
    //         else if(max_pt.first < d){
    //             // std::cout << "Overmax-- origin: " << *max_pt.second << ", mpt:" << *maybe_erase << std::endl;
    //             max_pt = std::make_pair(d, maybe_erase);
    //         }
    //     }
    //     if(max_pt.first >= 0)
    //     {
    //         stack.push_back({spi, max_pt.second});
    //         stack.push_back({max_pt.second, epi});
    //         // std::cout << "- - - - - " << std::endl;
    //         // std::cout << "Push " << *max_pt.second << "," << *epi << std::endl;
    //         // std::cout << "Push " << *spi << "," << *max_pt.second << std::endl;
    //     }
    // }
    return ArrayPt<float>(ptlst.begin(), ptlst.end());
}

//
ArrayShape<float> processGridmap(const Array2d &map, float grid_size)
{
    // 提取连通域
    std::vector<ArrayPt<int>> regions;
    extractAllRegions(map, map.size(), map[0].size(), regions);

    // 提取边集
    std::vector<ArrayEdge<float>> regions_edge;
    std::transform(regions.begin(), regions.end(), std::back_inserter(regions_edge), [&](const auto &region_pts)
                    { return extractRegionEdge(map, region_pts, map.size(), map[0].size(), grid_size); });

    // 顺时针排列
    std::vector<Shape<float>> shapes_arr;
    std::transform(regions_edge.begin(), regions_edge.end(), std::back_inserter(shapes_arr), [](const auto &edges)
                    { return mergeEdgeToShape(edges); });

    // 修剪多余的边
    std::vector<Shape<float>> trim_shapes;
    std::transform(shapes_arr.begin(), shapes_arr.end(), std::back_inserter(trim_shapes), [grid_size](const auto &shape) {
        Shape<float> trim_shape;
        std::transform(shape.begin(), shape.end(), std::back_inserter(trim_shape),[grid_size](const Poly<float>& poly){
            return douglasPeukcer(poly, grid_size);
        });
        return trim_shape;
    });

    // 边计数
    auto total_points = std::accumulate(trim_shapes.begin(), trim_shapes.end(), 0, [](int a, ArrayPoly<float> ap)
                                        { return a + std::accumulate(ap.begin(), ap.end(), 0, [](int b, Poly<float> p)
                                                                        { return b + p.size(); }); });
    printf("Total Point Num: %d\n", total_points);

    return trim_shapes;


}

} // namespace map_gen
