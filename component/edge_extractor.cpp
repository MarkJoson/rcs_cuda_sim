#include "map_generator/map_generator.h"
#include <iostream>
#include <numeric>
#include <algorithm>
#include <ranges>
#include <string_view>
#include <deque>
#include <unordered_map>
#include <boost/iterator/zip_iterator.hpp>

#include <opencv2/opencv.hpp>

using namespace map_gen;
using namespace std::literals;

constexpr int MAP_WIDTH = 160;
constexpr int MAP_HEIGHT = 120;
constexpr int GRID_SIZE = 5;

using Array2d = std::vector<std::vector<int>>;
using Pt = std::pair<int, int>;
using Edge = std::pair<Pt, Pt>;
using ArrayPt = std::vector<Pt>;
using Poly = ArrayPt;
using ArrayEdge = std::vector<Edge>;
using ArrayPoly = std::vector<ArrayPt>;

void printMap(const Array2d &map)
{
    for(auto &row : map)
    {
        // std::transform(map[y].begin(), map[y].end(), row_string.begin(), [](const int &x) {return std::to_string(x);});
        auto cvt = [](int a) { return a==0 ? "□" : "■"; };
        auto row_strings = std::accumulate(
            row.begin(),
            row.end(),
            std::string{},
            [cvt](const std::string &a, const int &b) {return a.empty() ? cvt(b): a + cvt(b);});
        std::cout << row_strings << std::endl;
    }
}


// 提取所有的独立联通区域
void extractAllRegions(const Array2d &map, int width, int height, ArrayPoly &regions)
{
    std::vector<std::vector<bool>> visited( width, std::vector<bool>(height, 0) );
    
    // 标记空旷的区域为visited=1，无需遍历
    for(int y=0; y<height; y++)
        for(int x=0; x<width; x++)
            visited[x][y] = 1 - map[x][y];

    int vx=-1, vy=0;
    while(1)
    {
        int ckpt_x=0, ckpt_y = 0;
        // 找到visited_list中一个非0的点
        for(ckpt_y=vy; ckpt_y<height; ckpt_y++)
        {
            // 从上一次的vx继续开始遍历
            for(ckpt_x=vx+1; ckpt_x<width; ckpt_x++)
                if(visited[ckpt_x][ckpt_y] == 0)
                    break;
            if(ckpt_x != width)
                break;
            vx = -1;
        }
        // 记录当前找到非零点的坐标
        vx = ckpt_x, vy=ckpt_y;
        
        // 检查完所有的点就退出
        if(ckpt_y == height) break;
        
        // 当前区域的点集
        std::vector<std::pair<int, int>> region_pts;
        // 从当前点出发，进行深度优先遍历，使用栈表
        std::vector<std::pair<int, int>> ptstack;
        // 标记初始点
        visited[ckpt_x][ckpt_y] = 1;
        ptstack.push_back({ckpt_x, ckpt_y});
        
        while(!ptstack.empty())
        {
            auto [x, y] = ptstack.back();
            region_pts.push_back({x, y});
            ptstack.pop_back();

            // 将邻居节点加入候选列表
            static constexpr std::array<std::pair<int, int>, 4> neighbors4 {{{-1, 0}, { 0, -1}, { 0, 1}, { 1, 0}}};
            for(const auto &[dx, dy] : neighbors4)
            {
                if(x+dx < 0 || x+dx >= width || y+dy < 0 || y+dy >= height) continue;
                if(!visited[x+dx][y+dy] && map[x+dx][y+dy] == 1)
                {
                    visited[x+dx][y+dy] = 1;
                    ptstack.push_back({x+dx,y+dy});
                }
            }
        }
        // 加入regions清单
        regions.push_back(region_pts);
    }
}


ArrayEdge extractRegionEdge(Array2d map, ArrayPt region_pts, int width, int height, int gs)
{
    static constexpr std::array<Pt, 4> neighbors4 {{{-1, 0}, { 0, -1}, { 1, 0}, { 0, 1}}};
    const std::array<Edge, 4> neighbor_edges {{
        {{0,gs}, {0,0}},
        {{0,0}, {gs, 0}},
        {{gs,0}, {gs, gs}},
        {{gs,gs}, {0, gs}},
    }};
        

    ArrayEdge edge_out;
    int edge_pixel = 0, edge_cnt = 0;
    for(auto &[x, y] : region_pts)
    {
        int n_degree = 0, n_flag = 0;

        for(int i=0; i<neighbors4.size(); i++)
        {
            const auto &[dx, dy] = neighbors4[i];
            if(x+dx < 0 || x+dx >= width || y+dy < 0 || y+dy >= height) continue;
            if(map[x+dx][y+dy] == 1) {
                n_degree += 1;
                n_flag |= 1 << i;
                edge_pixel ++;
            }
        }

        if(n_degree == 4) continue;

        for(int i=0; i<4; i++)
        {
            if (!(n_flag & 1<<i)) {
                edge_cnt ++;
                auto &[p1, p2] = neighbor_edges[i];
                edge_out.push_back({{x*gs+p1.first, y*gs+p1.second},{x*gs+p2.first, y*gs+p2.second}});
            }
        }
    }

    return edge_out;
}

uint64_t ptHash(const Pt &pt)
{
    return ((uint64_t)pt.first << 32 | (uint64_t)pt.second);
}

using PtHash = uint64_t;

ArrayPoly mergeEdgeToPoly(const ArrayEdge &edges)
{
    ArrayPoly polys;

    std::unordered_map<PtHash, int> map_spt_idx;
    std::unordered_map<PtHash, int> map_ept_idx;

    for(int i=0; i<edges.size(); i++)
    {
        const auto &[spt, ept] = edges[i];
        PtHash hash_spt = ptHash(spt), hash_ept = ptHash(spt);
        if(map_spt_idx.find(hash_spt)!=map_spt_idx.end() || map_ept_idx.find(hash_spt)!=map_ept_idx.end())
        {
            throw std::exception();
        }
        map_spt_idx[hash_spt] = i;
        map_ept_idx[hash_ept] = i;
    }

    std::set<Edge> edge_set(edges.begin(), edges.end());

    while(!edge_set.empty())
    {
        Poly poly;
        // 此次edge的点集
        std::set<PtHash> edge_pt_set;

        // 从随机一个边的起点开始
        Pt pt = edge_set.begin()->first;
        poly.push_back(pt);
        edge_pt_set.insert(ptHash(pt));
        while(true)
        {
            // 寻找pt点对应的边，获得线段的end
            int edge_id = map_spt_idx[ptHash(pt)];
            const Edge &edge = edges[edge_id];
            edge_set.erase(edge);
            
            // 如果点已经存在于点集
            if(edge_pt_set.find(ptHash(edge.second)) != edge_pt_set.end())
                break;
            
            // 加入下一个点到多边形和点集
            pt = edge.second;
            poly.push_back(pt);
            edge_pt_set.insert(ptHash(pt));
        }
        polys.push_back(poly);
    }
    return polys;
}




template <typename T>
cv::Mat vectorToMat(const std::vector<std::vector<T>>& vec, int type) {
    int rows = vec.size();
    int cols = vec.empty() ? 0 : vec[0].size();

    cv::Mat mat(rows, cols, type);

    auto it = mat.begin<T>();
    for (const auto& row : vec) {
        it = std::copy(row.begin(), row.end(), it);
    }

    return mat;
}

class MapViewer
{
public:
    cv::Mat matmap;
    MapViewer(const Array2d &map) {
        // 可视化 
        matmap = 255 - vectorToMat<int>(map, CV_8UC1)*255;
        cv::resize(matmap, matmap, matmap.size()*GRID_SIZE, 0, 0, 0);
        cv::transpose(matmap, matmap);
        cv::cvtColor(matmap, matmap, cv::COLOR_GRAY2BGR);
    }

    void show(const std::string &name) {
        cv::imshow(name, matmap);
    }
};

void genShow()
{
    // auto map_generator = std::make_unique<CellularAutomataGenerator>(MAP_WIDTH, MAP_HEIGHT);
    auto map_generator = std::make_unique<MessyBSPGenerator>(MAP_WIDTH, MAP_HEIGHT);
    map_generator->generate();
    auto map = map_generator->getMap();
    auto map_copy(map);

    // 提取连通域
    std::vector<ArrayPt> regions;
    extractAllRegions(map, map.size(), map[0].size(), regions);

    // 提取边集
    std::vector<ArrayEdge> regions_edge;
    std::transform(regions.begin(), regions.end(), std::back_inserter(regions_edge), [&](const ArrayPt &region_pts)->ArrayEdge {
        return extractRegionEdge(map, region_pts, map.size(), map[0].size(), GRID_SIZE);
    });
    
    std::vector<ArrayPoly> array_polys;
    std::transform(regions_edge.begin(), regions_edge.end(), std::back_inserter(array_polys), [](const ArrayEdge &edges) {
        return mergeEdgeToPoly(edges);
    });


    auto region_iter = boost::make_zip_iterator(boost::make_tuple(regions_edge.begin(), array_polys.begin()));
    auto iter_end = boost::make_zip_iterator(boost::make_tuple(regions_edge.end(), array_polys.end()));

    MapViewer dmv(map);

    char ch = ' ';
    for(;region_iter!=iter_end && ch!='q'; region_iter++)
    {
        dmv.show("dmv");
        ch = cv::waitKey(100);

        const auto &[region_edge, region_polys] = *region_iter; 
        
        for (const auto &[p1, p2] : region_edge)
            cv::line(dmv.matmap, cv::Point(p1.first, p1.second), cv::Point(p2.first, p2.second), cv::Scalar(255,0,0), 1);

        std::vector<std::vector<cv::Point>> polys;
        for (const auto &poly : region_polys)
        {
            std::vector<cv::Point> pts;
            std::transform(poly.begin(), poly.end(), std::back_inserter(pts), [](const Pt& pt) { return cv::Point(pt.first, pt.second); });
            polys.push_back(pts);
        }
        cv::fillPoly(dmv.matmap, polys, cv::Scalar(0,255,0,500));
    }
    if (ch == 'q') exit(0);
    dmv.show("dmv");
    cv::waitKey(100);
}

int main(int argc, char** args)
{
    while(1)
        genShow();
}

    // Array2d map = {
    //     {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    //     {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    //     {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    //     {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    //     {0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0},
    //     {0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0},
    //     {0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0},
    //     {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    //     {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    //     {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    //     {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    //     {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    //     };
    // Array2d map = {
    //     {1,1,1,1},
    //     {1,0,0,1},
    //     {1,0,0,1},
    //     {1,1,1,1},
    //     };

    // Array2d map = {
    //     {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1},
    //     {1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1},
    //     {1,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,1,1,1},
    //     {1,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,1,1,1},
    //     {1,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,1,1,1},
    //     {1,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,1,1,1},
    //     {1,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,1,1,1},
    //     {1,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,1,1,1},
    //     {1,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,1,1,1},
    //     {1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1},
    //     {1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1},
    //     {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1},
    //     };

    // Array2d map = {
    //     {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    //     {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    //     {0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0},
    //     {0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0},
    //     {0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0},
    //     {0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0},
    //     {0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0},
    //     {0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0},
    //     {0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0},
    //     {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    //     {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    //     {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    //     };
    // Array2d map = {
    //     {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    //     {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    //     {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    //     {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    //     {0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0},
    //     {0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0},
    //     {0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0},
    //     {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    //     {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    //     {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    //     {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    //     {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    //     };