#include "map_generator/map_generator.h"
#include <iostream>
#include <numeric>
#include <algorithm>
#include <ranges>
#include <string_view>
#include <deque>
#include <ranges>
#include <unordered_map>
#include <unordered_set>
#include <boost/iterator/zip_iterator.hpp>
#include <boost/range/join.hpp>
#include <opencv2/opencv.hpp>
#include <glog/logging.h>


using namespace map_gen;
using namespace std::literals;

constexpr int MAP_WIDTH = 80;
constexpr int MAP_HEIGHT = 60;
constexpr int GRID_SIZE = 10;

typedef struct Point {
    int x;
    int y;
    bool operator==(const Point& other) const {
        return x == other.x && y == other.y;
    }
} Pt;

std::ostream& operator<< (std::ostream& out, const Point& d) {
    out << " ["<<d.x<<","<<d.y<<"] ";
    return out;
}

struct Edge {
    Point start;
    Point end;
    bool operator==(const Edge& other) const {
        return start == other.start && end == other.end;
    }
};

template<>
struct std::hash<Point> {
public:
    size_t operator()(const Point& pt) const {return ((uint64_t)pt.x << 32 | (uint64_t)pt.y);}
};

template<>
class std::hash<Edge> {
public:
    size_t operator()(const Edge& e) const {return std::hash<Point>()(e.start) ^ std::hash<Point>()(e.start) << 1;}
};

// 计算单点Hash值
uint64_t ptHash(const Pt &pt) { return std::hash<Pt>()(pt); }

using Array2d = std::vector<std::vector<int>>;
using ArrayPt = std::vector<Pt>;
using Poly = ArrayPt;
using ArrayEdge = std::vector<Edge>;
using ArrayPoly = std::vector<ArrayPt>;
using PtHash = uint64_t;

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
        ArrayPt region_pts;
        // 从当前点出发，进行深度优先遍历，使用栈表
        ArrayPt ptstack;
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

// 提取region的边
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
                edge_out.push_back({{x*gs+p1.x, y*gs+p1.y},{x*gs+p2.x, y*gs+p2.y}});
            }
        }
    }

    return edge_out;
}

// 将边集合拆分成顺时针的多边形
ArrayPoly mergeEdgeToPoly(const ArrayEdge &edges)
{
    ArrayPoly polys;

    std::unordered_map<PtHash, std::list<int>> map_spt_idx;
    // std::unordered_map<PtHash, std::list<int>> map_ept_idx;

    for(int i=0; i<edges.size(); i++)
    {
        const auto &[spt, ept] = edges[i];
        PtHash hash_spt = ptHash(spt), hash_ept = ptHash(ept);

        if(map_spt_idx.find(hash_spt)==map_spt_idx.end()) map_spt_idx[hash_spt] = std::list<int>();
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

    std::unordered_set<Edge> edge_set(edges.begin(), edges.end());

    while(!edge_set.empty())
    {
        Poly poly;

        // 从随机一个边的起点开始
        Pt pt = edge_set.begin()->start;
        poly.push_back(pt);
        while(true)
        {
            // 寻找起始点spt点对应的边，获得线段的end
            auto spt_vec_iter = map_spt_idx.find(ptHash(pt));
            if(spt_vec_iter == map_spt_idx.end())
            {
                // 如果当前的曲线未闭合，连接head形成多边形闭合。
                if(!(pt == poly.front()))
                {
                    printf("Polygon is not closed!!!\n");
                    poly.push_back(poly[0]);
                }
                break;
            }

            auto &spt_vec = spt_vec_iter->second;
            int edge_id = spt_vec.front();
            spt_vec.pop_front();
            if(spt_vec.empty()) map_spt_idx.erase(ptHash(pt));

            const Edge &edge = edges[edge_id];
            edge_set.erase(edge);

            // 如果点已经存在于点集
            pt = edge.end;
            poly.push_back(pt);
        }
        polys.push_back(poly);
    }
    return polys;
}

float dist_to_line(Pt l1, Pt l2, Pt op)
{
    if(abs(l2.x-l1.x) == 0 && abs(l2.y-l1.y) == 0)
        throw std::exception();
    float dist = l1.x*l2.y + l2.x*op.y + op.x*l1.y - l1.x*op.y - op.x*l2.y - l2.x*l1.y;
    dist /= 2*sqrtf((l2.x-l1.x)*(l2.x-l1.x)+(l2.y-l1.y)*(l2.y-l1.y));
    return fabs(dist);
}

// 将点集精简
ArrayPt douglasPeukcer(const ArrayPt &pts, float grid_size)
{
    std::list<Pt> ptlst; //(pts.begin(), pts.end());

    // !----- 粗精简，合并同一条直线上的点
    // 拼接首尾tail, head, ..., tail, head
    auto tht_arr = boost::join( std::array<Pt,1>({*std::prev(pts.end())}), boost::join(pts, std::array<Pt,1>{pts[0]}));
    for(auto pt_iter=tht_arr.begin()+1; pt_iter != std::prev(tht_arr.end()); pt_iter++)
    {
        auto p1i = std::prev(pt_iter);
        auto p2i = std::next(pt_iter);
        // if(p1i==p2i) {p2i = pt_iter; continue;}
        // 如果不在直线上，则设置p1和p2到拐点，
        const auto &p1 = *p1i, &p2 = *p2i, &p3 = *pt_iter;
        if((p2.y-p1.y)*(p3.x-p2.x) != (p3.y-p2.y)*(p2.x-p1.x))
        {
            ptlst.push_back(*pt_iter);
        }
    }

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
    return ArrayPt(ptlst.begin(), ptlst.end());
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
        cv::cvtColor(matmap, matmap, cv::COLOR_GRAY2BGRA);
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
    std::transform(regions.begin(), regions.end(), std::back_inserter(regions_edge), [&](const auto &region_pts)->ArrayEdge {
        return extractRegionEdge(map, region_pts, map.size(), map[0].size(), GRID_SIZE);
    });

    std::vector<ArrayPoly> array_polys;
    std::transform(regions_edge.begin(), regions_edge.end(), std::back_inserter(array_polys), [](const auto &edges) {
        return mergeEdgeToPoly(edges);
    });

    std::vector<ArrayPoly> trim_polys;
    std::transform(array_polys.begin(), array_polys.end(), std::back_inserter(trim_polys), [](const auto &convexs){
        ArrayPoly trim_convexs;
        std::transform(convexs.begin(), convexs.end(), std::back_inserter(trim_convexs),[](const Poly& convex){
            return douglasPeukcer(convex, GRID_SIZE);
        });
        return trim_convexs;
    });

    auto region_iter = boost::make_zip_iterator(boost::make_tuple(regions_edge.begin(), array_polys.begin()));
    auto iter_end = boost::make_zip_iterator(boost::make_tuple(regions_edge.end(), array_polys.end()));

    MapViewer dmv(map);

    char ch = ' ';
    for(;region_iter!=iter_end && ch!='q'; region_iter++)
    {
        dmv.show("dmv");
        ch = cv::waitKey(10);
        if (ch == 'q') exit(0);

        const auto &[region_edge, trim_poly] = *region_iter;

        for (const auto &[p1, p2] : region_edge)
            cv::line(dmv.matmap, cv::Point(p1.x, p1.y), cv::Point(p2.x, p2.y), cv::Scalar(255,0,0), 3);

        std::vector<cv::Point> kps;
        std::for_each(trim_poly.begin(), trim_poly.end(), [&kps](const auto& poly) {
            std::transform(poly.begin(), poly.end(), std::back_inserter(kps), [](const Pt& pt) {
                return cv::Point(pt.x, pt.y);
            });
        });

        std::for_each(kps.begin(), kps.end(), [&dmv](const auto  &pt) {
            cv::drawMarker(dmv.matmap, pt, cv::Scalar(0,0,255), cv::MARKER_DIAMOND, 5, 2);
        });


        std::vector<std::vector<cv::Point>> polys;
        std::transform(trim_poly.begin(), trim_poly.end(), std::back_inserter(polys), [](const auto &convexs){
            std::vector<cv::Point> cv_convex;
            std::transform(convexs.begin(), convexs.end(), std::back_inserter(cv_convex), [](const Pt& pt) { return cv::Point(pt.x, pt.y); });
            return cv_convex;
        });
        cv::fillPoly(dmv.matmap, polys, cv::Scalar(0,255,0,0.2));
    }
    dmv.show("dmv");
    ch = cv::waitKey(0);
    if (ch == 'q') exit(0);
}

int main(int argc, char** args)
{
    google::InstallFailureSignalHandler();
    // genShow();
    for(int i=0;i<100;i++) {
        genShow();
    }
    printf("Generate Done!\n");
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