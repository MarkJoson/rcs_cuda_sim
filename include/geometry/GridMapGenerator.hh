#ifndef CUDASIM_GEOMETRY_GRIDMAP_GENERATOR_HH
#define CUDASIM_GEOMETRY_GRIDMAP_GENERATOR_HH

#include <opencv2/core.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <cuda_runtime.h>

#include "core/storage/GTensorConfig.hh"
#include "shapes.hh"
#include "transform.hh"

namespace cuda_simulator {
namespace geometry {

struct GridMapDescription {
    float resolu = 1;
    float2 origin = {0, 0};
    int2 grid_size = {0, 0};

    std::pair<int, int> world2Grid(Vector2 point) {
        int grid_x = std::max(0, int((point.x() - origin.x) / resolu));
        grid_x = std::min(grid_x, grid_size.x - 1);
        int grid_y = std::max(0, int((point.y() - origin.y) / resolu));
        grid_y = std::min(grid_y, grid_size.y - 1);
        return {grid_x, grid_y};
    }
};



class GridMapGenerator {
public:
    static constexpr int MAX_DIST = 1e6;
    explicit GridMapGenerator() = default;

    GridMapGenerator(float width, float height, Vector2 origin, float resolu) : desc_() {
        desc_.origin.x = origin.x();
        desc_.origin.y = origin.y();
        desc_.grid_size.x = std::ceil(width / resolu);
        desc_.grid_size.y = std::ceil(height / resolu);
        desc_.resolu = resolu;
        occ_map_ = cv::Mat::zeros(desc_.grid_size.y, desc_.grid_size.x, CV_8UC1);
    }

    GridMapDescription getGridMapDescritpion() const {
        return desc_;
    }

    void drawPolygon(const PolygonShapeDef& poly, const Transform2D& tf) {
        auto pt_list = std::vector<cv::Point>();

        std::transform(poly.vertices.begin(), poly.vertices.end(), std::back_insert_iterator(pt_list),
            [&](Vector2 pt) {
                auto new_pt = tf.localPointTransform(pt);
                auto [x, y] = desc_.world2Grid(new_pt);
                return cv::Point(x, y);
            });
        cv::fillConvexPoly(occ_map_, pt_list, cv::Scalar(255));
    }

    void drawCircle(const CircleShapeDef& circle, const Transform2D& tf) {
        auto center = tf.localPointTransform(circle.center);
        auto radius = circle.radius;
        auto [center_x, center_y] = desc_.world2Grid(center);
        auto radius_grid = std::ceil(radius / desc_.resolu);

        cv::circle(occ_map_, cv::Point(center_x, center_y), radius_grid, cv::Scalar(255), -1);
    }

    /***
     * @brief 计算欧几里得距离场（快速算法）
     * @param output 输出的距离场，每个元素包含了距离场的x，y分量，距离值，以及是否是边界
     */
    void fastEDT(core::TensorHandle& output) const {
        //-----------------------------  计算辅助地图：边界与inside  -------------------------------------
        // 生成可用的边界地图
        cv::Mat kernel = cv::Mat::ones(3, 3, CV_8U);
        kernel.convertTo(kernel, CV_8UC1);
        cv::Mat erosion, edge_map;
        cv::erode(occ_map_, erosion, kernel);
        // cv::dilate(m_occMap, erosionImg, kernel);
        cv::absdiff(occ_map_, erosion, edge_map);

        cv::Mat inside_flags = erosion;      // 指示该点是否在障碍物内，有障碍物的地方为255

        // ------------------------------------  计算距离场  --------------------------------------------
        cv::Mat col_chk_map = cv::Mat::zeros(desc_.grid_size.y, desc_.grid_size.x, CV_32FC1);
        cv::Mat col_chk_field = cv::Mat::zeros(desc_.grid_size.y, desc_.grid_size.x, CV_32FC2);
        col_chk_map.setTo(MAX_DIST);

        auto ks = std::vector<int>(desc_.grid_size.x + 1, 0);
        auto js = std::vector<int>(desc_.grid_size.x + 1, 0);

        cv::Mat dist_map = cv::Mat::zeros(desc_.grid_size.y, desc_.grid_size.x, CV_32FC1);       // 距离场
        cv::Mat dist_field = cv::Mat::zeros(desc_.grid_size.y, desc_.grid_size.x, CV_32FC2);     // 距离矢量场

        // lambda函数，检查edgemap，并更新当前的矩阵
        auto lambda = [&col_chk_map, &edge_map, &col_chk_field](int &obst_index, int x, int y) {
            // 至今没有遇到障碍物
            if (obst_index == -1)
                return;

            // 当前是障碍物
            if (edge_map.at<uint8_t>(y, x) == 255) {
                obst_index = y;
                col_chk_map.at<float>(y, x) = 0;
                col_chk_field.at<cv::Vec2f>(y, x) = cv::Vec2f(0, 0);
                return;
            }

            // 更新当前的距离最近距离
            float dist_sqr = (y - obst_index) * (y - obst_index);
            if (col_chk_map.at<float>(y, x) > dist_sqr) {
                col_chk_map.at<float>(y, x) = dist_sqr;
                col_chk_field.at<cv::Vec2f>(y, x) = cv::Vec2f(0, y - obst_index);
            }
        };

        // 计算两条相同曲率抛物线的交点
        auto parabolaIntersection = [](float Dl, float Dk, int l, int k)->int {
            if (l == k)
                return l;
            return std::ceil((Dl - Dk - k * k + l * l) / (2 * (l - k)));
        };

        // 先按列遍历
        for (int x = 0; x < desc_.grid_size.x; x++) {
            int obst_index = -1, obst_index_rev = -1;
            for (int y = 0; y < desc_.grid_size.y; y++) {
                lambda(obst_index, x, y);
            }
            for (int y = desc_.grid_size.y - 1; y >= 0; y--) {
                lambda(obst_index_rev, x, y);
            }
        }

        for (int j = 0; j < desc_.grid_size.y; j++) {
            int idx = 0;
            std::fill(ks.begin(), ks.end(), 0);
            std::fill(js.begin(), js.end(), 0);

            ks[0] = 0;
            js[0] = -MAX_DIST;

            for (int i = 0; i < desc_.grid_size.x; i++) {
                float val_ji = col_chk_map.at<float>(j, i);
                if (val_ji < MAX_DIST) { // 当前列j 有障碍物
                    // 计算与上一个抛物线的交点
                    int jd = parabolaIntersection(
                            val_ji, col_chk_map.at<float>(j, ks[idx]), i, ks[idx]);
                    // 交点在界外时，没有贡献
                    if (jd >= desc_.grid_size.x)
                        continue;
                    // 新交点的位置小于上一个交点的位置，妥妥的要替换
                    // 康康到底需要替换几条
                    while (jd < js[idx]) {
                        if(idx == 0)
                            throw std::runtime_error("idx == 0, at jd < js[idx]");
                        idx -= 1;
                        jd = parabolaIntersection(
                                val_ji, col_chk_map.at<float>(j, ks[idx]), i, ks[idx]);
                    }
                    // 这里应该就是新交点的位置了
                    if (jd < desc_.grid_size.x) {
                        idx += 1;
                        ks[idx] = i;
                        js[idx] = std::max(0, jd);
                    }
                }
            }
            js[0] = 0;
            js[idx + 1] = desc_.grid_size.x;

            for (int n = 0; n < idx + 1; n++) {
                int k = ks[n];
                int D = col_chk_map.at<float>(j, k);
                for (int i = js[n]; i < js[n + 1]; i++) {
                    double dist = sqrt(D + (k - i) * (k - i));
                    // 有障碍物的地方为255
                    float sign = inside_flags.at<uint8_t>(j, i) == 255 ? 1 : -1;

                    // 方向指向障碍物内部
                    if (dist != 0)
                        dist_field.at<cv::Vec2f>(j, i) =
                                sign * cv::Vec2f(i - k, col_chk_field.at<cv::Vec2f>(j, k)[1]) / dist;
                    else
                        dist_field.at<cv::Vec2f>(j, i) = cv::Vec2f(0, 0);

                    dist_map.at<float>(j, i) = (-sign) * dist * desc_.resolu;
                }
            }
        }

        //------------------------------------  导出距离场  --------------------------------------------
        // output.resize(desc_.grid_size.x * desc_.grid_size.y);
        if(output.shape()[0] != desc_.grid_size.y || output.shape()[1] != desc_.grid_size.x || !output.is_contiguous()) {
            throw std::runtime_error("output container is not match with the output!");
        }

        float4 *output_ptr = reinterpret_cast<float4*>(output.data());

        for (int y = 0; y < desc_.grid_size.y; y++) {
            for (int x = 0; x < desc_.grid_size.x; x++) {
                output_ptr[y * desc_.grid_size.x + x] = make_float4(
                        dist_field.at<cv::Vec2f>(y, x)[0], dist_field.at<cv::Vec2f>(y, x)[1],
                        dist_map.at<float>(y, x), edge_map.at<uint8_t>(y, x));
            }
        }
    }

private:
    GridMapDescription desc_;

    cv::Mat occ_map_;        // 障碍物占有地图
};


} // namespace geometry
} // namespace cuda_simulator

#endif // CUDASIM_GEOMETRY_GRIDMAP_GENERATOR_HH