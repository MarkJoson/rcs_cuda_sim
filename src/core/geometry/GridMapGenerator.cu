#include <opencv2/core.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/types.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <cuda_runtime_api.h>

#include "geometry/shapes.hh"
#include "geometry/GridMapGenerator.hh"
#include "core/storage/GTensorConfig.hh"

namespace cuda_simulator {
namespace core {
namespace geometry {

class CvMatViewer {
public:
    static constexpr int IMG_SCALING_FACTOR = 1;

    static cv::Mat floatMapToU8C3(const cv::Mat& float_map) {
        double min_val, max_val;
        cv::Point min_loc, max_loc;
        cv::Mat img_u8c1;
        float_map.copyTo(img_u8c1);

        // disp_img = cv::max(disp_img, 0);
        cv::minMaxLoc(img_u8c1, &min_val, &max_val, &min_loc, &max_loc);
        img_u8c1 = (img_u8c1-min_val) / (max_val-min_val) * 255;
        img_u8c1.convertTo(img_u8c1, CV_8UC1);

        cv::Mat img_color;
        cv::applyColorMap(img_u8c1, img_color, cv::COLORMAP_BONE);

        return img_color;
    }

    static void showFloatImg(const cv::Mat& float_img, const cv::Mat& addition_img) {
        // float_img = float_img * (addition_img==255);
        // cv::multiply(float_img, 0.1*(addition_img==255), float_img);

        cv::imshow("additionimg", addition_img);
        cv::Mat img = floatMapToU8C3(float_img);
        cv::Mat addition_img_color;
        cv::cvtColor(addition_img, addition_img_color, cv::COLOR_GRAY2BGR);
        cv::addWeighted(img, 0.7, addition_img_color, 0.3, 0, img);


        cv::resize(img, img, img.size()*IMG_SCALING_FACTOR, 0, 0, 0);
        cv::imshow("showFloatImg", img);
        cv::waitKey(0);
    }
};


class GridMapGenerator::Impl {
public:
    static constexpr int MAX_DIST = 1e6;

    Impl(const GridMapDescription& desc = {0,0,{},1}) : desc_(desc) {
        occ_map_ = cv::Mat::zeros(desc_.grid_size.y, desc_.grid_size.x, CV_8UC1);
    }

    void drawPolygon(const SimplePolyShapeDef& poly, const Transform2D& tf) {
        std::vector<std::vector<cv::Point>> pt_list(1);

        for(const auto& vertex : poly.vertices) {
            auto new_pt = tf.localPointTransform(vertex);
            auto [x, y] = desc_.world2Grid(new_pt);
            pt_list[0].push_back(cv::Point(x, y));
        }

        cv::fillPoly(occ_map_, pt_list, cv::Scalar(255));
    }

    void drawPolygon(const ComposedPolyShapeDef& poly, const Transform2D& tf) {
        std::vector<std::vector<cv::Point>> cv_polys;
        for(const auto& simple_poly : poly.positive_polys) {
            std::vector<cv::Point> cv_poly;
            for(const auto& vertex : simple_poly.vertices) {
                Vector2i global_pt = desc_.world2Grid(tf.localPointTransform(vertex));
                cv_poly.push_back(cv::Point(global_pt.x, global_pt.y));
            }
            cv_polys.push_back(cv_poly);
        }

        for(const auto& simple_poly : poly.negative_polys) {
            std::vector<cv::Point> cv_poly;
            for(auto iter=simple_poly.vertices.rbegin(); iter!= simple_poly.vertices.rend(); iter++) {
                Vector2i global_pt = desc_.world2Grid(tf.localPointTransform(*iter));
                cv_poly.push_back(cv::Point(global_pt.x, global_pt.y));
            }
            cv_polys.push_back(cv_poly);
        }

        cv::fillPoly(occ_map_, cv_polys, cv::Scalar(255));
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
    void fastEDT(core::GTensor& output) {
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
            // 当前是障碍物
            if (edge_map.at<uint8_t>(y, x) == 255) {
                obst_index = y;
                col_chk_map.at<float>(y, x) = 0;
                col_chk_field.at<cv::Vec2f>(y, x) = cv::Vec2f(0, 0);
                return;
            }

            // 已经遇到障碍物，更新当前的距离最近距离
            if (obst_index != -1) {
                float dist_sqr = (y - obst_index) * (y - obst_index);
                if (col_chk_map.at<float>(y, x) > dist_sqr) {
                    col_chk_map.at<float>(y, x) = dist_sqr;
                    col_chk_field.at<cv::Vec2f>(y, x) = cv::Vec2f(0, y - obst_index);
                }
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
        if(output.shape()[0] != desc_.grid_size.y || output.shape()[1] != desc_.grid_size.x || !output.isContiguous()) {
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

        // cv::Mat blob_map = cv::Mat(desc_.grid_size.y, desc_.grid_size.x, CV_32FC4, output_ptr);
        // std::vector<cv::Mat> dist_float_splited;
        // cv::split(blob_map, dist_float_splited);
        // CvMatViewer::showFloatImg(dist_float_splited[2], occ_map_);
    }

    GridMapDescription getGridMapDescription() const {
        return desc_;
    }

private:
    GridMapDescription desc_;
    cv::Mat occ_map_;        // 障碍物占有地图
};

GridMapGenerator::GridMapGenerator(const GridMapDescription& desc) {
    impl_ = std::make_unique<Impl>(desc);
}

GridMapGenerator::~GridMapGenerator() = default;

void GridMapGenerator::drawPolygon(const SimplePolyShapeDef& poly, const Transform2D& tf) {
    impl_->drawPolygon(poly, tf);
}

void GridMapGenerator::drawPolygon(const ComposedPolyShapeDef& poly, const Transform2D& tf) {
    impl_->drawPolygon(poly, tf);
}

void GridMapGenerator::drawCircle(const CircleShapeDef& circle, const Transform2D& tf) {
    impl_->drawCircle(circle, tf);
}

void GridMapGenerator::fastEDT(GTensor& output) {
    impl_->fastEDT(output);
}

GridMapDescription GridMapGenerator::getGridMapDescription() const {
    return impl_->getGridMapDescription();
}


} // namespace geometry
} // namespace core
} // namespace cuda_simulator
