#include <iostream>
#include <memory>
#include <opencv2/core/hal/interface.h>
#include <vector>

#include <opencv2/opencv.hpp>

#include "geometry/GeometryManager.cuh"
#include "core/MessageBus.hh"

using namespace cuda_simulator;
using namespace cuda_simulator::core::geometry;

void testGeometryManager() {
    // try {
        // 创建模拟器上下文


        // 创建几何管理器
        auto geom_manager = std::make_unique<GeometryManager>();


        // 创建一些静态多边形
        std::vector<Vector2f> static_vertices = {
            Vector2f(0.0, 0.0),
            Vector2f(1.0, 0.0),
            Vector2f(1.0, 1.0),
            Vector2f(0.0, 1.0)
        };

        SimplePolyShapeDef static_polygon(static_vertices);

        // 在环境组0中创建静态多边形
        Transform2D static_pose(Vector2f(2.0, 2.0), Rotation2D(M_PI/4));
        geom_manager->createStaticPolyObj(0, static_polygon, static_pose);

        // 创建一个动态多边形
        std::vector<Vector2f> dynamic_vertices = {
            Vector2f(-0.5, -0.5),
            Vector2f(0.5, -0.5),
            Vector2f(0.5, 0.5),
            Vector2f(-0.5, 0.5)
        };

        SimplePolyShapeDef dynamic_polygon(dynamic_vertices);

        // 创建动态物体
        auto dynamic_obj = geom_manager->createDynamicPolyObj(dynamic_polygon);

        // 启动几何管理器
        geom_manager->assemble();

        // TODO.
        core::getEnvGroupMgr()->syncToDevice();

        // 执行一次更新
        geom_manager->execute();

        std::cout << "GeometryManager test completed successfully!" << std::endl;

    // } catch (const std::exception& e) {
    //     std::cerr << "Test failed with error: " << e.what() << std::endl;
    // }
}

constexpr int IMG_SCALING_FACTOR = 1;

// 辅助函数：将EDF数据转换为可视化图像
cv::Mat visualizeEDF(const float4* edf_data, int width, int height) {
    cv::Mat visualization(height, width, CV_8UC3);

    // 找到最大距离值用于归一化
    float max_distance = 0.0f;
    for(int i = 0; i < width * height; i++) {
        max_distance = std::max(max_distance, std::abs(edf_data[i].z));
    }

    // 将EDF值转换为颜色
    for(int y = 0; y < height; y++) {
        for(int x = 0; x < width; x++) {
            float distance = edf_data[y * width + x].z;

            // 归一化距离值到0-1范围
            float normalized_distance = std::abs(distance) / max_distance;

            // 使用不同的颜色表示距离
            cv::Vec3b color;
            if(distance < 0) {
                // 物体内部为红色系
                color[0] = 0;  // B
                color[1] = 0.6*static_cast<uchar>(255 * normalized_distance);;  // G
                color[2] = static_cast<uchar>(255 * normalized_distance);  // R
            } else {
                // 物体外部为蓝色系
                color[0] = static_cast<uchar>(255 * normalized_distance);  // B
                color[1] = 0;  // G
                color[2] = 255-0.8*static_cast<uchar>(255 * normalized_distance);;  // R
            }
            visualization.at<cv::Vec3b>(y, x) = color;
        }
    }

    return visualization;
}

void testGeometryManagerEDF() {
    // try {
        // 创建模拟器上下文
        auto context = std::make_unique<core::SimulatorContext>();

        // 创建几何管理器
        auto geom_manager = std::make_unique<GeometryManager>();

        // 创建一些静态多边形进行测试

        // 1. 创建一个L形状的多边形
        std::vector<Vector2f> l_shape_vertices = {
            Vector2f(3.0, 3.0),
            Vector2f(7.0, 3.0),
            Vector2f(7.0, 4.0),
            Vector2f(4.0, 4.0),
            Vector2f(4.0, 7.0),
            Vector2f(3.0, 7.0)
        };

        SimplePolyShapeDef l_shape_polygon(l_shape_vertices);

        // 2. 创建一个小正方形
        std::vector<Vector2f> square_vertices = {
            Vector2f(1.0, 1.0),
            Vector2f(2.0, 1.0),
            Vector2f(2.0, 2.0),
            Vector2f(1.0, 2.0)
        };

        SimplePolyShapeDef square_polygon(square_vertices);

        // 在环境组0中创建静态多边形
        geom_manager->createStaticPolyObj(0, l_shape_polygon, Transform2D());
        geom_manager->createStaticPolyObj(0, square_polygon, Transform2D());

        geom_manager->assemble();

        // 获取EDF数据并可视化
        // 假设我们知道GRIDMAP的尺寸是100x100 (10m/0.1m)
        constexpr int GRID_WIDTH = 100;
        constexpr int GRID_HEIGHT = 100;

        // 创建窗口显示结果
        cv::namedWindow("EDF Visualization", cv::WINDOW_NORMAL);
        // cv::resizeWindow("EDF Visualization", 800, 800);

        // 获取EDF数据并转换为可视化图像
        core::GTensor static_esdf = geom_manager->getStaticESDF(0);

        // 获取环境组0的EDF数据
        auto esdf_data = static_esdf.typed_data<float4>();

        // 创建可视化图像
        cv::Mat vis_img = visualizeEDF(esdf_data, static_esdf.shape()[1], static_esdf.shape()[0]);

        // 显示图像
        cv::imshow("EDF Visualization", vis_img);

        // 保存图像
        cv::imwrite("edf_visualization.png", vis_img);

        std::cout << "EDF visualization has been saved to 'edf_visualization.png'" << std::endl;
        std::cout << "Press any key to exit..." << std::endl;

        cv::waitKey(0);

        std::cout << "GeometryManager EDF test completed successfully!" << std::endl;

    // } catch (const std::exception& e) {
    //     std::cerr << "Test failed with error: " << e.what() << std::endl;
    // }
}

int main() {
    // testGeometryManager();
    testGeometryManagerEDF();
    return 0;
}