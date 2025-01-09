#include <iostream>
#include <memory>
#include <vector>
#include "geometry/GeometryManager.cuh"

using namespace cuda_simulator;
using namespace cuda_simulator::geometry;

void testGeometryManager() {
    try {
        // 创建模拟器上下文
        auto context = std::make_unique<core::SimulatorContext>();

        // 创建几何管理器
        auto geom_manager = std::make_unique<GeometryManager>();

        // 注册几何管理器
        geom_manager->onRegister(context.get());

        // 创建一些静态多边形
        std::vector<Vector2> static_vertices = {
            Vector2(0.0, 0.0),
            Vector2(1.0, 0.0),
            Vector2(1.0, 1.0),
            Vector2(0.0, 1.0)
        };

        PolygonShapeDef static_polygon(
            static_vertices,
            Vector2(0.5, 0.5), // centroid
            0.707 // radius (approximate diagonal/2)
        );

        // 在环境组0中创建静态多边形
        Transform2D static_pose(Vector2(2.0, 2.0), Rotation2D(M_PI/4));
        geom_manager->createStaticPolyObj(0, static_polygon, static_pose);

        // 创建一个动态多边形
        std::vector<Vector2> dynamic_vertices = {
            Vector2(-0.5, -0.5),
            Vector2(0.5, -0.5),
            Vector2(0.5, 0.5),
            Vector2(-0.5, 0.5)
        };

        PolygonShapeDef dynamic_polygon(
            dynamic_vertices,
            Vector2(0.0, 0.0), // centroid
            0.707 // radius
        );

        // 创建动态物体
        auto dynamic_obj = geom_manager->createDynamicPolyObj(dynamic_polygon);

        // 启动几何管理器
        geom_manager->onStart(context.get());

        // 执行一次更新
        geom_manager->onExecute(context.get());

        std::cout << "GeometryManager test completed successfully!" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Test failed with error: " << e.what() << std::endl;
    }
}

int main() {
    testGeometryManager();
    return 0;
}