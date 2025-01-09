#ifndef CUDASIM_GEOMETRY_GEOMGETRY_MANAGER_HH
#define CUDASIM_GEOMETRY_GEOMGETRY_MANAGER_HH

// #include <__clang_cuda_builtin_vars.h>

// #include <__clang_cuda_builtin_vars.h>
#include <cstddef>
#include <cstdint>
#include <cuda.h>
#include <memory>
#include <opencv2/core/hal/interface.h>
#include <utility>
#include <vector>

#include <cuda_runtime.h>
#include <glm/glm.hpp>

#include "core/EnvGroupManager.hh"
#include "core/SimulatorContext.hh"
#include "core/geometry/types.hh"
#include "core/storage/GTensorConfig.hh"

#include "core/storage/TensorRegistry.hh"
#include "shapes.hh"
#include "transform.hh"
#include "GridMapGenerator.hh"

namespace cuda_simulator {
namespace geometry {

#define DUMP_DYM_LINES_CTA_SIZE 256

__global__ void dumpDynamicLinesKernel(
        const uint32_t num_dynobj_lines,
        const float     * __restrict__ dyn_obj_lines,       // 多边形边集合
        const uint32_t  * __restrict__ dyn_obj_line_ids,    // 多边形边集合对应的物体ID
        const float     * __restrict__ dyn_obj_poses,       // 多边形形状位姿，[obj, env_grp, env]
        float           * __restrict__ dyn_lines) {         // 当前多边形外形线段在坐标系的端点 [env_grp, env, lines, 4]

    // 每个block负责dyn_obj的一个line(y), 每个线程负责一个点(x)
    // const uint32_t point_id = threadIdx.x;          // 线段的点Id, 0起点，1终点
    // const uint32_t sub_batch_id = threadIdx.y;      // 子batch id, 32~256
    // const uint32_t prim_batch_id = blockIdx.x;      // 主batch id, 0~1024
    const uint32_t batch_id = blockIdx.x * blockDim.y + threadIdx.y;
    const uint32_t batch_size = blockDim.y * gridDim.x;

    const uint32_t line_id = blockIdx.y;
    const uint32_t dyn_obj_id = dyn_obj_line_ids[line_id] >> 16;

    if (line_id >= num_dynobj_lines) return;

    const uint32_t global_point_id = line_id * 2 + threadIdx.x;
    const float2 vertex = reinterpret_cast<const float2 *>(dyn_obj_lines)[global_point_id];

    const float4 pose = reinterpret_cast<const float4 *>(dyn_obj_poses)[batch_size * batch_id + dyn_obj_id];
    float sint = pose.z, cost = pose.w;
    float2 new_vertex = {
            vertex.x * cost - vertex.y * sint + pose.x,
            vertex.x * sint + vertex.y * cost + pose.y
    };

    // dyn_lines: [env_grp, env, lines,  4]
    const uint32_t output_indice = (num_dynobj_lines * batch_id + line_id) * 2 + threadIdx.x;
    reinterpret_cast<float2 *>(dyn_lines)[output_indice] = new_vertex;
}


class GeometryManager {
    constexpr static uint32_t MAX_STATIC_LINES = 16384;
    constexpr static uint32_t MAX_POLYGONS = 1024;
    constexpr static uint32_t MAX_DYNAMIC_VERTICES = MAX_POLYGONS * 8;

    constexpr static float GRIDMAP_RESOLU = 0.1;
    constexpr static float GRIDMAP_WIDTH = 10;
    constexpr static float GRIDMAP_HEIGHT = 10;
    constexpr static uint32_t GRIDMAP_GRIDSIZE_X = GRIDMAP_WIDTH / GRIDMAP_RESOLU;
    constexpr static uint32_t GRIDMAP_GRIDSIZE_Y = GRIDMAP_HEIGHT / GRIDMAP_RESOLU;

    struct SceneDescription {
        std::vector<std::pair<std::unique_ptr<ShapeDef>, Transform2D>> static_obj_defs_;    // 静态物体
    };

    // TODO. 加一个handle，用于从poses中提取某一个物体的位姿数组

public:
    struct DynamicPolyObj {};

    GeometryManager() {
        // 向环境组管理器声明参数
        // 参数：静态物体（静态物体数量，线段，esdf，gridmap）
        // 参数：动态物体（多边形数量，多边形点数集合，多边形点集合）

        // 从TensorRegistry申请动态属性：所有动态物体的位姿
    }

    ~GeometryManager() = default;

    // 静态物体在指定环境组中创建
    void createStaticPolyObj(int environ_group_id, const PolygonShapeDef &polygon_def, const Transform2D &pose) {
        scene_desc_->at(environ_group_id)
                .static_obj_defs_.push_back(std::make_pair(
                        std::make_unique<PolygonShapeDef>(polygon_def), pose));
    }

    // 动态物体在所有环境组中创建
    void createDynamicPolyObj(const PolygonShapeDef &polygon_def) {
        dynamic_obj_defs_.push_back(std::make_unique<PolygonShapeDef>(polygon_def));
    }

    void onRegister(core::SimulatorContext *context) {
        core::EnvGroupManager *env_grp_mgr = context->getEnvironGroupManager();
        // 主机内存，环境组参数
        scene_desc_ = env_grp_mgr->registerConfigItem<SceneDescription, core::MemoryType::HOST_MEM>(
                "scene_desc");

        // 静态物体，环境组参数
        num_static_lines_ = env_grp_mgr->registerConfigItem<uint32_t, core::MemoryType::CONSTANT_GPU_MEM>(
                "num_static_lines");
        static_lines_ = env_grp_mgr->registerConfigTensor<float>(
                "static_lines", {MAX_STATIC_LINES, 4});
        static_esdf_ = env_grp_mgr->registerConfigTensor<float>(
                "static_esdf",
                {MAX_STATIC_LINES, GRIDMAP_GRIDSIZE_X, GRIDMAP_GRIDSIZE_Y, 4});

        // 动态物体，环境组参数

        dyn_obj_lines_ = core::TensorRegistry::getInstance().createTensor<float>("dyn_obj_lines", {MAX_DYNAMIC_VERTICES, 4});
        dyn_obj_line_ids_ = core::TensorRegistry::getInstance().createTensor<uint32_t>("dyn_obj_line_ids", {MAX_DYNAMIC_VERTICES});
    }

    void onStart(core::SimulatorContext *context) {

        core::EnvGroupManager *env_grp_mgr = context->getEnvironGroupManager();

        // 一个环境组（32~1024个环境），固定数量的静态物体+固定数量的动态物体。每个物体的描述是：多边形点集合（多边形物体）/半径（圆物体）；
        // 每个静态物体的pose是固定的，在onStart的时候全部转换为ESDF+GridMap，以及线段集合
        // 每个动态物体对应一个pose，需要动态运行时转换为线段集合
        buildStaticSDF();
        dumpStaticLines();


        // 场景中所有的线段, [env_grp, env, lines, 4]
        dyn_lines_ = env_grp_mgr->createTensor<float>("scene_lines", {
            num_dynobj_lines_,
            4});
        // 场景中所有dyn object的pose, [obj, env_grp, env, 4]
        const uint32_t num_dynobj = dynamic_obj_defs_.size();
        dyn_poses_ = env_grp_mgr->createTensor<float>("dynamic_poses", {
            num_dynobj,
            core::EnvGroupManager::SHAPE_PLACEHOLDER_ENV_GRP,
            core::EnvGroupManager::SHAPE_PLACEHOLDER_ENV,
            4});

    }

    void onExecute(core::SimulatorContext *context) {
        // 将动态物体转换为线段
        dumpDynamicLines(context);
    }

    void onEnvGroupActive(core::SimulatorContext *context, int env_grp_id) {
        // 当EnvGroup从非激活状态变为激活状态时调用
    }

    void buildStaticSDF() {
        // 为所有的场景组生成静态物体的SDF
        for (int64_t env_grp_id=0; env_grp_id<scene_desc_->getNumEnvGroup(); env_grp_id++) {
            GridMapGenerator grid_map(GRIDMAP_WIDTH, GRIDMAP_HEIGHT, {0, 0}, GRIDMAP_RESOLU);
            for (auto &static_obj : scene_desc_->at(env_grp_id).static_obj_defs_) {
                auto &shape = static_obj.first;
                auto &pose = static_obj.second;

                if (shape->type == ShapeType::POLYGON) {
                    auto poly = reinterpret_cast<PolygonShapeDef *>(shape.get());
                    grid_map.drawPolygon(*poly, pose);
                }
            }
            core::TensorHandle selected_map = static_esdf_->at(env_grp_id);
            grid_map.fastEDT(selected_map);
        }
    }

    void dumpStaticLines() {
        // 将所有静态多边形转换为线段
        for(int64_t env_grp_id=0; env_grp_id<scene_desc_->getNumEnvGroup(); env_grp_id++) {
            const auto &static_obj_defs = scene_desc_->at(env_grp_id).static_obj_defs_;
            uint32_t num_static_lines = 0;
            for (auto &static_obj : static_obj_defs) {
                const auto &shape = static_obj.first;
                const auto &pose = static_obj.second;

                if (shape->type == ShapeType::POLYGON) {
                    auto poly = reinterpret_cast<PolygonShapeDef *>(shape.get());
                    for (size_t j=0; j<poly->vertices.size(); j++) {

                        const Vector2 vertex = poly->vertices[j];
                        const Vector2 next_vertex = poly->vertices[(j+1) % poly->vertices.size()];
                        Vector2 new_vertex = pose.localPointTransform(vertex);
                        Vector2 new_next_vertex = pose.localPointTransform(next_vertex);

                        float4* static_line_data = reinterpret_cast<float4*>(static_lines_->at(env_grp_id, num_static_lines, 0).data());
                        *static_line_data = make_float4(new_vertex.x(), new_vertex.y(), new_next_vertex.x(), new_next_vertex.y());
                        num_static_lines++;
                    }
                }
            }
            num_static_lines_->at(env_grp_id) = num_static_lines;
        }
    }

    void dumpDynamicLines(core::SimulatorContext *context) {
        // 将所有静态多边形转换为线段
        const uint32_t num_env_grp = context->getEnvironGroupManager()->getNumEnvGroup();
        const uint32_t num_env_per_grp = context->getEnvironGroupManager()->getNumEnvPerGroup();

        dim3 block(2, DUMP_DYM_LINES_CTA_SIZE/2, 1);
        dim3 grid(num_env_grp*num_env_per_grp / DUMP_DYM_LINES_CTA_SIZE, num_dynobj_lines_, 1);

        dumpDynamicLinesKernel<<<grid, block>>>(
                num_dynobj_lines_,
                dyn_obj_lines_->typed_data<float>(),
                dyn_obj_line_ids_->typed_data<uint32_t>(),
                dyn_poses_->typed_data<float>(),
                dyn_lines_->typed_data<float>());
    }

    core::TensorHandle *getDynamicLines() { return dyn_lines_; }
    core::TensorHandle *getDynamicPoses() { return dyn_poses_; }

private:
    // 场景描述，存储所有的基本物体信息
    core::EGHostMemConfigItem<SceneDescription> *scene_desc_;

    // 静态物体
    core::EGConstMemConfigItem<uint32_t>    *num_static_lines_;         // 静态多边形物体，外轮廓线段数量，用于激光雷达
    core::EGGlobalMemConfigTensor<float>    *static_lines_;             // 静态多边形物体，外轮廓线段端点，用于激光雷达
    core::EGGlobalMemConfigTensor<float>    *static_esdf_;              // 静态物体形成的ESDF: [env_grp, width, height, 4]

    // 动态物体，所有环境共享相同数量与形状的动态物体，但是每个环境的动态物体位姿可以不同

    std::vector<std::unique_ptr<ShapeDef>>  dynamic_obj_defs_;          // 动态物体定义
    uint32_t                                num_dynobj_lines_;          // 动态物体多边形线段数量
    core::TensorHandle *dyn_obj_line_ids_ = nullptr;                    // 点集合对应的物体id: [vertex, 1]，[16位物体id, 16位内部点id]
    core::TensorHandle *dyn_obj_lines_ = nullptr;                       // 动态物体多边形线段集合（局部坐标系）: [line, 4]

    // 为每个环境准备的数据
    core::TensorHandle *dyn_poses_ = nullptr;                           // 动态物体位姿集合: [obj, env_grp, env, 4]
    core::TensorHandle *dyn_lines_ = nullptr;                           // 动态物体线段集合: [env_grp, env, lines, 4]
    // TODO. 动态物体的初始位姿，放到onReset中初始化
};

} // namespace geometry
} // namespace cuda_simulator

#endif // CUDASIM_GEOMETRY_WORLD_MANAGER_HH