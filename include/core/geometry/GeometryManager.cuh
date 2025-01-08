#ifndef CUDASIM_GEOMETRY_GEOMGETRY_MANAGER_HH
#define CUDASIM_GEOMETRY_GEOMGETRY_MANAGER_HH

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>
#include <cuda.h>

#include "cuda_helper.h"
#include "transform.hh"
#include "shapes.hh"
#include "core/storage/TensorRegistry.hh"
#include "core/EnvGroupManager.hh"
#include "core/SimulatorContext.hh"

namespace cuda_simulator {
namespace geometry {


class GeometryManager {
    constexpr static uint32_t MAX_STATIC_LINES = 16384;
    constexpr static uint32_t MAX_POLYGONS = 1024;
    constexpr static uint32_t MAX_DYNAMIC_LINES = MAX_POLYGONS*8;


    constexpr static float GRIDMAP_RESOLU = 0.1;
    constexpr static float GRIDMAP_WIDTH = 10;
    constexpr static float GRIDMAP_HEIGHT = 10;
    constexpr static uint32_t GRIDMAP_GRIDSIZE_X = GRIDMAP_WIDTH / GRIDMAP_RESOLU;
    constexpr static uint32_t GRIDMAP_GRIDSIZE_Y = GRIDMAP_HEIGHT / GRIDMAP_RESOLU;

    struct SceneDescription {
        // 静态障碍物
        std::vector<std::pair<std::unique_ptr<ShapeDef>, Transform2D>> static_obj_defs_;
        // 动态障碍物
        std::vector<std::pair<std::unique_ptr<ShapeDef>, Transform2D>> dynamic_obj_defs_;
    };

// TODO. 加一个handle，用于从poses中提取某一个物体的位姿数组

public:
    GeometryManager() {
        // 向环境组管理器声明参数
        // 参数：静态障碍物（静态障碍物数量，线段，esdf，gridmap）
        // 参数：动态障碍物（多边形数量，多边形点数集合，多边形点集合）

        // 从TensorRegistry申请动态属性：所有动态障碍物的位姿

    }

    ~GeometryManager() = default;

    void createStaticPolyObj(int environ_group_id, const PolygonShapeDef& polygon_def, const Transform2D& pose) {
        scene_desc_->at(environ_group_id).static_obj_defs_.push_back(
            std::make_pair(std::make_unique<PolygonShapeDef>(polygon_def), pose));
    }

    void createDynamicPolyObj(int environ_group_id, const PolygonShapeDef& polygon_def, const Transform2D& pose) {
        scene_desc_->at(environ_group_id).dynamic_obj_defs_.push_back(
            std::make_pair(std::make_unique<PolygonShapeDef>(polygon_def), pose));
    }

    void onRegister(core::SimulatorContext* context) {
        core::EnvGroupManager* env_grp_mgr = context->getEnvironGroupManager();
        // 主机内存，环境组参数
        scene_desc_ = env_grp_mgr->registerConfigItem<SceneDescription, core::MemoryType::HOST_MEM>( "scene_desc");

        // 静态障碍物，环境组参数
        num_static_lines_ = env_grp_mgr->registerConfigItem<uint32_t, core::MemoryType::CONSTANT_GPU_MEM>(
            "num_static_lines");
        static_lines_ = env_grp_mgr->registerConfigTensor<float>(
            "static_lines",
            {MAX_STATIC_LINES, 4});
        static_esdf_ = env_grp_mgr->registerConfigTensor<float>(
            "static_esdf",
             {MAX_STATIC_LINES, GRIDMAP_GRIDSIZE_X, GRIDMAP_GRIDSIZE_Y, 4});

        // 动态障碍物，环境组参数
        dynobj_vert_id_prefixsum_ = env_grp_mgr->registerConfigTensor<uint32_t>(
            "num_dynobj_vertices",
            {MAX_POLYGONS, 1});
        dyn_obj_vertices_ = env_grp_mgr->registerConfigTensor<float>(
            "dyn_obj_vertices",
            {MAX_DYNAMIC_LINES, 2});
    }

    void onStart(core::SimulatorContext* context) {

        core::EnvGroupManager* env_grp_mgr = context->getEnvironGroupManager();

        // 一个环境组（32~1024个环境），固定数量的静态障碍物+固定数量的动态障碍物。每个障碍物的描述是：多边形点集合（多边形障碍物）/半径（圆障碍物）；
        // 每个静态障碍物的pose是固定的，在onStart的时候全部转换为ESDF+GridMap，以及线段集合
        // 每个动态障碍物对应一个pose，需要动态运行时转换为线段集合
        buildStaticSDF();

        // 场景中所有的线段
        dyn_lines_ = env_grp_mgr->createTensor<float>("scene_lines", {MAX_STATIC_LINES + MAX_DYNAMIC_LINES, 4});
        // 场景中所有dyn object的pose
        dyn_poses_ = env_grp_mgr->createTensor<float>("dynamic_poses", {MAX_POLYGONS, 4});
    }

    // 每一步在激光雷达之前，更新得到场景中的线段,紧跟staticLines数据之后
    void dumpDynamicLines() {
        // 将所有静态多边形转换为线段
    }


    void buildStaticSDF() {

    }

    void onEnvGroupActive(core::SimulatorContext* context, int env_grp_id) {
        // 当Envgroup从非激活状态变为激活状态时调用
    }


private:
    // GeometryManager不能存储具体的障碍物，每个环境组障碍物都不一样

    // 场景描述，存储所有的基本物体信息
    core::EGHostMemConfigItem<SceneDescription> *scene_desc_;

    // 静态障碍物
    core::EGConstMemConfigItem<uint32_t> *num_static_lines_;            // 静态障碍物线段
    core::EGGlobalMemConfigTensor<float> *static_lines_;                // 静态障碍物线段
    core::EGGlobalMemConfigTensor<float> *static_esdf_;                 // 静态障碍物ESDF: [env_grp, env, width, height, 4]

    // 动态障碍物
    core::EGGlobalMemConfigTensor<uint32_t> *dynobj_vert_id_prefixsum_; // 动态障碍物多边形点数: [env_grp, env, obj, 1]
    core::EGGlobalMemConfigTensor<float> *dyn_obj_vertices_;            // 动态障碍物多边形点集合: [env_grp, env, vertex, 2]

    // 为每个环境准备的数据
    core::TensorHandle *dyn_lines_;                                   // 动态障碍物线段集合
    core::TensorHandle *dyn_poses_;                                   // 动态障碍物位姿集合
};

}
}

#endif // CUDASIM_GEOMETRY_WORLD_MANAGER_HH