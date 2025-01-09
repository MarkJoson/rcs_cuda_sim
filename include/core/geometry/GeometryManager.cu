#ifndef CUDASIM_GEOMETRY_GEOMGETRY_MANAGER_HH
#define CUDASIM_GEOMETRY_GEOMGETRY_MANAGER_HH

// #include <__clang_cuda_builtin_vars.h>
#include <cstddef>
#include <cstdint>
#include <cuda.h>
#include <memory>
#include <opencv2/core/hal/interface.h>
#include <stdexcept>
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

#define DUMP_DYM_LINES_CTA_SIZE (256u)

/**
 * @brief 依据动态形状的位置，将动态形状的多边形边，转换到世界坐标系下
 * @param num_dynobj_lines 动态物体边的数量总和
 * @param dyn_obj_lines 动态物体边外形
 * @param dyn_obj_line_ids 动态物体边对应的object id
 * @param dyn_obj_poses 动态物体位置[obj_id, env_group, env, 4]
 * @param dyn_lines 输出，动态物体的边 [env_group, env, lines, 4]
 */
__global__ void transformDynamicLinesKernel(
        const uint32_t num_dynobj_lines,
        const uint32_t num_envs,
        const float     * __restrict__ dyn_obj_lines,       // 多边形边集合
        const uint32_t  * __restrict__ dyn_obj_line_ids,    // 多边形边集合对应的物体ID
        const float     * __restrict__ dyn_obj_poses,       // 多边形形状位姿，[obj, env_grp, env]
        float           * __restrict__ dyn_lines) {         // 当前多边形外形线段在坐标系的端点 [env_grp, env, lines, 4]

    // 每个block负责dyn_obj的一个line(y), 每个线程负责一个点(x)
    // const uint32_t point_id = threadIdx.x;          // 线段的点Id, 0起点，1终点
    // const uint32_t sub_batch_id = threadIdx.y;      // 子batch id, 32~256
    // const uint32_t prim_batch_id = blockIdx.x;      // 主batch id, 0~1024
    const uint32_t batch_id = blockIdx.x * blockDim.y + threadIdx.y;

    const uint32_t line_id = blockIdx.y;
    const uint32_t dyn_obj_id = dyn_obj_line_ids[line_id] >> 16;

    if (line_id >= num_dynobj_lines || batch_id >= num_envs) return;

    const uint32_t global_point_id = line_id * 2 + threadIdx.x;
    const float2 vertex = reinterpret_cast<const float2 *>(dyn_obj_lines)[global_point_id];

    // [obj_id, batch, 4]
    const float4 pose = reinterpret_cast<const float4 *>(dyn_obj_poses)[dyn_obj_id * num_envs + batch_id];
    float sint = pose.z, cost = pose.w;
    float2 new_vertex = {
            vertex.x * cost - vertex.y * sint + pose.x,
            vertex.x * sint + vertex.y * cost + pose.y
    };

    const uint32_t output_indice = (num_dynobj_lines * batch_id + line_id) * 2 + threadIdx.x;
    reinterpret_cast<float2 *>(dyn_lines)[output_indice] = new_vertex;
}


class GeometryManager {
    constexpr static uint32_t MAX_STATIC_LINES = 16384;

    constexpr static float GRIDMAP_RESOLU = 0.1;
    constexpr static float GRIDMAP_WIDTH = 10;
    constexpr static float GRIDMAP_HEIGHT = 10;
    constexpr static uint32_t GRIDMAP_GRIDSIZE_X = GRIDMAP_WIDTH / GRIDMAP_RESOLU;
    constexpr static uint32_t GRIDMAP_GRIDSIZE_Y = GRIDMAP_HEIGHT / GRIDMAP_RESOLU;

    // TODO. 加一个handle，用于从poses中提取某一个物体的位姿数组
public:
    class DynamicObjectProxy {
        friend class GeometryManager;
    public:
        ShapeDef* getShapeDef() {
            return manager_->dyn_scene_desc_[obj_id_].get();
        }
        core::TensorHandle getShapePose() {
            return (*manager_->dyn_poses_)[obj_id_];
        }
    protected:
        DynamicObjectProxy(int obj_id, GeometryManager *manager) :
            obj_id_(obj_id), manager_(manager) {}
    private:
        int obj_id_;
        GeometryManager* manager_;
    };

    GeometryManager() { }
    ~GeometryManager() = default;

    // 静态物体在指定环境组中创建
    void createStaticPolyObj(int environ_group_id, const PolygonShapeDef &polygon_def, const Transform2D &pose) {
        static_scene_descs_->at(environ_group_id).push_back(
            std::make_pair(std::make_unique<PolygonShapeDef>(polygon_def), pose));
    }

    // 动态物体在所有环境组中创建
    DynamicObjectProxy createDynamicPolyObj(const PolygonShapeDef &polygon_def) {
        // 累加动态物体的边数
        num_dyn_shape_lines_ += polygon_def.vertices.size();
        int obj_id = dyn_scene_desc_.size();
        dyn_scene_desc_.push_back(std::make_unique<PolygonShapeDef>(polygon_def));
        return DynamicObjectProxy(obj_id, this);
    }

    void onRegister(core::SimulatorContext *context) {
        core::EnvGroupManager *env_grp_mgr = context->getEnvironGroupManager();
        // 主机内存，环境组参数
        static_scene_descs_ = env_grp_mgr->registerConfigItem<StaticSceneDescription, core::MemoryType::HOST_MEM>("scene_desc");

        // 静态物体，环境组参数
        num_static_lines_ = env_grp_mgr->registerConfigItem<uint32_t, core::MemoryType::CONSTANT_GPU_MEM>("num_static_lines");
        static_lines_ = env_grp_mgr->registerConfigTensor<float>("static_lines", {MAX_STATIC_LINES, 4});
        static_esdf_ = env_grp_mgr->registerConfigTensor<float>("static_esdf",{MAX_STATIC_LINES, GRIDMAP_GRIDSIZE_X, GRIDMAP_GRIDSIZE_Y, 4});
    }

    void onStart(core::SimulatorContext *context) {
        // 初始化静态物体相关信息：SDF+Lines
        renderStaticEDF();
        assembleStaticWorld();

        // 动态物体
        assembleDynamicWorld();

        // 初始化动态物体存储空间
        core::EnvGroupManager *env_grp_mgr = context->getEnvironGroupManager();
        // 场景中所有的线段, [env_grp, env, lines, 4]
        dyn_lines_ = env_grp_mgr->createTensor<float>("scene_lines", {
            core::EnvGroupManager::SHAPE_PLACEHOLDER_ENV_GRP,
            core::EnvGroupManager::SHAPE_PLACEHOLDER_ENV,
            num_dyn_shape_lines_,
            4});
        // 场景中所有dyn object的pose, [obj, env_grp, env, 4]
        const uint32_t num_dynobj = dyn_scene_desc_.size();
        dyn_poses_ = env_grp_mgr->createTensor<float>("dynamic_poses", {
            num_dynobj,
            core::EnvGroupManager::SHAPE_PLACEHOLDER_ENV_GRP,
            core::EnvGroupManager::SHAPE_PLACEHOLDER_ENV,
            4});

    }

    void onExecute(core::SimulatorContext *context) {
        // 将动态物体转换为线段
        transformDynamicLines(context);
    }

    void renderStaticEDF() {
        // 为所有的场景组生成静态物体的SDF
        for (int64_t env_grp_id=0; env_grp_id<static_scene_descs_->getNumEnvGroup(); env_grp_id++) {
            GridMapGenerator grid_map(GRIDMAP_WIDTH, GRIDMAP_HEIGHT, {0, 0}, GRIDMAP_RESOLU);
            for (auto &static_obj : static_scene_descs_->at(env_grp_id)) {
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

    void assembleStaticWorld() {
        /// 将所有静态多边形转换为线段
        // 遍历所有env group的数据
        for(int64_t group_id=0; group_id < static_scene_descs_->getNumEnvGroup(); group_id++) {
            // 本group的场景中 static线段的数量
            uint32_t num_static_lines_in_group = 0;
            // 获得当前场景的写入地址
            core::TensorHandle static_line_tensor = static_lines_->at(group_id, num_static_lines_in_group, 0);
            if(!static_line_tensor.is_contiguous())
                throw std::runtime_error("static_line_tensor by every env is not contiguous!");
            float4* static_line_data = reinterpret_cast<float4*>(static_line_tensor.data());

            for (auto &static_obj : static_scene_descs_->at(group_id)) {
                const auto &shape = static_obj.first;
                const auto &pose = static_obj.second;

                // TODO. 当前仅支持Polygon型Static Object
                if(shape->type != ShapeType::POLYGON)
                    throw std::runtime_error("Only Support Polygon Object at Present!");

                // 将Polygon的所有边加入static_line_data
                auto poly_shape = reinterpret_cast<PolygonShapeDef *>(shape.get());
                for (size_t j=0; j<poly_shape->vertices.size(); j++) {

                    const Vector2 vertex = poly_shape->vertices[j];
                    const Vector2 next_vertex = poly_shape->vertices[(j+1) % poly_shape->vertices.size()];
                    Vector2 new_vertex = pose.localPointTransform(vertex);
                    Vector2 new_next_vertex = pose.localPointTransform(next_vertex);

                    if(num_static_lines_in_group >= MAX_STATIC_LINES)
                        throw std::runtime_error("Number of Static Lines exceeds the container capacity!");

                    static_line_data[num_static_lines_in_group++] = make_float4(new_vertex.x(), new_vertex.y(), new_next_vertex.x(), new_next_vertex.y());
                }
            }
            num_static_lines_->at(group_id) = num_static_lines_in_group;
        }
    }

    void assembleDynamicWorld() {
        dyn_shape_lines_ = core::TensorRegistry::getInstance().createTensor<float>(
                "dyn_obj_lines", {num_dyn_shape_lines_, 4});
        dyn_shape_line_ids_ = core::TensorRegistry::getInstance().createTensor<uint32_t>(
                "dyn_obj_line_ids", {num_dyn_shape_lines_});

        int line_idx = 0;
        for(size_t dyn_shape_id = 0; dyn_shape_id < dyn_scene_desc_.size(); dyn_shape_id++) {
            const std::unique_ptr<ShapeDef>& dyn_shape = dyn_scene_desc_[dyn_shape_id];
            if(dyn_shape->type != ShapeType::POLYGON)
                throw std::runtime_error("Only Support Polygon Object at Present!");

            PolygonShapeDef* poly_shape = reinterpret_cast<PolygonShapeDef *>(dyn_shape.get());

            for (size_t vertex_id=0; vertex_id<poly_shape->vertices.size(); vertex_id++) {
                const Vector2 vertex = poly_shape->vertices[vertex_id];
                const Vector2 next_vertex = poly_shape->vertices[(vertex_id+1) % poly_shape->vertices.size()];

                float4* static_line_data = reinterpret_cast<float4*>((*dyn_shape_lines_)[{line_idx, 0}].data());
                // 线段信息
                *static_line_data = make_float4(vertex.x(), vertex.y(), next_vertex.x(), next_vertex.y());
                // 线段所属的obj id
                (*dyn_shape_line_ids_)[line_idx] = (uint32_t)dyn_shape_id << 16;

                line_idx++;
            }
        }
    }

    void transformDynamicLines(core::SimulatorContext *context) {
        // 将所有静态多边形转换为线段
        const uint32_t num_env_grp = context->getEnvironGroupManager()->getNumEnvGroup();
        const uint32_t num_env_per_grp = context->getEnvironGroupManager()->getNumEnvPerGroup();

        uint32_t num_envs = num_env_grp*num_env_per_grp;
        uint32_t blocksize_y = std::min(num_envs, static_cast<uint32_t>(DUMP_DYM_LINES_CTA_SIZE/2));
        uint32_t grid_x = std::max((num_envs+blocksize_y-1) / blocksize_y, static_cast<uint32_t>(1u));
        dim3 block(2, blocksize_y, 1);
        dim3 grid(grid_x, num_dyn_shape_lines_, 1);

        transformDynamicLinesKernel<<<grid, block>>>(
                num_dyn_shape_lines_,
                num_envs,
                dyn_shape_lines_->typed_data<float>(),
                dyn_shape_line_ids_->typed_data<uint32_t>(),
                dyn_poses_->typed_data<float>(),
                dyn_lines_->typed_data<float>());
    }

    core::TensorHandle *getDynamicLines() { return dyn_lines_; }
    core::TensorHandle *getDynamicPoses() { return dyn_poses_; }

private:
    // 静态物体描述，带位姿，随环境组变化
    using StaticSceneDescription = std::vector<std::pair<std::unique_ptr<ShapeDef>, Transform2D>>;
    // 动态物体描述，不带位姿，全局共享
    using DynamicSceneDescription = std::vector<std::unique_ptr<ShapeDef>>;

    // 静态物体
    core::EGHostMemConfigItem<StaticSceneDescription> *static_scene_descs_;
    core::EGConstMemConfigItem<uint32_t> *num_static_lines_=nullptr; // 静态多边形物体，外轮廓线段数量，用于激光雷达
    core::EGGlobalMemConfigTensor<float> *static_lines_=nullptr;     // 静态多边形物体，外轮廓线段端点，用于激光雷达
    core::EGGlobalMemConfigTensor<float> *static_esdf_=nullptr;      // 静态物体形成的ESDF: [env_grp, width, height, 4]

    // 动态物体
    DynamicSceneDescription dyn_scene_desc_;                        // 动态物体定义（仅支持多边形）
    uint32_t                num_dyn_shape_lines_;                   // 多边形线段数量
    core::TensorHandle      *dyn_shape_line_ids_ = nullptr;         // 点集合对应的物体id: [line] -> {16位物体id, 16位内部点id}
    core::TensorHandle      *dyn_shape_lines_ = nullptr;            // 多边形线段集合（局部坐标系）: [line, 4]

    // 为每个环境准备的数据
    core::TensorHandle *dyn_poses_ = nullptr;                       // 动态物体位姿集合: [obj, env_grp, env, 4]
    core::TensorHandle *dyn_lines_ = nullptr;                       // 动态物体线段集合: [env_grp, env, lines, 4]

    // TODO. 动态物体的初始位姿，放到onReset中初始化

    // TODO. 静态预渲染

    // TODO. 静态/动态物体的加速结构, BVH, 四叉树

};

} // namespace geometry
} // namespace cuda_simulator

#endif // CUDASIM_GEOMETRY_WORLD_MANAGER_HH