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

#include "core/SimulatorContext.hh"
#include "core/EnvGroupManager.cuh"

#include "core/storage/GTensorConfig.hh"
#include "core/storage/Scalar.hh"
#include "core/storage/TensorRegistry.hh"

#include "geometry_types.hh"
#include "shapes.hh"
#include "GridMapGenerator.hh"


namespace cuda_simulator {
namespace core {
namespace geometry {

#define DUMP_DYM_LINES_CTA_SIZE (256u)

__global__ void transformDynamicLinesKernel(
        const uint32_t num_dynobj_lines,
        const uint32_t num_envs,
        const float     * __restrict__ dyn_obj_lines,       // 多边形边集合
        const uint32_t  * __restrict__ dyn_obj_line_ids,    // 多边形边集合对应的物体ID
        const float     * __restrict__ dyn_obj_poses,       // 多边形形状位姿，[obj, group, env]
        float           * __restrict__ dyn_lines);

class GeometryManager {
    constexpr static uint32_t MAX_STATIC_LINES = 16384;

    constexpr static float GRIDMAP_RESOLU = 0.002;
    constexpr static float GRIDMAP_WIDTH = 3;
    constexpr static float GRIDMAP_HEIGHT = 3;
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
        TensorHandle getShapePose() {
            return manager_->dyn_poses_[obj_id_];
        }
    protected:
        DynamicObjectProxy(int obj_id, GeometryManager *manager) :
            obj_id_(obj_id), manager_(manager) {}
    private:
        int obj_id_;
        GeometryManager* manager_;
    };

    GeometryManager() {
        EnvGroupManager *group_mgr = getEnvGroupMgr();
        // 主机内存，环境组参数
        static_scene_descs_ = group_mgr->registerConfigItem<StaticSceneDescription, MemoryType::HOST_MEM>("scene_desc");
        // 静态物体，环境组参数
        num_static_lines_ = group_mgr->registerConfigItem<uint32_t, MemoryType::CONSTANT_GPU_MEM>("num_static_lines");
        static_lines_ = group_mgr->registerConfigTensor<float>("static_lines", {MAX_STATIC_LINES, 4});
        GridMapDescription grid_map_desc = GridMapGenerator(
        {
                GRIDMAP_WIDTH,
                GRIDMAP_HEIGHT,
                {0, 0},
                GRIDMAP_RESOLU
            }).getGridMapDescritpion();
        static_esdf_ = group_mgr->registerConfigTensor<float>("static_esdf",{grid_map_desc.grid_size.y, grid_map_desc.grid_size.x, 4});
    }
    ~GeometryManager() = default;

    // 在指定环境组中创建静态物体
    template<typename ShapeType>
    void createStaticPolyObj(int group_id, const ShapeType &polygon_def, const Transform2D &pose) {
        static_scene_descs_->at(group_id).push_back(
            std::make_pair(std::make_unique<ShapeType>(polygon_def), pose));
    }

    // 在所有环境组中创建动态物体
    DynamicObjectProxy createDynamicPolyObj(const SimplePolyShapeDef &polygon_def) {
        // 累加动态物体的边数
        num_dyn_shape_lines_ += polygon_def.vertices.size();
        int obj_id = dyn_scene_desc_.size();
        dyn_scene_desc_.push_back(std::make_unique<SimplePolyShapeDef>(polygon_def));
        return DynamicObjectProxy(obj_id, this);
    }

    // 组装环境，生成静态物体的SDF，以及静态物体的线段
    void assemble() {
        // 初始化静态物体相关信息：SDF+Lines
        renderStaticEDF();
        assembleStaticWorld();

        // 动态物体
        assembleDynamicWorld();

    }

    // 更新环境状态，包括动态物体的线段
    void execute() {
        // 将动态物体转换为线段
        transformDynamicLines();
    }

    TensorHandle getDynamicLines() { return dyn_lines_; }
    TensorHandle getDynamicPoses() { return dyn_poses_; }
    TensorHandle getStaticESDF(int group_id) {
        return static_esdf_->at(group_id);
    }

protected:
    void renderStaticEDF() {
        // 为所有的场景组生成静态物体的SDF
        for (int64_t group_id=0; group_id<static_scene_descs_->getNumEnvGroup(); group_id++) {
            GridMapGenerator grid_map({GRIDMAP_WIDTH, GRIDMAP_HEIGHT, {0, 0}, GRIDMAP_RESOLU});
            for (auto &static_obj : static_scene_descs_->at(group_id)) {
                auto &shape = static_obj.first;
                auto &pose = static_obj.second;

                if (shape->type == ShapeType::SIMPLE_POLYGON) {
                    auto poly = reinterpret_cast<SimplePolyShapeDef *>(shape.get());
                    grid_map.drawPolygon(*poly, pose);
                } else if(shape->type == ShapeType::COMPOSED_POLYGON) {
                    auto poly = reinterpret_cast<ComposedPolyShapeDef *>(shape.get());
                    grid_map.drawPolygon(*poly, pose);
                } else {
                    throw std::runtime_error("Shape Type Not Support at Present!");
                }
            }
            TensorHandle map_tensor = static_esdf_->at(group_id);
            grid_map.fastEDT(map_tensor);
        }
    }

    void assembleStaticWorld() {
        /// 将所有静态多边形转换为线段
        // 遍历所有env group的数据
        for(int64_t group_id=0; group_id < static_scene_descs_->getNumEnvGroup(); group_id++) {
            // 本group的场景中 static线段的数量
            uint32_t num_static_lines_in_group = 0;
            // 获得当前场景的写入地址
            TensorHandle static_line_tensor = static_lines_->at(group_id, num_static_lines_in_group, 0);
            if(!static_line_tensor.is_contiguous())
                throw std::runtime_error("static_line_tensor by every env is not contiguous!");
            float4* static_line_data = reinterpret_cast<float4*>(static_line_tensor.data());

            auto poly_handle_fn = [&num_static_lines_in_group, &static_line_data]<typename T>(const T* shape, const Transform2D &transform) {
                for(auto line_iter = shape->begin(transform); line_iter != shape->end(transform); ++line_iter) {

                    if(num_static_lines_in_group >= MAX_STATIC_LINES)
                        throw std::runtime_error("Number of Static Lines exceeds the container capacity!");

                    static_line_data[num_static_lines_in_group++] = make_float4(
                        (*line_iter).start.x, (*line_iter).start.y,
                        (*line_iter).end.x, (*line_iter).end.y);
                }
            };


            for (auto &static_obj : static_scene_descs_->at(group_id)) {
                const auto &shape = static_obj.first;
                if(shape->type == ShapeType::SIMPLE_POLYGON) {          // 简单多边形
                    poly_handle_fn(reinterpret_cast<SimplePolyShapeDef *>(shape.get()), static_obj.second);
                } else if(shape->type == ShapeType::COMPOSED_POLYGON) {     // 复合多边形
                    poly_handle_fn(reinterpret_cast<ComposedPolyShapeDef *>(shape.get()), static_obj.second);
                } else {
                    throw std::runtime_error("Shape Type Not Support at Present!");
                }
            }
            num_static_lines_->at(group_id) = num_static_lines_in_group;
        }
    }

    void assembleDynamicWorld() {
        /// Dynamic Object 仅保留多边形的凸包

        std::vector<float4> h_dyn_shape_lines(num_dyn_shape_lines_);
        std::vector<uint32_t> h_dyn_shape_line_ids(num_dyn_shape_lines_);

        int line_idx = 0;

        auto poly_handle_fn = [&h_dyn_shape_line_ids, &h_dyn_shape_lines, &line_idx]<typename T>(const T* shape, int shape_id) {
            for(auto line_iter = shape->begin({}); line_iter != shape->end({}); ++line_iter) {
                // 线段信息
                h_dyn_shape_lines[line_idx] = make_float4((*line_iter).start.x, (*line_iter).start.y,
                                                            (*line_iter).end.x, (*line_iter).end.y);
                // 线段所属的obj id
                h_dyn_shape_line_ids[line_idx] = (uint32_t)shape_id << 16;
                line_idx++;
            }
        };

        for(size_t dyn_shape_id = 0; dyn_shape_id < dyn_scene_desc_.size(); dyn_shape_id++) {
            const std::unique_ptr<ShapeDef>& shape = dyn_scene_desc_[dyn_shape_id];

            if(shape->type == ShapeType::SIMPLE_POLYGON) {          // 简单多边形
                poly_handle_fn(reinterpret_cast<SimplePolyShapeDef *>(shape.get()), dyn_shape_id);
            } else if(shape->type == ShapeType::COMPOSED_POLYGON) {     // 复合多边形
                poly_handle_fn(reinterpret_cast<ComposedPolyShapeDef *>(shape.get()), dyn_shape_id);
            } else {
                throw std::runtime_error("Shape Type Not Support at Present!");
            }
        }

        // 拷贝dyn_shape_lines_
        TensorRegistry::getInstance().createTensor<float>(dyn_shape_lines_, "dyn_obj_lines", {});
        dyn_shape_lines_.fromHostArray(h_dyn_shape_lines.data(), NumericalDataType::kFloat32, h_dyn_shape_lines.size() * 4);
        dyn_shape_lines_.reshape({num_dyn_shape_lines_, 4});

        // 拷贝dyn_shape_line_ids_
        TensorRegistry::getInstance().createTensor<uint32_t>(dyn_shape_line_ids_, "dyn_obj_line_ids", {});
        dyn_shape_line_ids_.fromHostVector(h_dyn_shape_line_ids);

        // 申请动态数组，dyn_lines，单个场景中的所有全局坐标系线段, [group, env, lines, 4]
        // TODO. 取代这个，在kernel中实时计算
        getContext()->getEnvGroupMgr()->createTensor<float>(
            dyn_lines_, "scene_lines", {
                EnvGroupManager::SHAPE_PLACEHOLDER_GROUP,
                EnvGroupManager::SHAPE_PLACEHOLDER_ENV,
                num_dyn_shape_lines_,
                4
            }
        );

        // 场景中所有dyn object的pose, [obj, group, env, 4]
        getContext()->getEnvGroupMgr()->createTensor<float>(
            dyn_poses_, "dynamic_poses", {
                static_cast<int64_t>(dyn_scene_desc_.size()),
                EnvGroupManager::SHAPE_PLACEHOLDER_GROUP,
                EnvGroupManager::SHAPE_PLACEHOLDER_ENV,
                4
            }
        );

        std::cout << "Dynamic Object Line: " << dyn_shape_lines_ << std::endl;
        std::cout << "Dynamic Object Line Ids: " << dyn_shape_line_ids_ << std::endl;
    }

    void transformDynamicLines() {
        // 将所有静态多边形转换为线段
        const uint32_t num_group = getEnvGroupMgr()->getNumGroup();
        const uint32_t num_env_per_group = getEnvGroupMgr()->getNumEnvPerGroup();

        if (num_dyn_shape_lines_ == 0) return;

        uint32_t num_envs = num_group*num_env_per_group;
        uint32_t blocksize_y = std::min(num_envs, static_cast<uint32_t>(DUMP_DYM_LINES_CTA_SIZE/2));
        uint32_t grid_x = std::max((num_envs+blocksize_y-1) / blocksize_y, static_cast<uint32_t>(1u));
        dim3 block(2, blocksize_y, 1);
        dim3 grid(grid_x, num_dyn_shape_lines_, 1);

        transformDynamicLinesKernel<<<grid, block>>>(
                num_dyn_shape_lines_,
                num_envs,
                dyn_shape_lines_.typed_data<float>(),
                dyn_shape_line_ids_.typed_data<uint32_t>(),
                dyn_poses_.typed_data<float>(),
                dyn_lines_.typed_data<float>());

        // TODO. 现在不需要这个
        cudaDeviceSynchronize();

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA Error: %s\n", cudaGetErrorString(err));
        }
    }

private:
    // 静态物体描述，带位姿，随环境组变化
    using StaticSceneDescription = std::vector<std::pair<std::unique_ptr<ShapeDef>, Transform2D>>;
    // 动态物体描述，不带位姿，全局共享
    using DynamicSceneDescription = std::vector<std::unique_ptr<ShapeDef>>;

    // 静态物体
    EGHostMemConfigItem<StaticSceneDescription> *static_scene_descs_ = nullptr;
    EGConstMemConfigItem<uint32_t> *num_static_lines_=nullptr; // 静态多边形物体，外轮廓线段数量，用于激光雷达
    EGGlobalMemConfigTensor<float> *static_lines_=nullptr;     // 静态多边形物体，外轮廓线段端点，用于激光雷达
    EGGlobalMemConfigTensor<float> *static_esdf_=nullptr;      // 静态物体形成的ESDF: [group, width, height, 4]

    // 动态物体
    DynamicSceneDescription dyn_scene_desc_;                        // 动态物体定义（仅支持多边形）
    uint32_t                num_dyn_shape_lines_ = 0;               // 多边形线段数量
    TensorHandle      dyn_shape_line_ids_;                    // 点集合对应的物体id: [line] -> {16位物体id, 16位内部点id}
    TensorHandle      dyn_shape_lines_;                       // 多边形线段集合（局部坐标系）: [line, 4]

    // 为每个环境准备的数据
    TensorHandle      dyn_poses_;                             // 动态物体位姿集合: [obj, group, env, 4]
    TensorHandle      dyn_lines_;                             // 动态物体线段集合: [group, env, lines, 4]

    // TODO. 动态物体的初始位姿，放到onReset中初始化

    // TODO. 静态2d预渲染

    // TODO. 静态/动态物体的加速结构, BVH, 四叉树

};

} // namespace geometry
} // namespace core
} // namespace cuda_simulator

#endif // CUDASIM_GEOMETRY_WORLD_MANAGER_HH