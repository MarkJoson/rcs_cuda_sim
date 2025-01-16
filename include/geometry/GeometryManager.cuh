#ifndef CUDASIM_GEOMETRY_GEOMGETRY_MANAGER_HH
#define CUDASIM_GEOMETRY_GEOMGETRY_MANAGER_HH

#include <cstdint>
#include <cuda.h>
#include <memory>

#include "core/core_types.hh"
#include "geometry_types.hh"
#include "shapes.hh"

namespace cuda_simulator {
namespace core {
namespace geometry {

class GeometryManager;

class DynamicObjectProxy {
    friend class GeometryManager;
public:
    const ShapeDef *getShapeDef();

    TensorHandle getShapePose();
protected:
    DynamicObjectProxy(int obj_id, GeometryManager *manager) :
        obj_id_(obj_id), manager_(manager) {}

  private:
    int obj_id_;
    GeometryManager* manager_;
};

class GeometryManager {
public:
    GeometryManager();

    // 在指定环境组中创建静态物体
    template<typename ShapeType>
    void createStaticPolyObj(int group_id, const ShapeType &polygon_def, const Transform2D &pose) {
        pushStaticPolyObj(group_id, std::make_unique<ShapeType>(polygon_def), pose);
    }

    void pushStaticPolyObj(int group_id,
                           const std::unique_ptr<ShapeDef> &shape_def,
                           const Transform2D &pose);

    DynamicObjectProxy createDynamicPolyObj(const SimplePolyShapeDef &polygon_def);

    void assemble();

    void execute();

    uint32_t getNumDynLines();

    const TensorHandle &getStaticLines();

    EGConstMemConfigItem<uint32_t> *getNumStaticLines();

    const TensorHandle &getDynamicPoses();

    TensorHandle getStaticESDF(int group_id);

    const ShapeDef *getShapeDef(int obj_id);

    const TensorHandle &getDynamicLines();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};


} // namespace geometry
} // namespace core
} // namespace cuda_simulator

#endif // CUDASIM_GEOMETRY_GEOMGETRY_MANAGER_HH