#ifndef CUDASIM_GEOMETRY_GEOMGETRY_MANAGER_HH
#define CUDASIM_GEOMETRY_GEOMGETRY_MANAGER_HH

#include <cstdint>
#include <cuda.h>
#include <memory>

#include "core/core_types.hh"
#include "geometry_types.hh"
#include "shapes.hh"

#include "core/EnvGroupManager.cuh"

namespace cuda_simulator {
namespace core {
namespace geometry {

class GeometryManager;

class DynamicObjectProxy {
  friend class GeometryManager;

public:
  const ShapeDef *getShapeDef();
  GTensor getDynObjectPose();

protected:
  DynamicObjectProxy(int obj_id, GeometryManager *manager) : obj_id_(obj_id), manager_(manager) {}

private:
  int obj_id_;
  GeometryManager *manager_;
};

class GeometryManager {
public:
  GeometryManager();
  ~GeometryManager();

  // 在指定环境组中创建静态物体
  template <typename ShapeType>
  void createStaticPolyObj(int group_id, const ShapeType &polygon_def, const Transform2D &pose) {
    pushStaticPolyObj(group_id, std::make_unique<ShapeType>(polygon_def), pose);
  }

  void pushStaticPolyObj(int group_id, std::unique_ptr<ShapeDef> &&shape_def, const Transform2D &pose);

  DynamicObjectProxy createDynamicPolyObj(const SimplePolyShapeDef &polygon_def);

  void assemble();

  void execute();

  const ShapeDef *getShapeDef(int obj_id);

  const TensorItemHandle<float> *getStaticLines() const;
  const ConstMemItemHandle<uint32_t> *getNumStaticLines() const;
  const TensorItemHandle<float> *getStaticESDF() const;

  uint32_t getNumDynLines();

  const GTensor *getDynamicLines();
  const GTensor *getDynamicPoses();
  const GTensor getDynObjectPose(int obj_id);

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace geometry
} // namespace core
} // namespace cuda_simulator

#endif // CUDASIM_GEOMETRY_GEOMGETRY_MANAGER_HH