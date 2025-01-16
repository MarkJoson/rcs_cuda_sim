#include <algorithm>
#include <cstdlib>
#include <vector>

#include "geometry/GeometryManager.cuh"
#include "core/SimulatorContext.hh"

#include "component/MapGenerator.hh"

#include "mapgen_generate.hh"
#include "mapgen_postprocess.hh"

namespace cuda_simulator {
namespace map_gen {

void MapGenerator::onEnvironGroupInit() {
  const int num_group = core::getEnvGroupMgr()->getNumGroup();
  // 生成地图
  CellularAutomataGenerator map_generator(MAP_WIDTH / GRID_SIZE,
                                          MAP_HEIGHT / GRID_SIZE);

  for (int i = 0; i < num_group; i++) {
    map_generator.generate();
    impl::GridMap map = map_generator.getMap();
    auto shapes = MapPostProcess::gridMapToLines(map, GRID_SIZE);

    // 为每个形状生成ShapeDef并添加到GeometryManager
    for (const impl::Shape<float> &polygons : shapes) {
      // 外边界逆时针排列，按原顺序排布点
      const impl::SimplePoly<float> &outter_shape = polygons.front();
      core::geometry::SimplePolyShapeDef outter_shape_def{
          std::vector<core::geometry::Vector2f>(outter_shape.begin(),
                                                outter_shape.end())};

      // 当只有一个边界时，按照朴素多边形处理
      if (polygons.size() == 1) {
        core::getGeometryManager()->createStaticPolyObj(i, outter_shape_def,
                                                        {});
        continue;
      }

      // 内边界顺时针排列，按逆顺序排布点
      const impl::SimplePoly<float> &inner_shape = polygons.back();
      core::geometry::SimplePolyShapeDef inner_shape_def{
          std::vector<core::geometry::Vector2f>(inner_shape.rbegin(),
                                                inner_shape.rend())};

      core::geometry::ComposedPolyShapeDef composed_shape{{outter_shape_def},
                                                          {inner_shape_def}};

      core::getGeometryManager()->createStaticPolyObj(i, composed_shape, {});
    }
  }
}

} // namespace mapgen
} // namespace cuda_simulator
