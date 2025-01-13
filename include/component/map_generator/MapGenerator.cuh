
#include <algorithm>
#include <cstdlib>
#include <vector>

#include "geometry/geometry_types.hh"
#include "geometry/shapes.hh"
#include "geometry/GeometryManager.cuh"
#include "core/SimulatorContext.hh"
#include "core/Component.hh"

#include "mapgen_generate.hh"
#include "mapgen_postprocess.hh"

namespace cuda_simulator {
namespace map_gen {

class MapGenerator : public core::Component {
public:
    MapGenerator(float map_width, float map_height, float grid_size)
        : core::Component("map_generator"), MAP_WIDTH(map_width), MAP_HEIGHT(map_height), GRID_SIZE(grid_size) {

    };
    ~MapGenerator() = default;

    // void
    void onEnvironGroupInit() override{
        const int num_group = core::getEnvGroupMgr()->getNumGroup();
        // 生成地图
        CellularAutomataGenerator map_generator(MAP_WIDTH/GRID_SIZE, MAP_HEIGHT/GRID_SIZE);

        for (int i = 0; i < num_group; i++) {
            map_generator.generate();
            impl::GridMap map = map_generator.getMap();
            auto shapes = MapPostProcess::gridMapToLines(map, GRID_SIZE);

            // 为每个形状生成ShapeDef并添加到GeometryManager
            for(const impl::Shape<float>& polygons : shapes) {
                // 外边界逆时针排列，按原顺序排布点
                const impl::SimplePoly<float>& outter_shape = polygons.front();
                core::geometry::SimplePolyShapeDef outter_shape_def{
                    std::vector<core::geometry::Vector2f>(outter_shape.begin(), outter_shape.end())};

                // 当只有一个边界时，按照朴素多边形处理
                if(polygons.size() == 1) {
                    core::getGeometryManager()->createStaticPolyObj(i, outter_shape_def, {});
                    continue;
                }

                // 内边界顺时针排列，按逆顺序排布点
                const impl::SimplePoly<float>& inner_shape = polygons.back();
                core::geometry::SimplePolyShapeDef inner_shape_def{
                    std::vector<core::geometry::Vector2f>(inner_shape.rbegin(), inner_shape.rend())};

                core::geometry::ComposedPolyShapeDef composed_shape{
                    {outter_shape_def},
                    {inner_shape_def}
                };

                core::getGeometryManager()->createStaticPolyObj(i, composed_shape, {});
            }
        }
    }

    void onNodeInit() override { }
    void onNodeExecute(const core::NodeExecInputType &input, core::NodeExecOutputType &output) override { }
    void onNodeReset(const core::TensorHandle& reset_flags, core::NodeExecStateType &state) override { }
private:
    const float MAP_WIDTH;
    const float MAP_HEIGHT;
    const float GRID_SIZE;
};



} // namespace mapgen
} // namespace cuda_simulator
