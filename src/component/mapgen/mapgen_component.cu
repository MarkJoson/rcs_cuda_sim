#include "core/Component.hh"
#include "core/SimulatorContext.hh"
#include "mapgen_generate.hh"
#include "mapgen_postprocess.hh"

namespace cuda_simulator {
namespace map_gen {



class MapgenComponent : public core::Component {
public:
    MapgenComponent(int map_width, int map_height, int grid_size)
        : core::Component("map_generator"), MAP_WIDTH(map_width), MAP_HEIGHT(map_height), GRID_SIZE(grid_size) {

    };
    ~MapgenComponent() = default;

    // void
    void onEnvironGroupInit() override{
        // 生成地图
        auto map_generator = std::make_unique<CellularAutomataGenerator>(MAP_WIDTH, MAP_HEIGHT);
        // auto map_generator = std::make_unique<MessyBSPGenerator>(MAP_WIDTH, MAP_HEIGHT);

        map_generator->generate();
        auto map = map_generator->getMap();
        auto shapes = MapPostProcess::gridMapToLines(map, GRID_SIZE);

        std::vector<float2> lbegins, lends;
        std::for_each(shapes.begin(), shapes.end(), [&lbegins, &lends](const auto& polygons){
            for(size_t pgi=0; pgi<polygons.size(); pgi++) {
                auto pg = polygons[pgi];
                pg.push_back(pg.front());
                // if(polygons.size() != 2 || (polygons.size() == 2 && pgi == 1)) continue;
                for(size_t i=0; i<pg.size()-1; i++) {
                    auto lb = make_float2(pg[i].x, pg[i].y);
                    auto le = make_float2(pg[i+1].x, pg[i+1].y);
                    // if(polygons.size() == 2)
                    std::swap(lb, le);
                    lbegins.push_back(lb);
                    lends.push_back(le);
                }
            }
        });

        // 将shape添加到GeometryManager中
        // core::getGeometryManager()
    }
private:
    const int MAP_WIDTH;
    const int MAP_HEIGHT;
    const int GRID_SIZE;
};



} // namespace mapgen
} // namespace cuda_simulator
