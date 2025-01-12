#include "core/Component.hh"
#include "core/SimulatorContext.hh"

namespace cuda_simulator {
namespace mapgen {



class MapgenComponent : public core::Component {
public:
    MapgenComponent() = default;
    ~MapgenComponent() = default;

    // void
    void onEnvironGroupInit(core::SimulatorContext* context) {
        auto map_generator = std::make_unique<map_gen::CellularAutomataGenerator>(MAP_WIDTH, MAP_HEIGHT);
        // auto map_generator = std::make_unique<MessyBSPGenerator>(MAP_WIDTH, MAP_HEIGHT);
        map_generator->generate();
        auto map = map_generator->getMap();
        auto shapes = map_gen::processGridmap(map, GRID_SIZE);

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
    }

};



} // namespace mapgen
} // namespace cuda_simulator
