#ifndef CUDASIM_COMPONENT_MAPGEN_CUH
#define CUDASIM_COMPONENT_MAPGEN_CUH

#include "core/Component.hh"

namespace cuda_simulator {
namespace map_gen {

class MapGenerator : public core::Component {
public:
    MapGenerator(float map_width, float map_height, float grid_size)
        : core::Component("map_generator"), MAP_WIDTH(map_width), MAP_HEIGHT(map_height), GRID_SIZE(grid_size) {
    };
    ~MapGenerator() = default;

    // void
    void onEnvironGroupInit() override;

    void onNodeInit() override { }
    void onNodeExecute(const core::NodeExecInputType& , core::NodeExecOutputType &) override { }
    void onNodeReset(const core::TensorHandle& , core::NodeExecStateType &) override { }
private:
    const float MAP_WIDTH;
    const float MAP_HEIGHT;
    const float GRID_SIZE;
};


} // namespace mapgen
} // namespace cuda_simulator

#endif // CUDASIM_COMPONENT_MAPGEN_CUH