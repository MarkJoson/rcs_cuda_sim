#include "core/SimulatorContext.hh"
#include "core/MessageBus.hh"
#include "component/map_generator/MapGenerator.cuh"


using namespace cuda_simulator;
using namespace cuda_simulator::core;


int main() {
    getContext()->initialize();
    map_gen::MapGenerator *map_generator = getContext()->createComponent<map_gen::MapGenerator>(3, 3, 0.05);
    // todo. 统一地图尺寸
    getContext()->setup();

    return 0;
}

