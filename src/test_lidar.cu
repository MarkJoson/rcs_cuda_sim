#include "core/SimulatorContext.hh"
#include "core/MessageBus.hh"
#include "component/map_generator/MapGenerator.cuh"
#include "component/LidarSensor.cuh"
#include "component/RobotEntry.cuh"

using namespace cuda_simulator;
using namespace cuda_simulator::core;


int main() {
    getContext()->initialize();
    map_gen::MapGenerator *map_generator = getContext()->createComponent<map_gen::MapGenerator>(3, 3, 0.05);
    robot_entry::RobotEntry *robot_entry = getContext()->createComponent<robot_entry::RobotEntry>(1);
    lidar_sensor::LidarSensor *lidar_sensor = getContext()->createComponent<lidar_sensor::LidarSensor>();

    // todo. 统一地图尺寸
    getContext()->setup();

    getContext()->trigger("default");


    return 0;
}

