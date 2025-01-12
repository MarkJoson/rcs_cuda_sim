#include "core/Component.hh"
#include "core/SimulatorContext.hh"

namespace cuda_simulator {
namespace mapgen {



class MapgenComponent : public core::Component {
public:
    MapgenComponent() = default;
    ~MapgenComponent() = default;

    // void
    void onEnvironGroupInit(core::SimulatorContext* context) = 0;

};



} // namespace mapgen
} // namespace cuda_simulator
