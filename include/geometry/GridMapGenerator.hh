#ifndef CUDASIM_GEOMETRY_GRIDMAP_GENERATOR_HH
#define CUDASIM_GEOMETRY_GRIDMAP_GENERATOR_HH

#include <algorithm>
#include <cmath>
#include <memory>

#include <cuda_runtime_api.h>

#include "core/core_types.hh"
#include "geometry/shapes.hh"

namespace cuda_simulator {
namespace core {
namespace geometry {

struct GridMapDescription {
    float resolu = 1;
    float2 origin = {0, 0};
    int2 grid_size = {0, 0};

    Vector2i world2Grid(Vector2f point) {
        int grid_x = std::max(0, int((point.x - origin.x) / resolu));
        grid_x = std::min(grid_x, grid_size.x - 1);
        int grid_y = std::max(0, int((point.y - origin.y) / resolu));
        grid_y = std::min(grid_y, grid_size.y - 1);
        return {grid_x, grid_y};
    }

    GridMapDescription(float w, float h, Vector2f ori, float res) {
        if (res <= 0)
            throw std::runtime_error("resolution must be positive");
        resolu = res;
        origin.x = ori.x;
        origin.y = ori.y;
        grid_size.x = std::ceil(w / res);
        grid_size.y = std::ceil(h / res);
    }
};

class GridMapGenerator {
public:
    GridMapGenerator(const GridMapDescription& desc = {0,0,{},1});
    void drawPolygon(const SimplePolyShapeDef& poly, const Transform2D& tf);
    void drawPolygon(const ComposedPolyShapeDef& poly, const Transform2D& tf);
    void drawCircle(const CircleShapeDef& circle, const Transform2D& tf);
    void fastEDT(TensorHandle& output);
    GridMapDescription getGridMapDescription() const;
private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace geometry
} // namespace core
} // namespace cuda_simulator

#endif // CUDASIM_GEOMETRY_GRIDMAP_GENERATOR_HH