#ifndef __MAPGEN_POSTPROCESS_
#define __MAPGEN_POSTPROCESS_

#include <vector>
#include <iostream>
#include <unordered_set>

namespace map_gen
{

typedef struct Point
{
    int x;
    int y;
    bool operator==(const Point &other) const
    {
        return x == other.x && y == other.y;
    }
} Pt;

static std::ostream &operator<<(std::ostream &out, const Point &d)
{
    out << " [" << d.x << "," << d.y << "] ";
    return out;
}

struct Edge
{
    Point start;
    Point end;
    bool operator==(const Edge &other) const
    {
        return start == other.start && end == other.end;
    }
};

using Array2d = std::vector<std::vector<int>>;
using ArrayEdge = std::vector<Edge>;

using PtHash = uint64_t;
using ArrayPt = std::vector<Pt>;
using Poly = ArrayPt;

using ArrayPoly = std::vector<ArrayPt>;
using Shape = ArrayPoly;

using ArrayShape = std::vector<Shape>;

ArrayShape processGridmap(const Array2d &map, float grid_size);
std::vector<std::vector<std::pair<int, int>>> cvtToPairPt(const ArrayShape& in);

} // namespace map_gen

template <>
struct std::hash<map_gen::Point>
{
public:
    size_t operator()(const map_gen::Point &pt) const { return ((uint64_t)pt.x << 32 | (uint64_t)pt.y); }
};

template <>
class std::hash<map_gen::Edge>
{
public:
    size_t operator()(const map_gen::Edge &e) const { return std::hash<map_gen::Point>()(e.start) ^ std::hash<map_gen::Point>()(e.start) << 1; }
};



#endif //__MAPGEN_POSTPROCESS_