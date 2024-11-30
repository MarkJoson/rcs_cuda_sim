#ifndef __MAPGEN_POSTPROCESS_
#define __MAPGEN_POSTPROCESS_

#include <vector>
#include <iostream>
#include <unordered_set>

namespace map_gen
{

template <typename T>
struct Point
{
    T x;
    T y;
    bool operator==(const Point<T> &other) const
    {
        return x == other.x && y == other.y;
    }
};

using Pointd = Point<int>;
using Pointf = Point<float>;

template<typename T>
static std::ostream &operator<<(std::ostream &out, const Point<T> &d)
{
    out << " [" << d.x << "," << d.y << "] ";
    return out;
}

template<typename T>
struct Edge
{
    Point<T> start;
    Point<T> end;
    bool operator==(const Edge<T> &other) const
    {
        return start == other.start && end == other.end;
    }
};

using Array2d = std::vector<std::vector<int>>;

template<typename T>
using ArrayEdge = std::vector<Edge<T>>;

using PtHash = uint64_t;
template<typename T>
using ArrayPt = std::vector<Point<T>>;

template<typename T>
using Poly = ArrayPt<T>;

template<typename T>
using ArrayPoly = std::vector<ArrayPt<T>>;

template<typename T>
using Shape = ArrayPoly<T>;

template<typename T>
using ArrayShape = std::vector<Shape<T>>;

ArrayShape<float> processGridmap(const Array2d &map, float grid_size);

} // namespace map_gen

template <>
struct std::hash<map_gen::Point<float>>
{
public:
    size_t operator()(const map_gen::Point<float> &pt) const { return ((uint64_t)pt.x << 32 | (uint64_t)pt.y); }
};

template <typename T>
class std::hash<map_gen::Edge<T>>
{
public:
    size_t operator()(const map_gen::Edge<T> &e) const {
        return std::hash<map_gen::Point<T>>()(e.start) ^ std::hash<map_gen::Point<T>>()(e.start) << 1;
    }
};



#endif //__MAPGEN_POSTPROCESS_