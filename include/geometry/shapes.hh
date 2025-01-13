#ifndef CUDASIM_GEOMETRY_SHAPES_HH
#define CUDASIM_GEOMETRY_SHAPES_HH


#include <vector>
#include <memory>
#include "geometry_types.hh"

namespace cuda_simulator {
namespace core {
namespace geometry {

// ^--------------基础形状定义--------------
struct ShapeDef {
    ShapeType type;
    float restitution = 1.0f;
    float friction = 0.0f;

// ==================== 代码实现 ====================
    ShapeDef(ShapeType type, float restitution = 1.0f, float friction = 0.0f)
        : type(type)
        , restitution(restitution)
        , friction(friction) {}
};


// ^--------------圆形--------------
struct CircleShapeDef : public ShapeDef {
    Vector2f center;
    float radius;

// ==================== 代码实现 ====================
    CircleShapeDef(const Vector2f& center, float radius)
        : ShapeDef(ShapeType::CIRCLE)
        , center(center)
        , radius(radius) {}
};


// ^--------------简单多边形--------------
// 即没有洞的多边形
struct SimplePolyShapeDef : public ShapeDef {
    std::vector<Vector2f> vertices;
    std::vector<Vector2f> convex_hull;
    Vector2f centroid;
    float radius;

    SimplePolyShapeDef(const std::vector<Vector2f>& vertices)
        : ShapeDef(ShapeType::SIMPLE_POLYGON)
    {
        this->vertices = vertices;

        // 计算形心
        centroid = calcCentroid(vertices);
        // 计算外切圆半径
        radius = calcRadius(vertices, centroid);
        // 计算凸包
        convex_hull = calcConvexHull(vertices);
    }

    static Vector2f calcCentroid(const std::vector<Vector2f> &vertices) {
                float signedArea = 0.0f;
        float cx = 0.0f;
        float cy = 0.0f;
        for (size_t i = 0; i < vertices.size(); ++i) {
            Vector2f v0 = vertices[i];
            Vector2f v1 = vertices[(i + 1) % vertices.size()];
            float a = v0.x * v1.y - v1.x * v0.y;
            signedArea += a;
            cx += (v0.x + v1.x) * a;
            cy += (v0.y + v1.y) * a;
        }
        signedArea *= 0.5f;
        cx /= (6.0f * signedArea);
        cy /= (6.0f * signedArea);
        return Vector2f(cx, cy);
    }

    static float calcRadius(const std::vector<Vector2f> &vertices, const Vector2f &centroid) {
        float radius = 0.0f;
        for (const auto& vertex : vertices) {
            float dist = (vertex - centroid).length();
            if (dist > radius) {
                radius = dist;
            }
        }
        return radius;
    }

    static std::vector<Vector2f> calcConvexHull(const std::vector<Vector2f> &vertices) {
        std::vector<Vector2f> convex_hull = vertices; // 使用Graham扫描法或其他凸包算法
        std::sort(convex_hull.begin(), convex_hull.end(), [](const Vector2f& a, const Vector2f& b) {
            return a.x < b.x || (a.x == b.x && a.y < b.y);
        });
        std::vector<Vector2f> hull;
        for (const auto& point : convex_hull) {
            while (hull.size() >= 2 &&
               (hull[hull.size() - 1] - hull[hull.size() - 2]).cross(point - hull[hull.size() - 1]) <= 0) {
            hull.pop_back();
            }
            hull.push_back(point);
        }
        size_t t = hull.size() + 1;
        for (auto it = convex_hull.rbegin(); it != convex_hull.rend(); ++it) {
            while (hull.size() >= t &&
               (hull[hull.size() - 1] - hull[hull.size() - 2]).cross(*it - hull[hull.size() - 1]) <= 0) {
            hull.pop_back();
            }
            hull.push_back(*it);
        }
        hull.pop_back();
        return hull;
    }

    class LineIterator {
    public:
        LineIterator(const std::vector<Vector2f>& vertices, const Transform2D& transform, size_t index)
            : vertices_(vertices), transform_(transform), index_(index) {}

        bool operator!=(const LineIterator& other) const {
            return index_ != other.index_;
        }

        LineIterator& operator++() {
            ++index_;
            return *this;
        }

        Linef operator*() const {
            Vector2f start = transform_.localPointTransform(vertices_[index_]);
            Vector2f end = transform_.localPointTransform(vertices_[(index_ + 1) % vertices_.size()]);
            return Line(start, end);
        }

        Linef operator->() const {
            return **this;
        }

    private:
        const std::vector<Vector2f>& vertices_;
        const Transform2D& transform_;
        size_t index_;
    };

    LineIterator begin(const Transform2D& transform) const { return LineIterator(vertices, transform, 0); }
    LineIterator end(const Transform2D& transform) const { return LineIterator(vertices, transform, vertices.size()); }
};



// ^--------------多边形组合体--------------
// 所有的正方向多边形 - 负方向多边形。可以有洞。每个连通边界都是一个简单多边形
struct ComposedPolyShapeDef : public ShapeDef {
    std::vector<SimplePolyShapeDef> positive_polys;  // 正向多边形
    std::vector<SimplePolyShapeDef> negative_polys;  // 负向多边形（洞）
    std::vector<Vector2f> convex_hull;                // 凸包
    Vector2f centroid;                                // 组合体的中心点
    float radius;                                    // 外切圆半径

    ComposedPolyShapeDef(const std::vector<SimplePolyShapeDef>& positive_polys,
                         const std::vector<SimplePolyShapeDef>& negative_polys = {})
        : ShapeDef(ShapeType::COMPOSED_POLYGON)
        , positive_polys(positive_polys)
        , negative_polys(negative_polys) {
        calcProperties();
    }

    ComposedPolyShapeDef(
        const std::vector<std::vector<Vector2f>>& positive_vertices,
        const std::vector<std::vector<Vector2f>>& negative_vertices = {})
        : ShapeDef(ShapeType::COMPOSED_POLYGON) {
        // 创建正向多边形
        for (const auto& vertices : positive_vertices) {
            positive_polys.emplace_back(vertices);
        }

        // 创建负向多边形（洞）
        for (const auto& vertices : negative_vertices) {
            negative_polys.emplace_back(vertices);
        }

        // 计算组合体的属性
        calcProperties();
    }

    void addPositivePoly(const SimplePolyShapeDef& poly) {
        positive_polys.push_back(poly);
        calcProperties();
    }

    void addNegativePoly(const SimplePolyShapeDef& poly) {
        negative_polys.push_back(poly);
        calcProperties();
    }

    class LineIterator {
    public:
        LineIterator(const std::vector<SimplePolyShapeDef>& positive_polys,
                    const std::vector<SimplePolyShapeDef>& negative_polys,
                    const Transform2D& transform,
                    size_t poly_index,
                    size_t vertex_index,
                    bool is_positive)
            : positive_polys_(positive_polys)
            , negative_polys_(negative_polys)
            , transform_(transform)
            , poly_index_(poly_index)
            , vertex_index_(vertex_index)
            , is_positive_(is_positive) {}

        bool operator!=(const LineIterator& other) const {
            return poly_index_ != other.poly_index_ ||
                   vertex_index_ != other.vertex_index_ ||
                   is_positive_ != other.is_positive_;
        }

        LineIterator& operator++() {
            const auto& current_polys = is_positive_ ? positive_polys_ : negative_polys_;
            if(poly_index_ >= current_polys.size()) return *this;

            ++vertex_index_;
            const auto& current_vertices = current_polys[poly_index_].vertices;
            if(vertex_index_ >= current_vertices.size()) {
                vertex_index_ = 0;
                ++poly_index_;
                if(poly_index_ >= current_polys.size() && is_positive_) {       // 切换到负向多边形
                    is_positive_ = false;
                    poly_index_ = 0;
                }
            }
            return *this;
        }

        Linef operator*() const {
            const auto& current_polys = is_positive_ ? positive_polys_ : negative_polys_;
            const auto& vertices = current_polys[poly_index_].vertices;
            Vector2f start = transform_.localPointTransform(vertices[vertex_index_]);
            Vector2f end = transform_.localPointTransform(vertices[(vertex_index_ + 1) % vertices.size()]);

            // TODO. 当返回negetive_polys的时候，将start和end交换
            return Linef(start, end);
        }

    private:
        const std::vector<SimplePolyShapeDef>& positive_polys_;
        const std::vector<SimplePolyShapeDef>& negative_polys_;
        const Transform2D& transform_;
        size_t poly_index_;
        size_t vertex_index_;
        bool is_positive_;
    };

    LineIterator begin(const Transform2D& transform) const {
        return LineIterator(positive_polys, negative_polys, transform, 0, 0, true);
    }

    LineIterator end(const Transform2D& transform) const {
        return LineIterator(positive_polys, negative_polys, transform, negative_polys.size(), 0, false);
    }

private:
    void calcProperties() {
        // 计算加权中心点
        float total_area = 0.0f;
        Vector2f weighted_center(0.0f, 0.0f);

        // 处理正向多边形
        for (const auto& poly : positive_polys) {
            float area = calcArea(poly.vertices);
            total_area += area;
            weighted_center += poly.centroid * area;
        }

        // 处理负向多边形（洞）
        for (const auto& poly : negative_polys) {
            float area = calcArea(poly.vertices);
            total_area -= area;
            weighted_center -= poly.centroid * area;
        }

        centroid = weighted_center / total_area;

        // 计算外切圆半径
        radius = 0.0f;
        for (const auto& poly : positive_polys) {
            for (const auto& vertex : poly.vertices) {
                float dist = (vertex - centroid).length();
                radius = std::max(radius, dist);
            }
        }

        // 计算凸包
        convex_hull.clear();
        for (const auto& poly : positive_polys) {
            for (const auto& vertex : poly.convex_hull) {
                convex_hull.push_back(vertex);
            }
        }
        for (const auto& poly : negative_polys) {
            for (const auto& vertex : poly.convex_hull) {
                convex_hull.push_back(vertex);
            }
        }
        convex_hull = SimplePolyShapeDef::calcConvexHull(convex_hull);

    }

    static float calcArea(const std::vector<Vector2f>& vertices) {
        float area = 0.0f;
        for (size_t i = 0; i < vertices.size(); ++i) {
            const Vector2f& v1 = vertices[i];
            const Vector2f& v2 = vertices[(i + 1) % vertices.size()];
            area += v1.cross(v2);
        }
        return std::abs(area) * 0.5f;
    }
};

} // namespace geometry
} // namespace core
} // namespace cuda_simulator

#endif // CUDASIM_GEOMETRY_SHAPES_HH