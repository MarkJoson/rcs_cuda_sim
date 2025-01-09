#ifndef CUDASIM_GEOMETRY_WORLD_HH
#define CUDASIM_GEOMETRY_WORLD_HH

#pragma once
#include <unordered_set>
#include <memory>

#include "transform.hh"
#include "shapes.hh"


namespace cuda_simulator {
namespace geometry {

class Object2D {
public:
    Object2D(int id,
             ObjectType objType,
             const Transform2D& transform,
             ShapeType shapeType,
             std::shared_ptr<ShapeDef> shapeDef)
        : id_(id)
        , objType_(objType)
        , transform_(transform)
        , shapeType_(shapeType)
        , shapeDef_(shapeDef) {}

    // Getters
    int getId() const { return id_; }
    ObjectType getObjectType() const { return objType_; }
    const Transform2D& getTransform() const { return transform_; }
    const Vector2& getPosition() const { return transform_.getPosition(); }
    const Rotation2D& getRotation() const { return transform_.getRotation(); }
    float getRotationAngle() const { return transform_.getRotation().getAngle(); }
    ShapeType getShapeType() const { return shapeType_; }
    const std::shared_ptr<ShapeDef>& getShapeDef() const { return shapeDef_; }

private:
    int id_;
    ObjectType objType_;
    Transform2D transform_;
    ShapeType shapeType_;
    std::shared_ptr<ShapeDef> shapeDef_;
};

class WorldManager {
public:
    WorldManager() = default;

    int createObject2D(const Transform2D& transform,
                      ObjectType objType,
                      ShapeType shapeType,
                      std::shared_ptr<ShapeDef> shapeDef) {
        int objId = static_cast<int>(objects_.size());

        auto obj = std::make_shared<Object2D>(
            objId, objType, transform, shapeType, shapeDef);

        objects_.push_back(obj);

        if (objType == ObjectType::Static) {
            staticObjectSet_.insert(objId);
        } else {
            dynamicObjectSet_.insert(objId);
        }

        return objId;
    }

    std::shared_ptr<Object2D> getObject(int id) const {
        return (id >= 0 && id < objects_.size()) ? objects_[id] : nullptr;
    }

    const std::unordered_set<int>& getStaticObjects() const { return staticObjectSet_; }
    const std::unordered_set<int>& getDynamicObjects() const { return dynamicObjectSet_; }

private:
    std::vector<std::shared_ptr<Object2D>> objects_;
    std::unordered_set<int> staticObjectSet_;
    std::unordered_set<int> dynamicObjectSet_;
};

} // namespace geometry
} // namespace cuda_simulator

#endif // CUDASIM_GEOMETRY_WORLD_HH
