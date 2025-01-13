import numpy as np
from common_types import *
from enum import Enum

class ShapeType(Enum):
    CircleShape     = 0,
    SegmentShape    = 1,
    PolygonShape    = 2,
    ChainShape      = 3,

@dataclass
class ShapeDef:
    ''' 多个Shape可以共用一个ShapeDef, ShapeDef存储了形状的关键信息 '''
    resitution  : float = field(default=1.0, kw_only=True)
    friction    : float = field(default=0.0, kw_only=True)

@dataclass
class CircleShapeDef(ShapeDef):
    center      : Vec2
    radius      : float

@dataclass
class SegmentShapeDef(ShapeDef):
    point1      : Vec2
    point2      : Vec2

@dataclass
class PolygonShapeDef(ShapeDef):
    vertices    : List[Vec2]   # 多边形外部顶点坐标
    centroid    : Vec2         # 形心坐标
    radius      : float        # 外切圆半径

@dataclass
class ChainSegmentDef(ShapeDef):
    chain_id    : int
    ghost_tail  : Vec2
    segment     : SegmentShapeDef
    ghost_head  : Vec2

class ObjectType(Enum):
    StaticObject = 0,
    DynamicObject = 1


@dataclass
class Rotation2D:
    angle   : float         # angle_rad, 当前坐标系到基准坐标系的角度
    s       : float         # sin
    c       : float         # cos
    def mulRotation(self, rot:"Rotation2D"):
        return Rotation2D(
            angle=self.angle+rot.angle,
            s=self.s*rot.c+self.c*rot.s,
            c=self.c*rot.c-self.s*rot.s,)

@dataclass
class Transform2D:
    p       : Vec2          # 基准坐标系下的方向向量
    q       : Rotation2D    # 当前坐标系到基准坐标系的旋转矩阵

    def localPointTransform(self, pt: Vec2):
        new_pt = pt.copy()
        new_pt[0] = pt[0] * self.q.c - pt[1] * self.q.s
        new_pt[1] = pt[0] * self.q.s + pt[1] * self.q.c
        new_pt += self.p
        return new_pt

    def inverseTransformPoint(self, pt: Vec2):
        new_pt = pt - self.p
        new_pt[0] = new_pt[0] * self.q.c + new_pt[1] * self.q.s
        new_pt[1] = -new_pt[0] * self.q.s + new_pt[1] * self.q.c
        return new_pt

    def mulTransform(self, transform: "Transform2D"):
        return Transform2D(p=self.p+transform.p, q=self.q.mulRotation(transform.q))



@dataclass
class Object2D:
    id          : int
    obj_type    : ObjectType
    transform   : Transform2D
    shape_type  : ShapeType
    shape_def   : ShapeDef
    # TODO. contact信息
    #

    def getPosition(self):
        return self.transform.p

    def getRotation(self):
        return self.transform.q

    def getRotAngle(self):
        return self.transform.q.angle

    def getTransform(self):
        return self.transform

# shapes: 缓冲区，shape_id: 缓冲区指针

@dataclass
class WorldManager:
    objs            : List[Object2D] = list()
    static_obj_set  : Set[int] = set()
    dyn_obj_set     : Set[int] = set()

    def createObject2D(self, transform: Transform2D, obj_type: ObjectType, shape_type:ShapeType, shape_def:ShapeDef):
        obj_id = len(self.objs)
        obj = Object2D(id=obj_id, obj_type=obj_type, transform=transform, shape_type=shape_type, shape_def=shape_def)
        self.objs.append(obj)

        if obj_type == ObjectType.StaticObject:
            self.static_obj_set.add(obj_id)
        elif obj_type == ObjectType.DynamicObject:
            self.dyn_obj_set.add(obj_id)

        return obj_id

