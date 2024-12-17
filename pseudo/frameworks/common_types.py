import numpy as np
from numpy.typing import NDArray
from typing import Sequence, NewType
from enum import Enum, auto
from torch import Tensor
from typing import Sequence, Mapping, NewType, Callable, Optional, Tuple, List, Deque, Dict, TypeAlias, Set, Union, TypeAlias
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

__all__ = ['MessageDataShape', 'MessageID', 'ComponentID', 'GraphID', 'ReduceMethod', 'ContextType', 'ContextID',
           'Tensor', 'TYPE_CHECKING',
           'ABC', 'abstractmethod',
           'Sequence', 'Mapping', 'NewType', 'Callable', 'Optional', 'Tuple', 'List', 'Deque', 'Dict', 'Set', 'Union',
           'dataclass', 'field',
           'Vec2']


MessageDataShape    : TypeAlias = Sequence[int]
MessageID           : TypeAlias = str
ComponentID         : TypeAlias = str
GraphID             : TypeAlias = str                               # 不同的执行流，所有的subscriber构成了一张依赖图，但在一步内可能将依赖图分为多个子图执行
ContextID           : TypeAlias = str

Vec2                : TypeAlias = NDArray



TYPE_CHECKING = False

class ReduceMethod(Enum):
    STACK       = 0         # 堆叠
    REPLACE     = 1         # 替换
    SUM         = 2         # 求和
    MAX         = 3         # 求最大值
    MIN         = 4         # 求最小值
    AVERAGE     = 5         # 求平均值

class ContextType(Enum):
    """上下文类型枚举"""
    SPACE_MANAGER = auto()       # 空间管理器
    TIME_MANAGER = auto()        # 时间管理器
    MESSAGE_BUS = auto()         # 消息总线
    COMPONENT = auto()           # 组件
    # 可以继续添加其他类型...
