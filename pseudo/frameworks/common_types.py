import torch as th
from typing import Sequence, NewType
from enum import Enum
from torch import Tensor
from typing import Sequence, Mapping, NewType, Callable, Optional, Tuple, List, Deque, Dict
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

__all__ = ['RawTensor', 'TensorShape', 'MessageID', 'ComponentID', 'GraphID', 'ReduceMethod',
           'Tensor', 'TYPE_CHECKING',
           'ABC', 'abstractmethod',
           'Sequence', 'Mapping', 'NewType', 'Callable', 'Optional', 'Tuple', 'List', 'Deque', 'Dict',
           'dataclass', 'field']


RawTensor       = NewType('RawTensor', th.Tensor)
TensorShape     = NewType('TensorShape', Sequence[int])
MessageID       = NewType('MessageID', str)
ComponentID     = NewType('ComponentID', str)
GraphID         = NewType('GraphID', str)            # 不同的执行流，所有的subscriber构成了一张依赖图，但在一步内可能将依赖图分为多个子图执行

TYPE_CHECKING = False

class ReduceMethod(Enum):
    STACK       = 0         # 堆叠
    REPLACE     = 1         # 替换
    SUM         = 2         # 求和
    MAX         = 3         # 求最大值
    MIN         = 4         # 求最小值
    AVERAGE     = 5         # 求平均值

