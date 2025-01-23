import torch
from typing import ClassVar, overload

MAX: ReduceMethod
MIN: ReduceMethod
STACK: ReduceMethod
SUM: ReduceMethod
kCPU: DeviceType
kCUDA: DeviceType
kFloat32: NumericalDataType
kFloat64: NumericalDataType
kInt32: NumericalDataType
kInt64: NumericalDataType
kNumDevices: DeviceType
kUInt32: NumericalDataType
kUInt8: NumericalDataType

class Component(ExecuteNode):
    dependences_: list[str]
    def __init__(self, name: str, tag: str) -> None:
        """__init__(self: cudasim_pycore.Component, name: str, tag: str) -> None"""
    def addDependence(self, dependence: str) -> None:
        """addDependence(self: cudasim_pycore.Component, dependence: str) -> None"""
    def getDependences(self) -> list[str]:
        """getDependences(self: cudasim_pycore.Component) -> list[str]"""
    def onEnvironGroupInit(self) -> None:
        """onEnvironGroupInit(self: cudasim_pycore.Component) -> None"""

class DeviceType:
    __members__: ClassVar[dict] = ...  # read-only
    __entries: ClassVar[dict] = ...
    kCPU: ClassVar[DeviceType] = ...
    kCUDA: ClassVar[DeviceType] = ...
    kNumDevices: ClassVar[DeviceType] = ...
    def __init__(self, value: int) -> None:
        """__init__(self: cudasim_pycore.DeviceType, value: int) -> None"""
    def __eq__(self, other: object) -> bool:
        """__eq__(self: object, other: object) -> bool"""
    def __hash__(self) -> int:
        """__hash__(self: object) -> int"""
    def __index__(self) -> int:
        """__index__(self: cudasim_pycore.DeviceType) -> int"""
    def __int__(self) -> int:
        """__int__(self: cudasim_pycore.DeviceType) -> int"""
    def __ne__(self, other: object) -> bool:
        """__ne__(self: object, other: object) -> bool"""
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class EnvGroupManager:
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def getNumActiveGroup(self) -> int:
        """getNumActiveGroup(self: cudasim_pycore.EnvGroupManager) -> int"""
    def getNumEnvPerGroup(self) -> int:
        """getNumEnvPerGroup(self: cudasim_pycore.EnvGroupManager) -> int"""
    def getNumGroup(self) -> int:
        """getNumGroup(self: cudasim_pycore.EnvGroupManager) -> int"""
    def registerConfigTensor(self, name: str, shape: list[int]) -> TensorItemHandle:
        """registerConfigTensor(self: cudasim_pycore.EnvGroupManager, name: str, shape: list[int]) -> cudasim_pycore.TensorItemHandle"""
    def sampleActiveGroupIndices(self) -> None:
        """sampleActiveGroupIndices(self: cudasim_pycore.EnvGroupManager) -> None"""
    def syncToDevice(self) -> None:
        """syncToDevice(self: cudasim_pycore.EnvGroupManager) -> None"""

class ExecuteNode:
    def __init__(self, name: str, tag: str) -> None:
        """__init__(self: cudasim_pycore.ExecuteNode, name: str, tag: str) -> None"""
    def addInput(self, input: NodeOutputInfo) -> None:
        """addInput(self: cudasim_pycore.ExecuteNode, input: cudasim_pycore.NodeOutputInfo) -> None"""
    def addOutput(self, output: NodeOutputInfo) -> None:
        """addOutput(self: cudasim_pycore.ExecuteNode, output: cudasim_pycore.NodeOutputInfo) -> None"""
    def addState(self, state: NodeStateInfo) -> None:
        """addState(self: cudasim_pycore.ExecuteNode, state: cudasim_pycore.NodeStateInfo) -> None"""
    def getInputInfo(self, message_name: str) -> NodeInputInfo:
        """getInputInfo(self: cudasim_pycore.ExecuteNode, message_name: str) -> cudasim_pycore.NodeInputInfo"""
    def getInputs(self) -> dict[str, NodeInputInfo]:
        """getInputs(self: cudasim_pycore.ExecuteNode) -> dict[str, cudasim_pycore.NodeInputInfo]"""
    def getName(self) -> str:
        """getName(self: cudasim_pycore.ExecuteNode) -> str"""
    def getOutputInfo(self, message_name: str) -> NodeOutputInfo:
        """getOutputInfo(self: cudasim_pycore.ExecuteNode, message_name: str) -> cudasim_pycore.NodeOutputInfo"""
    def getOutputs(self) -> dict[str, NodeOutputInfo]:
        """getOutputs(self: cudasim_pycore.ExecuteNode) -> dict[str, cudasim_pycore.NodeOutputInfo]"""
    def getStateInfo(self, state_name: str) -> NodeStateInfo:
        """getStateInfo(self: cudasim_pycore.ExecuteNode, state_name: str) -> cudasim_pycore.NodeStateInfo"""
    def getStates(self) -> dict[str, NodeStateInfo]:
        """getStates(self: cudasim_pycore.ExecuteNode) -> dict[str, cudasim_pycore.NodeStateInfo]"""
    def getTag(self) -> str:
        """getTag(self: cudasim_pycore.ExecuteNode) -> str"""
    def onNodeExecute(self, input: dict[str, list[GTensor]], output: dict[str, GTensor], state: dict[str, GTensor]) -> None:
        """onNodeExecute(self: cudasim_pycore.ExecuteNode, input: dict[str, list[cudasim_pycore.GTensor]], output: dict[str, cudasim_pycore.GTensor], state: dict[str, cudasim_pycore.GTensor]) -> None"""
    def onNodeInit(self) -> None:
        """onNodeInit(self: cudasim_pycore.ExecuteNode) -> None"""
    def onNodeReset(self, reset_flags: GTensor, state: dict[str, GTensor]) -> None:
        """onNodeReset(self: cudasim_pycore.ExecuteNode, reset_flags: cudasim_pycore.GTensor, state: dict[str, cudasim_pycore.GTensor]) -> None"""
    def onNodeStart(self) -> None:
        """onNodeStart(self: cudasim_pycore.ExecuteNode) -> None"""
    @property
    def input_info_(self) -> dict[str, NodeInputInfo]: ...
    @property
    def name_(self) -> str: ...
    @property
    def output_info_(self) -> dict[str, NodeOutputInfo]: ...
    @property
    def state_info_(self) -> dict[str, NodeStateInfo]: ...
    @property
    def tag_(self) -> str: ...

class GTensor:
    @overload
    def __init__(self) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: cudasim_pycore.GTensor) -> None

        2. __init__(self: cudasim_pycore.GTensor, shape: list[int], dtype: cudasim_pycore.NumericalDataType, device_type: cudasim_pycore.DeviceType) -> None

        3. __init__(self: cudasim_pycore.GTensor, scalar: cudasim_pycore.Scalar, device_type: cudasim_pycore.DeviceType) -> None
        """
    @overload
    def __init__(self, shape: list[int], dtype: NumericalDataType, device_type: DeviceType) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: cudasim_pycore.GTensor) -> None

        2. __init__(self: cudasim_pycore.GTensor, shape: list[int], dtype: cudasim_pycore.NumericalDataType, device_type: cudasim_pycore.DeviceType) -> None

        3. __init__(self: cudasim_pycore.GTensor, scalar: cudasim_pycore.Scalar, device_type: cudasim_pycore.DeviceType) -> None
        """
    @overload
    def __init__(self, scalar: Scalar, device_type: DeviceType) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: cudasim_pycore.GTensor) -> None

        2. __init__(self: cudasim_pycore.GTensor, shape: list[int], dtype: cudasim_pycore.NumericalDataType, device_type: cudasim_pycore.DeviceType) -> None

        3. __init__(self: cudasim_pycore.GTensor, scalar: cudasim_pycore.Scalar, device_type: cudasim_pycore.DeviceType) -> None
        """
    def clone(self) -> GTensor:
        """clone(self: cudasim_pycore.GTensor) -> cudasim_pycore.GTensor"""
    def copyFrom(self, other: GTensor) -> None:
        """copyFrom(self: cudasim_pycore.GTensor, other: cudasim_pycore.GTensor) -> None"""
    def copyTo(self, other: GTensor) -> None:
        """copyTo(self: cudasim_pycore.GTensor, other: cudasim_pycore.GTensor) -> None"""
    def device(self) -> DeviceType:
        """device(self: cudasim_pycore.GTensor) -> cudasim_pycore.DeviceType"""
    def dim(self) -> int:
        """dim(self: cudasim_pycore.GTensor) -> int"""
    def dtype(self) -> NumericalDataType:
        """dtype(self: cudasim_pycore.GTensor) -> cudasim_pycore.NumericalDataType"""
    def elemCount(self) -> int:
        """elemCount(self: cudasim_pycore.GTensor) -> int"""
    def elemSize(self) -> int:
        """elemSize(self: cudasim_pycore.GTensor) -> int"""
    def expand(self, new_shape: list[int]) -> GTensor:
        """expand(self: cudasim_pycore.GTensor, new_shape: list[int]) -> cudasim_pycore.GTensor"""
    def fill(self, value: Scalar) -> None:
        """fill(self: cudasim_pycore.GTensor, value: cudasim_pycore.Scalar) -> None"""
    def getTorchTensor(self) -> torch.Tensor:
        """getTorchTensor(self: cudasim_pycore.GTensor) -> torch.Tensor"""
    def isContiguous(self) -> bool:
        """isContiguous(self: cudasim_pycore.GTensor) -> bool"""
    def numel(self) -> int:
        """numel(self: cudasim_pycore.GTensor) -> int"""
    def print(self) -> None:
        """print(self: cudasim_pycore.GTensor) -> None"""
    @staticmethod
    def rands(shape: list[int], dtype: NumericalDataType = ..., device_type: DeviceType = ...) -> GTensor:
        """rands(shape: list[int], dtype: cudasim_pycore.NumericalDataType = <NumericalDataType.kFloat32: 0>, device_type: cudasim_pycore.DeviceType = <DeviceType.kCUDA: 0>) -> cudasim_pycore.GTensor"""
    def randsLike(self) -> GTensor:
        """randsLike(self: cudasim_pycore.GTensor) -> cudasim_pycore.GTensor"""
    def reshape(self, shape: list[int]) -> GTensor:
        """reshape(self: cudasim_pycore.GTensor, shape: list[int]) -> cudasim_pycore.GTensor"""
    @staticmethod
    def setSeed(seed: int) -> None:
        """setSeed(seed: int) -> None"""
    @staticmethod
    def setTensorDefaultDeviceId(device_id: int) -> None:
        """setTensorDefaultDeviceId(device_id: int) -> None"""
    def shape(self) -> list[int]:
        """shape(self: cudasim_pycore.GTensor) -> list[int]"""
    def squeeze(self, dim: int) -> GTensor:
        """squeeze(self: cudasim_pycore.GTensor, dim: int) -> cudasim_pycore.GTensor"""
    def unsqueeze(self, dim: int) -> GTensor:
        """unsqueeze(self: cudasim_pycore.GTensor, dim: int) -> cudasim_pycore.GTensor"""
    def zero(self) -> None:
        """zero(self: cudasim_pycore.GTensor) -> None"""
    @staticmethod
    def zeros(shape: list[int], dtype: NumericalDataType = ..., device_type: DeviceType = ...) -> GTensor:
        """zeros(shape: list[int], dtype: cudasim_pycore.NumericalDataType = <NumericalDataType.kFloat32: 0>, device_type: cudasim_pycore.DeviceType = <DeviceType.kCUDA: 0>) -> cudasim_pycore.GTensor"""
    def zerosLike(self) -> GTensor:
        """zerosLike(self: cudasim_pycore.GTensor) -> cudasim_pycore.GTensor"""

class MessageBus:
    def __init__(self) -> None:
        """__init__(self: cudasim_pycore.MessageBus) -> None"""
    def addTrigger(self, trigger_tag: str) -> None:
        """addTrigger(self: cudasim_pycore.MessageBus, trigger_tag: str) -> None"""
    def buildGraph(self) -> None:
        """buildGraph(self: cudasim_pycore.MessageBus) -> None"""
    def clearAll(self) -> None:
        """clearAll(self: cudasim_pycore.MessageBus) -> None"""
    def getCurrentExecuteOrder(self) -> int:
        """getCurrentExecuteOrder(self: cudasim_pycore.MessageBus) -> int"""
    def getMessageQueue(self, node_name: str, message_name: str) -> MessageQueue:
        """getMessageQueue(self: cudasim_pycore.MessageBus, node_name: str, message_name: str) -> cudasim_pycore.MessageQueue"""
    def getMessageShape(self, message_name: str) -> list[int]:
        """getMessageShape(self: cudasim_pycore.MessageBus, message_name: str) -> list[int]"""
    def getNodeOrder(self, node_id: int) -> int:
        """getNodeOrder(self: cudasim_pycore.MessageBus, node_id: int) -> int"""
    def registerComponent(self, node: ExecuteNode) -> None:
        """registerComponent(self: cudasim_pycore.MessageBus, node: cudasim_pycore.ExecuteNode) -> None"""
    def registerInput(self, component: ExecuteNode, info: NodeInputInfo) -> None:
        """registerInput(self: cudasim_pycore.MessageBus, component: cudasim_pycore.ExecuteNode, info: cudasim_pycore.NodeInputInfo) -> None"""
    def registerOutput(self, component: ExecuteNode, info: NodeOutputInfo) -> None:
        """registerOutput(self: cudasim_pycore.MessageBus, component: cudasim_pycore.ExecuteNode, info: cudasim_pycore.NodeOutputInfo) -> None"""
    def registerState(self, component: ExecuteNode, info: NodeStateInfo) -> None:
        """registerState(self: cudasim_pycore.MessageBus, component: cudasim_pycore.ExecuteNode, info: cudasim_pycore.NodeStateInfo) -> None"""
    def resetExecuteOrder(self) -> None:
        """resetExecuteOrder(self: cudasim_pycore.MessageBus) -> None"""
    def trigger(self, trigger_tag: str) -> None:
        """trigger(self: cudasim_pycore.MessageBus, trigger_tag: str) -> None"""

class MessageQueue:
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def getHistoryGTensor(self, offset: int) -> GTensor:
        """getHistoryGTensor(self: cudasim_pycore.MessageQueue, offset: int) -> cudasim_pycore.GTensor"""
    def getWriteTensorRef(self) -> GTensor:
        """getWriteTensorRef(self: cudasim_pycore.MessageQueue) -> cudasim_pycore.GTensor"""
    def reset(self) -> None:
        """reset(self: cudasim_pycore.MessageQueue) -> None"""
    def resetEnvData(self, env_group_id: int, env_id: int) -> None:
        """resetEnvData(self: cudasim_pycore.MessageQueue, env_group_id: int, env_id: int) -> None"""

class NodeInputInfo:
    history_offset: int
    reduce_method: ReduceMethod
    def __init__(self, name: str, shape: list[int], history_offset: int, reduce_method: ReduceMethod) -> None:
        """__init__(self: cudasim_pycore.NodeInputInfo, name: str, shape: list[int], history_offset: int, reduce_method: cudasim_pycore.ReduceMethod) -> None"""
    @property
    def name(self) -> str: ...
    @property
    def shape(self) -> list[int]: ...

class NodeOutputInfo:
    dtype: NumericalDataType
    history_padding_val: GTensor | None
    def __init__(self, name: str, shape: list[int], dtype: NumericalDataType, history_padding_val: GTensor | None) -> None:
        """__init__(self: cudasim_pycore.NodeOutputInfo, name: str, shape: list[int], dtype: cudasim_pycore.NumericalDataType, history_padding_val: Optional[cudasim_pycore.GTensor]) -> None"""
    @property
    def name(self) -> str: ...
    @property
    def shape(self) -> list[int]: ...

class NodeStateInfo:
    dtype: NumericalDataType
    init_val: GTensor | None
    def __init__(self, name: str, shape: list[int], dtype: NumericalDataType, init_val: GTensor | None) -> None:
        """__init__(self: cudasim_pycore.NodeStateInfo, name: str, shape: list[int], dtype: cudasim_pycore.NumericalDataType, init_val: Optional[cudasim_pycore.GTensor]) -> None"""
    @property
    def name(self) -> str: ...
    @property
    def shape(self) -> list[int]: ...

class NumericalDataType:
    __members__: ClassVar[dict] = ...  # read-only
    __entries: ClassVar[dict] = ...
    kFloat32: ClassVar[NumericalDataType] = ...
    kFloat64: ClassVar[NumericalDataType] = ...
    kInt32: ClassVar[NumericalDataType] = ...
    kInt64: ClassVar[NumericalDataType] = ...
    kUInt32: ClassVar[NumericalDataType] = ...
    kUInt8: ClassVar[NumericalDataType] = ...
    def __init__(self, value: int) -> None:
        """__init__(self: cudasim_pycore.NumericalDataType, value: int) -> None"""
    def __eq__(self, other: object) -> bool:
        """__eq__(self: object, other: object) -> bool"""
    def __hash__(self) -> int:
        """__hash__(self: object) -> int"""
    def __index__(self) -> int:
        """__index__(self: cudasim_pycore.NumericalDataType) -> int"""
    def __int__(self) -> int:
        """__int__(self: cudasim_pycore.NumericalDataType) -> int"""
    def __ne__(self, other: object) -> bool:
        """__ne__(self: object, other: object) -> bool"""
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class ReduceMethod:
    __members__: ClassVar[dict] = ...  # read-only
    MAX: ClassVar[ReduceMethod] = ...
    MIN: ClassVar[ReduceMethod] = ...
    STACK: ClassVar[ReduceMethod] = ...
    SUM: ClassVar[ReduceMethod] = ...
    __entries: ClassVar[dict] = ...
    def __init__(self, value: int) -> None:
        """__init__(self: cudasim_pycore.ReduceMethod, value: int) -> None"""
    def __eq__(self, other: object) -> bool:
        """__eq__(self: object, other: object) -> bool"""
    def __hash__(self) -> int:
        """__hash__(self: object) -> int"""
    def __index__(self) -> int:
        """__index__(self: cudasim_pycore.ReduceMethod) -> int"""
    def __int__(self) -> int:
        """__int__(self: cudasim_pycore.ReduceMethod) -> int"""
    def __ne__(self, other: object) -> bool:
        """__ne__(self: object, other: object) -> bool"""
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class Scalar:
    @overload
    def __init__(self, arg0: float) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: cudasim_pycore.Scalar, arg0: float) -> None

        2. __init__(self: cudasim_pycore.Scalar, arg0: float) -> None

        3. __init__(self: cudasim_pycore.Scalar, arg0: int) -> None

        4. __init__(self: cudasim_pycore.Scalar, arg0: int) -> None

        5. __init__(self: cudasim_pycore.Scalar, arg0: int) -> None

        6. __init__(self: cudasim_pycore.Scalar, arg0: int) -> None
        """
    @overload
    def __init__(self, arg0: float) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: cudasim_pycore.Scalar, arg0: float) -> None

        2. __init__(self: cudasim_pycore.Scalar, arg0: float) -> None

        3. __init__(self: cudasim_pycore.Scalar, arg0: int) -> None

        4. __init__(self: cudasim_pycore.Scalar, arg0: int) -> None

        5. __init__(self: cudasim_pycore.Scalar, arg0: int) -> None

        6. __init__(self: cudasim_pycore.Scalar, arg0: int) -> None
        """
    @overload
    def __init__(self, arg0: int) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: cudasim_pycore.Scalar, arg0: float) -> None

        2. __init__(self: cudasim_pycore.Scalar, arg0: float) -> None

        3. __init__(self: cudasim_pycore.Scalar, arg0: int) -> None

        4. __init__(self: cudasim_pycore.Scalar, arg0: int) -> None

        5. __init__(self: cudasim_pycore.Scalar, arg0: int) -> None

        6. __init__(self: cudasim_pycore.Scalar, arg0: int) -> None
        """
    @overload
    def __init__(self, arg0: int) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: cudasim_pycore.Scalar, arg0: float) -> None

        2. __init__(self: cudasim_pycore.Scalar, arg0: float) -> None

        3. __init__(self: cudasim_pycore.Scalar, arg0: int) -> None

        4. __init__(self: cudasim_pycore.Scalar, arg0: int) -> None

        5. __init__(self: cudasim_pycore.Scalar, arg0: int) -> None

        6. __init__(self: cudasim_pycore.Scalar, arg0: int) -> None
        """
    @overload
    def __init__(self, arg0: int) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: cudasim_pycore.Scalar, arg0: float) -> None

        2. __init__(self: cudasim_pycore.Scalar, arg0: float) -> None

        3. __init__(self: cudasim_pycore.Scalar, arg0: int) -> None

        4. __init__(self: cudasim_pycore.Scalar, arg0: int) -> None

        5. __init__(self: cudasim_pycore.Scalar, arg0: int) -> None

        6. __init__(self: cudasim_pycore.Scalar, arg0: int) -> None
        """
    @overload
    def __init__(self, arg0: int) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: cudasim_pycore.Scalar, arg0: float) -> None

        2. __init__(self: cudasim_pycore.Scalar, arg0: float) -> None

        3. __init__(self: cudasim_pycore.Scalar, arg0: int) -> None

        4. __init__(self: cudasim_pycore.Scalar, arg0: int) -> None

        5. __init__(self: cudasim_pycore.Scalar, arg0: int) -> None

        6. __init__(self: cudasim_pycore.Scalar, arg0: int) -> None
        """
    def toDouble(self) -> float:
        """toDouble(self: cudasim_pycore.Scalar) -> float"""
    def toFloat(self) -> float:
        """toFloat(self: cudasim_pycore.Scalar) -> float"""
    def toInt32(self) -> int:
        """toInt32(self: cudasim_pycore.Scalar) -> int"""
    def toInt64(self) -> int:
        """toInt64(self: cudasim_pycore.Scalar) -> int"""
    def toUInt32(self) -> int:
        """toUInt32(self: cudasim_pycore.Scalar) -> int"""
    def toUInt8(self) -> int:
        """toUInt8(self: cudasim_pycore.Scalar) -> int"""

class TensorItemHandle:
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def activeGroupAt(self, group_id: int) -> GTensor:
        """activeGroupAt(self: cudasim_pycore.TensorItemHandle, group_id: int) -> cudasim_pycore.GTensor"""
    def groupAt(self, group_id: int) -> GTensor:
        """groupAt(self: cudasim_pycore.TensorItemHandle, group_id: int) -> cudasim_pycore.GTensor"""

class TensorRegistry:
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def clear(self) -> None:
        """clear(self: cudasim_pycore.TensorRegistry) -> None"""
    def createTensor(self, uri: str, shape: list[int], dtype: NumericalDataType, device: DeviceType) -> GTensor:
        """createTensor(self: cudasim_pycore.TensorRegistry, uri: str, shape: list[int], dtype: cudasim_pycore.NumericalDataType, device: cudasim_pycore.DeviceType) -> cudasim_pycore.GTensor"""
    def getAllTensorUri(self) -> list[str]:
        """getAllTensorUri(self: cudasim_pycore.TensorRegistry) -> list[str]"""
    @staticmethod
    def getInstance() -> TensorRegistry:
        """getInstance() -> cudasim_pycore.TensorRegistry"""
    def getTensor(self, uri: str) -> GTensor:
        """getTensor(self: cudasim_pycore.TensorRegistry, uri: str) -> cudasim_pycore.GTensor"""
    def getTensorsByPrefix(self, prefix: str) -> list[GTensor]:
        """getTensorsByPrefix(self: cudasim_pycore.TensorRegistry, prefix: str) -> list[cudasim_pycore.GTensor]"""
    def removeTensor(self, uri: str) -> None:
        """removeTensor(self: cudasim_pycore.TensorRegistry, uri: str) -> None"""
    def removeTensorsByPrefix(self, prefix: str) -> None:
        """removeTensorsByPrefix(self: cudasim_pycore.TensorRegistry, prefix: str) -> None"""
    def size(self) -> int:
        """size(self: cudasim_pycore.TensorRegistry) -> int"""

def getTensorRegistry() -> TensorRegistry:
    """getTensorRegistry() -> cudasim_pycore.TensorRegistry"""
