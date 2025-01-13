from enum import Enum
from typing import List, Dict, Optional, Callable, Any, Sequence, Mapping, Union
import torch as th

class InvalidPathError(Exception):
    pass

class ComponentPath:
    """表示组件在组件树中的位置"""
    def __init__(self, path_string: str = ""):
        self.segments = path_string.split('/') if path_string else []

    def add_segment(self, segment: str) -> 'ComponentPath':
        """添加路径段"""
        self.segments.append(segment)
        return self

    def to_string(self) -> str:
        return '/'.join(self.segments)

    def __eq__(self, other: 'ComponentPath') -> bool:
        return self.segments == other.segments

    def __hash__(self) -> int:
        return hash(self.to_string())

class NodeCategory(Enum):
    """执行节点类别"""
    COMPUTATION = 1    # 计算节点
    FLOW_CONTROL = 2   # 流程控制节点
    DATA_TRANSFER = 3  # 数据传输节点

class ComputationNode:
    """计算图节点"""
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.category = NodeCategory.COMPUTATION
        self.input_nodes: List[ComputationNode] = []
        self.output_tensors: List[str] = []

    def require_input_from(self, node: 'ComputationNode'):
        """添加输入依赖"""
        if node not in self.input_nodes:
            self.input_nodes.append(node)

    def declare_output(self, tensor_id: str):
        """声明输出张量"""
        self.output_tensors.append(tensor_id)

    def compute(self, env_group: 'EnvironGroup'):
        """执行计算"""
        raise NotImplementedError

class SimComponent(ComputationNode):
    """仿真组件基类"""
    def __init__(self, component_id: str):
        super().__init__(component_id)
        self.sub_components: List[SimComponent] = []
        self.parent_component: Optional[SimComponent] = None

    def add_sub_component(self, component: 'SimComponent'):
        """添加子组件"""
        component.parent_component = self
        self.sub_components.append(component)

    def get_component_path(self) -> ComponentPath:
        """获取组件完整路径"""
        path = ComponentPath()
        current = self
        while current is not None:
            path.segments.insert(0, current.node_id)
            current = current.parent_component
        return path

    def traverse(self, visitor_func: Callable, skip_recursion: bool = False,
                leaf_only: bool = False, *args, **kwargs):
        """遍历组件树"""
        if not leaf_only:
            visitor_func(self, *args, **kwargs)

        if not skip_recursion:
            for sub_comp in self.sub_components:
                sub_comp.traverse(visitor_func, skip_recursion, leaf_only, *args, **kwargs)

    def initialize(self, env_group: 'EnvironGroup'):
        """初始化组件"""
        pass

    def reset(self, env_group: 'EnvironGroup', reset_mask: th.Tensor):
        """重置组件状态"""
        pass

class TensorRegistry:
    """张量注册表"""
    def __init__(self):
        self._tensor_storage: Dict[str, th.Tensor] = {}
        self._tensor_metadata: Dict[str, Dict] = {}

    def register(self, tensor_id: str, tensor: th.Tensor,
                metadata: Optional[Dict] = None):
        """注册张量"""
        self._tensor_storage[tensor_id] = tensor
        if metadata:
            self._tensor_metadata[tensor_id] = metadata

    def get_tensor(self, tensor_id: str) -> th.Tensor:
        """获取张量"""
        if tensor_id not in self._tensor_storage:
            raise KeyError(f"Unknown tensor: {tensor_id}")
        return self._tensor_storage[tensor_id]

    def get_metadata(self, tensor_id: str) -> Dict:
        """获取张量元数据"""
        return self._tensor_metadata.get(tensor_id, {})

class EnvironGroup:
    """环境组"""
    def __init__(self, group_id: int, env_count: int):
        self.group_id = group_id
        self.env_count = env_count
        self.tensor_registry = TensorRegistry()

        # 属性存储
        self.group_properties: Dict = {}    # 环境组级别属性
        self.env_properties: Dict = {}      # 单个环境级别属性
        self.component_states: Dict = {}    # 组件状态存储

    def get_state(self, state_id: str) -> th.Tensor:
        """获取组件状态"""
        return self.tensor_registry.get_tensor(state_id)

    def get_tensor(self, tensor_id: str) -> th.Tensor:
        """获取任意张量"""
        return self.tensor_registry.get_tensor(tensor_id)

class SimulationManager:
    """仿真管理器"""
    def __init__(self):
        self._env_groups: Dict[int, EnvironGroup] = {}
        self._component_configs: Dict[str, Dict] = {}

    def register_group_property(self, component: SimComponent,
                              property_name: str,
                              creator: Callable[['EnvironGroup', int], th.Tensor]) -> None:
        """注册环境组属性"""
        pass

    def register_env_property(self, component: SimComponent,
                            property_name: str,
                            creator: Callable[['EnvironGroup', int], th.Tensor]) -> None:
        """注册环境属性"""
        pass

    def register_component_state(self, component: SimComponent,
                               state_name: str,
                               creator: Callable[['EnvironGroup', int], th.Tensor],
                               public: bool = True) -> None:
        """注册组件状态"""
        pass

class ComputationGraph:
    """计算图"""
    def __init__(self):
        self.all_nodes: List[ComputationNode] = []
        self.execution_order: List[ComputationNode] = []

    def add_computation_node(self, node: ComputationNode):
        """添加计算节点"""
        self.all_nodes.append(node)

    def compile(self):
        """编译计算图"""
        visited = set()
        temp_mark = set()
        ordered_nodes = []

        def topological_sort(node: ComputationNode):
            if node in temp_mark:
                raise ValueError("Circular dependency detected")
            if node in visited:
                return

            temp_mark.add(node)
            for dep_node in node.input_nodes:
                topological_sort(dep_node)
            temp_mark.remove(node)
            visited.add(node)
            ordered_nodes.append(node)

        for node in self.all_nodes:
            if node not in visited:
                topological_sort(node)

        self.execution_order = ordered_nodes