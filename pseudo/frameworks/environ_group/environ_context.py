from common_types import *
from typing import Any, Type
from environ_config import EnvironGroupConfig
from environ_group_interface import EGManagedObject
from threading import Lock

@dataclass
class ContextDescriptor:
    """上下文描述符"""
    context_type: ContextType
    context_cls : Type[Any]
    provider: "EGManagedObject"
    dependencies: List[ContextType] = field(default_factory=list)  # 依赖的其他上下文类型


class GlobalContextRegistry:
    """全局上下文注册表"""
    _instance = None

    def __init__(self):
        self.descriptors: Dict[str, ContextDescriptor] = {}
        self._lock = Lock()

    @classmethod
    def get_instance(cls) -> 'GlobalContextRegistry':
        if cls._instance is None:
            cls._instance = GlobalContextRegistry()
        return cls._instance

    def register_provider(self, context_id: str, descriptor: ContextDescriptor):
        """注册上下文提供者"""
        with self._lock:
            self.descriptors[context_id] = descriptor

    def get_descriptor(self, context_id: str) -> Optional[ContextDescriptor]:
        return self.descriptors.get(context_id)

    def get_all_descriptors(self) -> Dict[str, ContextDescriptor]:
        return self.descriptors.copy()


class EnvGroupContextManager:
    """环境组上下文管理器"""
    def __init__(self, env_group_id: str, env_group_cfg: EnvironGroupConfig):
        self.env_group_id = env_group_id
        self.env_group_cfg = env_group_cfg
        self.contexts: Dict[str, Any] = {}
        self.registry = GlobalContextRegistry.get_instance()

    def init_contexts(self):
        """初始化所有上下文"""
        initialized = set()

        # 递归调用函数，初始化依赖的上下文
        def init_context(context_id: str):
            if context_id in initialized:
                return

            descriptor = self.registry.get_descriptor(context_id)
            if not descriptor:
                raise KeyError(f"Context provider {context_id} not registered")

            # 先初始化依赖的上下文
            for dep_type in descriptor.dependencies:
                dep_id = self._find_context_id_by_type(dep_type)
                if dep_id:
                    init_context(dep_id)

            # 收集依赖的上下文
            deps:Dict[ContextType, ContextBase] = {}
            for dep_type in descriptor.dependencies:
                dep_ctx = self._find_context_by_type(dep_type)
                if not dep_ctx:
                    raise RuntimeError(f"Required dependency {dep_type} not found")
                deps[dep_type] = dep_ctx

            # 调用提供者的Init方法创建上下文
            context = descriptor.provider.onEnvGroupInit(
                ContextBase(dependence=deps, env_config=self.env_group_cfg, ctx_manager=self))
            self.contexts[context_id] = context
            initialized.add(context_id)

        # 初始化所有已注册的上下文
        for context_id in self.registry.get_all_descriptors():
            init_context(context_id)

    def reset_contexts(self, reset_flags: Tensor):
        """重置所有上下文"""
        for context_id, descriptor in self.registry.descriptors.items():
            context = self.contexts.get(context_id)
            if context:
                descriptor.provider.onEnvGroupReset(context, reset_flags)

    def get_context(self, context_id: str) -> Optional["ContextBase"]:
        return self.contexts.get(context_id)

    def _find_context_id_by_type(self, context_type: ContextType) -> Optional[str]:
        for ctx_id, descriptor in self.registry.descriptors.items():
            if descriptor.context_type == context_type:
                return ctx_id
        return None

    def _find_context_by_type(self, context_type: ContextType) -> Optional[Any]:
        ctx_id = self._find_context_id_by_type(context_type)
        return self.contexts.get(ctx_id) if ctx_id else None

@dataclass
class ContextBase(ABC):
    dependence  : Dict[ContextType, Any]
    env_config  : EnvironGroupConfig
    ctx_manager : EnvGroupContextManager

    def get_dependence_ctx(self, ctx_type):
        return self.dependence[ctx_type]

    def get_base_dict(self):
        kw = ['dependence', 'env_config', 'ctx_manager']
        return {k:self.__dict__[k] for k in kw}

    def get_other_ctx(self, ctx_id:ContextID):
        self.ctx_manager.get_context(ctx_id)