from common_types import *
from typing import Generic, TypeVar
from environ_config import EnvironGroupConfig

if TYPE_CHECKING:
    from environ_context import ContextBase

@dataclass
class EGManagedObject(ABC):
    """环境组接口类"""

    @abstractmethod
    def onEnvGroupInit(self, basic_context: "ContextBase") -> Optional[ContextBase]:
        """
        初始化新的环境组
        Args:
            env_group_cfg: 环境组配置
        Returns:
            环境组相关的上下文数据
        """
        raise NotImplementedError

    @abstractmethod
    def onEnvGroupReset(self, context: ContextBase, reset_flags: Tensor):
        """
        重置环境组
        Args:
            context: 环境组上下文
            reset_flags: 重置标志张量，指示哪些环境需要重置
        """
        raise NotImplementedError