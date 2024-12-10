from common_types import *

if TYPE_CHECKING:
    from message_bus import MessageBusContext
    from components import ComponentCtx

@dataclass
class EnvironGroupConfig:
    num_envs:int


class IEnvironContext(ABC):
    @abstractmethod
    def getMsgBusCtx(self) -> "MessageBusContext":
        raise NotImplementedError

    @abstractmethod
    def getEnvironConfig(self) -> "EnvironGroupConfig":
        raise NotImplementedError


@dataclass
class EnvironGroupContext(IEnvironContext):
    mb_ctx: "MessageBusContext"
    com_ctx: Mapping[str, "ComponentCtx"]
    env_group_cfg: EnvironGroupConfig

    def getMsgBusCtx(self) -> "MessageBusContext":
        return self.mb_ctx

    def getEnvironConfig(self) -> EnvironGroupConfig:
        return self.env_group_cfg


