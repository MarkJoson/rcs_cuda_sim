from common_types import *
from message_bus_base import MessageBusBase
from mesage_handler import Publisher, Subscriber
from environ_group import IEnvironContext

@dataclass
class ComponentCtx(ABC):
    eg_ctx_if: IEnvironContext

    @staticmethod
    def build(eg_ctx: IEnvironContext):
        return ComponentCtx(eg_ctx_if=eg_ctx)


@dataclass
class Component(ABC):
    id        : ComponentID
    graph_id  : GraphID
    mb_accessor : MessageBusBase
    pubs        : List[Publisher]   = list()
    subs        : List[Subscriber]  = list()
    enabled     : bool = True

    @abstractmethod
    def onRegister(self):
        ''' 当该组件被挂载时调用，在该方法中定义组件的输入输出：注册pub, sub '''
        raise NotImplementedError

    @abstractmethod
    def onInit(self, env_group_cfg:IEnvironContext) -> ComponentCtx:
        ''' 当初始化环境组时调用，在该方法中定义组件的状态：返回组件状态上下文 '''
        raise NotImplementedError

    @abstractmethod
    def onExecuate(self, context:ComponentCtx, input: Mapping[MessageID, Tensor]):
        ''' 执行期调用 '''
        raise NotImplementedError

    @abstractmethod
    def onReset(self, context:ComponentCtx, reset_flag:Tensor):
        ''' 定义某个环境在reset时组件的动作 '''
        raise NotImplementedError

    @abstractmethod
    def onEnabledChanged(self, enabled:bool):
        ''' 当不存在对应消息发布者时，组件被禁用的回调函数 '''
        raise NotImplementedError

    def setEnabled(self, enabled:bool):
        if self.enabled == enabled:
            return

        self.enabled = enabled

        # 设置所有发布者和接收者
        for pub in self.pubs:
            pub.setEnabled(enabled=enabled)

        for sub in self.subs:
            sub.setEnabled(enabled=enabled)

        self.onEnabledChanged(enabled=enabled)

    def createSubscriber(self, message_id:MessageID, shape:MessageDataShape, reduce_method:ReduceMethod, history_offset:int=0):
        sub = self.mb_accessor.createSubscriber(
            component_id=self.id,
            message_id=message_id,
            shape=shape,
            reduce_method=reduce_method,
            history_offset=history_offset,
        )
        self.subs.append(sub)
        return sub

    def createPublisher(self, message_id:MessageID, shape:MessageDataShape, history_padding_val:Tensor):
        pub = self.mb_accessor.createPublisher(
            component_id=self.id,
            message_id=message_id,
            shape=shape,
            history_padding_val=history_padding_val
        )
        self.pubs.append(pub)
        return pub