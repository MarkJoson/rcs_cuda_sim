from common_types import *
from mesage_handler import Publisher, Subscriber
from environ_group import EGManagedObject, ContextBase
from common_types import Tensor

if TYPE_CHECKING:
    from message_bus import MessageBus

@dataclass
class ComponentContext(ContextBase):
    pass
    # eg_ctx_if: IEnvironContext

    # @staticmethod
    # def build(eg_ctx: IEnvironContext):
    #     return ComponentCtx(eg_ctx_if=eg_ctx)


@dataclass
class Component(EGManagedObject):
    id          : ComponentID
    graph_id    : GraphID
    context_id  : ContextID
    msgbus      : "MessageBus"
    pubs        : List[Publisher]   = list()
    subs        : List[Subscriber]  = list()
    enabled     : bool = True

    @abstractmethod
    def onRegister(self):
        ''' 当该组件被挂载时调用，在该方法中定义组件的输入输出：注册pub, sub '''
        raise NotImplementedError

    @abstractmethod
    def onExecuate(self, context:Optional[ContextBase], input: Mapping[MessageID, Tensor]):
        ''' 执行期调用 '''
        raise NotImplementedError

    @abstractmethod
    def onReset(self, context:ComponentContext, reset_flag:Tensor):
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

    def createSubscriber(self,
                         message_id:MessageID,
                         shape:MessageDataShape,
                         history_offset:int=0,
                         history_padding_val: Optional[Tensor] = None,
                         reduce_method:ReduceMethod = ReduceMethod.STACK):
        sub = self.msgbus.createSubscriber(
            component_id=self.id,
            message_id=message_id,
            shape=shape,
            history_offset=history_offset,
            history_padding_val=history_padding_val,
            reduce_method=reduce_method,
        )
        self.subs.append(sub)
        return sub

    def createPublisher(self, message_id:MessageID, shape:MessageDataShape):
        pub = self.msgbus.createPublisher(
            component_id=self.id,
            message_id=message_id,
            shape=shape,
        )
        self.pubs.append(pub)
        return pub