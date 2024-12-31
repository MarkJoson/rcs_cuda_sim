from common_types import *

if TYPE_CHECKING:
    from message_bus import MessageBus


@dataclass
class MessageHandler:
    component_id        : ComponentID                                   # 所属对象
    message_id          : MessageID                                     # 消息名
    shape               : MessageDataShape                                   # 形状
    is_enabled          : bool = field(default=True)                    # 是否使能，如果没有发布者，订阅者将被禁用

    def setEnabled(self, enabled:bool):
        self.is_enabled = enabled

@dataclass
class Publisher(MessageHandler):
    subscribers         : Sequence['Subscriber']= list()                # 订阅者
    msg_bus             : Optional['MessageBus'] = None                     # 发布者接口

    def publish(self, environ_ctx, tensor:Tensor):
        # TODO. check shape
        assert isinstance(self.msg_bus, 'MessageBus')

        self.msg_bus.publish(
            environ_ctx=environ_ctx,
            publish_node=self.component_id,
            tensor_id=self.message_id,
            tensor=tensor)

@dataclass
class Subscriber(MessageHandler):
    history_offset          : int  = 0                                      # 时间点，0代表立即tensor，即需要发布者的最新数据
    accept_invalid_history  : bool = False                                  # 是否接收无效的历史输入
    history_padding_val     : Optional[Tensor] = None                       # 无效历史的padding值

    reduce_method           : ReduceMethod          = ReduceMethod.STACK    # 多个publisher时的处理方法
    stack_dim               : int = 1                                       # 堆叠输入维度，当reduce_method==STACK时，该项显示了堆叠的维度
    stack_order             : Sequence[ComponentID] = list()                # 堆叠输入的顺序

    publishers              : Sequence['Publisher'] = list()
