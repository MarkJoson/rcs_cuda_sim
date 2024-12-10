from common_types import *

class IPublish(ABC):
    @abstractmethod
    def publish(
        self,
        environ_ctx,
        publish_node: ComponentID,      # 发布者ID
        tensor_id: MessageID,         # 发布消息ID
        tensor: Tensor,

    ):
        raise NotImplementedError

@dataclass
class MessageHandler:
    component_id        : ComponentID                                   # 所属对象
    message_id          : MessageID                                     # 消息名
    shape               : TensorShape                                   # 形状
    is_enabled          : bool = field(default=True)                    # 是否使能，如果没有发布者，订阅者将被禁用

    def setEnabled(self, enabled:bool):
        self.is_enabled = enabled

@dataclass
class Subscriber(MessageHandler):
    publishers          : Sequence['Publisher'] = list()
    history_offset      : int                   = 0                     # 时间点，0代表立即tensor，即需要发布者的最新数据
    reduce_method       : ReduceMethod          = ReduceMethod.STACK    # 多个publisher时的处理方法
    stack_dim           : int = 1                                       # 堆叠输入维度，当reduce_method==STACK时，该项显示了堆叠的维度
    stack_order         : Sequence[ComponentID] = list()                # 堆叠输入的顺序


@dataclass
class Publisher(MessageHandler):
    subscribers         : Sequence['Subscriber']= list()                # 订阅者
    history_padding_val : Tensor = field(default_factory=Tensor)        # 无效历史的padding值
    publish_if          : Optional[IPublish] = None                     # 发布者接口

    def publish(self, environ_ctx, tensor:Tensor):
        # TODO. check shape
        assert self.publish_if

        self.publish_if.publish(
            environ_ctx=environ_ctx,
            publish_node=self.component_id,
            tensor_id=self.message_id,
            tensor=tensor)