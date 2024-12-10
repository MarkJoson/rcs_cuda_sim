from common_types import *

if TYPE_CHECKING:
    from mesage_handler import Publisher, Subscriber

class MessageBusBase(ABC):
    @abstractmethod
    def registerComponent(
        self,
        component_id:ComponentID,
        exec_graph:GraphID
    ):
        raise NotImplementedError

    @abstractmethod
    def createPublisher(
        self,
        component_id: ComponentID,                  # 发布者所属节点ID
        message_id: MessageID,                      # 发布Tensor的消息Id
        shape: MessageDataShape,                    # 发布Tensor的形状
    ) -> "Publisher":
        raise NotImplementedError

    @abstractmethod
    def createSubscriber(
        self,
        component_id: ComponentID,                  # 接收者所属节点Id
        message_id: MessageID,                      # 接收Tensor的消息Id
        shape: MessageDataShape,                    # 接收Tensor的形状
        history_offset:int,                         # 接受Tensor的时间点
        history_padding_val: Optional[Tensor],                # 无效历史的padding值
        reduce_method: ReduceMethod,                # 多个相同消息Tensor时的合并方法
    ) -> "Subscriber":
        raise NotImplementedError