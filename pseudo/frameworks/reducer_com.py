import torch as th
from common_types import *
from components import ComponentCtx, Component
from message_bus_base import MessageBusBase
from environ_group import IEnvironContext

@dataclass
class ReducerContext(ComponentCtx):
    reduce_method: ReduceMethod

class ReducerCom(Component):
    def __init__(
        self,
        component_id: ComponentID,
        message_id: MessageID,
        new_message_id: MessageID,
        shape: MessageDataShape,
        reduce_method: ReduceMethod,
        msgbus: MessageBusBase,
        history_padding_val: List[Tensor]
    ):
        super().__init__(component_id, GraphID("reduce"), msgbus)
        self.message_id = message_id
        self.new_message_id = new_message_id
        self.tensor_shape = shape
        self.reduce_method = reduce_method
        self.history_padding_val = th.stack(history_padding_val)

    def onRegister(self):
        self.input_sub = self.createSubscriber(
            self.message_id,
            self.tensor_shape,
            ReduceMethod.STACK
        )

        self.output_pub = self.createPublisher(
            self.new_message_id,
            self.tensor_shape,
            self.history_padding_val
        )

    def onInit(self, env_group_cfg: IEnvironContext) -> ComponentCtx:
        return ReducerContext.build(eg_ctx=env_group_cfg)

    def onExecuate(self, context: ComponentCtx, input: Mapping[MessageID, Tensor]):
        tensor = input[self.message_id]
        # 根据reduce_method进行相应的张量运算
        if self.reduce_method == ReduceMethod.SUM:
            result = th.sum(tensor, dim=1)
        elif self.reduce_method == ReduceMethod.AVERAGE:
            result = th.mean(tensor, dim=1)
        elif self.reduce_method == ReduceMethod.MAX:
            result = th.max(tensor, dim=1)[0]
        elif self.reduce_method == ReduceMethod.MIN:
            result = th.min(tensor, dim=1)[0]
        elif self.reduce_method == ReduceMethod.REPLACE:
            result = tensor[-1]  # 使用最后一个输入

        self.output_pub.publish(context, result)

    def onReset(self, context: ComponentCtx, reset_flag: Tensor):
        pass

    def onEnabledChanged(self, enabled: bool):
        pass