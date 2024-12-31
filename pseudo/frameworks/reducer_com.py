import torch as th
from common_types import *
from component import ComponentContext, Component
from environ_group import ContextBase

@dataclass
class ReducerContext(ComponentContext):
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
    ):
        super().__init__(id=component_id, graph_id="default", context_id="reduced_ctx", mb_accessor=msgbus)
        self.message_id = message_id
        self.new_message_id = new_message_id
        self.tensor_shape = shape
        self.reduce_method = reduce_method

    def get_output_pub(self):
        return self.output_pub

    def get_input_sub(self):
        return self.input_sub

    def onRegister(self):
        # ! Reducer的接收者不需要进行缓存，由实际的subscriber处理
        self.input_sub = self.createSubscriber(
            message_id=self.message_id,
            shape=self.tensor_shape,
            reduce_method=ReduceMethod.STACK
        )

        self.output_pub = self.createPublisher(
            self.new_message_id,
            self.tensor_shape,
        )

    def onEnvGroupInit(self, ctx: ContextBase) -> Optional[ComponentContext]:
        return None

    def onEnvGroupReset(self, context: ContextBase, reset_flags: th.Tensor):
        pass

    def onExecuate(self, context: ComponentContext, input: Mapping[MessageID, Tensor]):
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

    def onReset(self, context: ComponentContext, reset_flag: Tensor):
        pass

    def onEnabledChanged(self, enabled: bool):
        pass