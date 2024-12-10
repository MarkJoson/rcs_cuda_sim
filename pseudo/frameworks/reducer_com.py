from common_types import *
from components import ComponentCtx, Component
from message_bus_base import MessageBusBase
from environ_group import IEnvironContext

@dataclass
class ReducerContext(ComponentCtx):
    reduce_method: ReduceMethod

class ReducerCom(Component):
    def __init__(self,
                 component_id: ComponentID,
                 msg_id: MessageID,
                 shape: TensorShape,
                 reduce_method: ReduceMethod,
                 msgbus: MessageBusBase):
        super().__init__(component_id, GraphID("reduce"), msgbus)
        self.msg_id = msg_id
        self.tensor_shape = shape
        self.reduce_method = reduce_method

    def onRegister(self):
        self.input_sub = self.createSubscriber(
            self.msg_id,
            self.tensor_shape,
            ReduceMethod.STACK
        )
        self.output_pub = self.createPublisher(
            self.msg_id,
            self.tensor_shape,
            1,
            None  # TODO: 适当的padding值
        )

    def onInit(self, env_group_cfg: IEnvironContext) -> ComponentCtx:
        return ReducerContext.build(eg_ctx=env_group_cfg)

    def onExecuate(self, context: ComponentCtx, input: Mapping[MessageID, Tensor]):
        tensor = input[self.msg_id]
        # 根据reduce_method进行相应的张量运算
        if self.reduce_method == ReduceMethod.SUM:
            raise NotImplementedError
            # result = th.sum(tensor, dim=1)
        elif self.reduce_method == ReduceMethod.AVERAGE:
            raise NotImplementedError
            # result = th.mean(tensor, dim=1)
        elif self.reduce_method == ReduceMethod.MAX:
            raise NotImplementedError
            # result = th.max(tensor, dim=1)[0]
        elif self.reduce_method == ReduceMethod.MIN:
            raise NotImplementedError
            # result = th.min(tensor, dim=1)[0]
        elif self.reduce_method == ReduceMethod.REPLACE:
            raise NotImplementedError
            # result = tensor[-1]  # 使用最后一个输入

        self.output_pub.publish(context, result)

    def onReset(self, context: ComponentCtx, reset_flag: Tensor):
        pass

    def onEnabledChanged(self, enabled: bool):
        pass