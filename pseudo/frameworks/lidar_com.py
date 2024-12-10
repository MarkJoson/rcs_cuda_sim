from common_types import *
from components import ComponentCtx, Component
from message_bus_base import MessageBusBase
from environ_group import IEnvironContext

class LidarContext(ComponentCtx):
    pass

class LidarCom(Component):
    def __init__(self, msgbus:MessageBusBase):
        super().__init__(ComponentID("lidar"), GraphID("dafault"), msgbus)

    def onRegister(self):
        self.pose_sub = self.createSubscriber(MessageID("pose"), TensorShape((3,)), ReduceMethod.REPLACE)


    def onInit(self, eg_ctx: IEnvironContext) -> ComponentCtx:
        return LidarContext.build(eg_ctx=eg_ctx)