from common_types import *
from components import ComponentContext, Component
from message_bus_base import MessageBusBase
from environ_group import IEnvironContext

class LidarContext(ComponentContext):
    pass

class LidarCom(Component):
    def __init__(self, msgbus:MessageBusBase):
        super().__init__("lidar", "dafault", msgbus)

    def onRegister(self):
        self.pose_sub = self.createSubscriber(MessageID("pose"), MessageDataShape((3,)), ReduceMethod.REPLACE)


    def onEnvGroupInit(self, eg_ctx: IEnvironContext) -> ComponentContext:
        return LidarContext.build(eg_ctx=eg_ctx)