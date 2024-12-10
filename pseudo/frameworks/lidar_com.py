from common_types import *
from components import ComponentContext, Component
from message_bus_base import MessageBusBase
from environ_group import ContextBase

class LidarContext(ComponentContext):
    pass

class LidarCom(Component):
    def __init__(self, msgbus:MessageBusBase):
        super().__init__("lidar", "dafault", "lidar_ctx", msgbus)


    def onRegister(self):
        self.pose_sub = self.createSubscriber(
            message_id="pose", shape=(3,), reduce_method=ReduceMethod.REPLACE)


    def onEnvGroupInit(self, ctx: ContextBase) -> Optional[ComponentContext]:
        return LidarContext(**ctx.get_base_dict())