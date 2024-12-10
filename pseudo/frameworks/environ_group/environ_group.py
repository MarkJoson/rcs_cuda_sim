from common_types import *
from environ_config import EnvironGroupConfig
from environ_context import EnvGroupContextManager


# 添加一个环境组管理器来统一管理所有环境组
class EnvironGroupHub:
    def __init__(self):
        self.env_groups: Dict[str, EnvGroupContextManager] = {}

    def create_env_group(self, env_group_id: str, config: EnvironGroupConfig) -> EnvGroupContextManager:
        env_ctx_manager = EnvGroupContextManager(env_group_id, config)
        env_ctx_manager.init_contexts()
        self.env_groups[env_group_id] = env_ctx_manager
        return env_ctx_manager

    def get_env_group(self, env_group_id: str) -> Optional[EnvGroupContextManager]:
        return self.env_groups.get(env_group_id)

    def remove_env_group(self, env_group_id: str):
        if env_group_id in self.env_groups:
            del self.env_groups[env_group_id]

