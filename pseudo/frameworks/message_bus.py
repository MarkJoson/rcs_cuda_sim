import torch as th
import networkx as nx
from collections import deque, defaultdict
from common_types import *
from components import Component
from message_bus_base import MessageBusBase
from mesage_handler import IPublish, Publisher, Subscriber
from reducer_com import ReducerCom
from environ_group import EGManagedObject, ContextBase, EnvironGroupConfig

@dataclass
class MessageQueue:
    history: Deque[Tensor]
    max_history_len: int = 0                # Maximum history length needed by subscribers
    valid_count: int = 0                    # Count of valid messages in the queue
    latest_update_time: int = 0             # Track when the queue was last updated

    def __init__(self, max_history_len: int):
        self.history = deque(maxlen=max_history_len)
        self.max_history_len = max_history_len
        self.valid_count = 0
        self.latest_update_time = 0
        # 添加缓存以提高性能
        self._cache = {}
        self._cache_valid = False

    def get_history(self, offset: int) -> Tensor:
        cache_key = offset
        if self._cache_valid and cache_key in self._cache:
            return self._cache[cache_key]

        result = self.history[-1-offset]
        self._cache[cache_key] = result
        return result

    def append(self, data: Tensor):
        self._cache_valid = False  # 使缓存失效
        self.history.append(data)
        self.valid_count += min(self.max_history_len, self.valid_count + 1)
        self.latest_update_time += 1
        return self.valid_count

    def reset(self):
        self.history.clear()
        self.valid_count = 0
        self.latest_update_time = 0


@dataclass
class MessageBusContext(ContextBase):
    msg_queues: Dict[Tuple[ComponentID, MessageID], MessageQueue]  # Map (publisher_id, message_id) to queue


class MessageBus(EGManagedObject, MessageBusBase, IPublish):
    components          : Dict[ComponentID, 'Component'] = dict()
    component_graph_ids : Dict[ComponentID, GraphID] = dict()
    message_routes      : Dict[MessageID, Tuple[List[Publisher],List[Subscriber]]] = defaultdict(lambda :(list(), list()))
    reducer_nodes       : Dict[MessageID, Dict[ReduceMethod, ComponentID]] = dict()
    message_graph       : nx.MultiDiGraph = nx.MultiDiGraph()
    active_graph        : nx.MultiDiGraph = nx.MultiDiGraph()
    execution_graph     : nx.DiGraph = nx.DiGraph()
    execution_order     : Sequence[Sequence[ComponentID]] = list()

    # 分成不同的执行子图，如state_step,
    def registerComponent(self, component: Component):
        com_id = component.id

        # 检查当前component是否在
        assert com_id in self.components

        self.components[com_id] = component
        self.component_graph_ids[com_id] = component.graph_id


    def createPublisher(
        self,
        component_id: ComponentID,                  # 发布者所属节点ID
        message_id: MessageID,                      # 发布Tensor的消息Id
        shape: MessageDataShape,                    # 发布Tensor的形状
    ) -> "Publisher":
        new_publisher = Publisher(
            component_id=component_id,
            message_id=message_id,
            shape=shape,
            publish_if=self
        )

        self.message_routes[message_id][0].append(new_publisher)

        return new_publisher


    def createSubscriber(
        self,
        component_id: ComponentID,                  # 接收者所属节点Id
        message_id: MessageID,                      # 接收Tensor的消息Id
        shape: MessageDataShape,                    # 接收Tensor的形状
        history_offset:int,                         # 接受Tensor的时间点
        history_padding_val: Tensor,                # 无效历史的padding值
        reduce_method: ReduceMethod,                # 多个相同消息Tensor时的合并方法
    ) -> "Subscriber":
        new_subscriber = Subscriber(
            component_id=component_id,
            message_id=message_id,
            shape=shape,
            history_offset=history_offset,
            history_padding_val=history_padding_val,
            reduce_method=reduce_method,
        )

        self.message_routes[message_id][1].append(new_subscriber)

        return new_subscriber


    def publish(self, mbctx:MessageBusContext, publish_node: ComponentID, tensor_id: MessageID, tensor: Tensor):

        queue_key = (publish_node, tensor_id)
        if queue_key not in mbctx.msg_queues:
            raise ValueError(f"No message queue found for publisher {publish_node} and message {tensor_id}")

        if not isinstance(tensor, Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(tensor)}")

        expected_shape = self.message_routes[tensor_id][0][0].shape
        if tensor.shape[1:] != expected_shape:  # 忽略batch维度
            raise ValueError(f"Shape mismatch: expected {expected_shape}, got {tensor.shape[1:]}")

        valid_count = mbctx.msg_queues[queue_key].append(tensor)

        # Check if any subscribers need to be triggered
        publishers, subscribers = self.message_routes[tensor_id]
        for sub in subscribers:
            # 有效数据大于历史数据
            if sub.accept_invalid_history or valid_count >= sub.history_offset:
                # Trigger component execution
                self._trigger_component_execution(mbctx=mbctx, component_id=sub.component_id)

    def onEnvGroupInit(self, ctx: ContextBase) -> MessageBusContext:
        """初始化消息总线的环境组上下文"""
        ctx = MessageBusContext(
            msg_queues={},
            **ctx.get_base_dict()
        )
        self._initialize_message_queues(ctx)
        return ctx

    def onEnvGroupReset(self, context: MessageBusContext, reset_flags: Tensor):
        """重置消息队列"""
        for queue in context.msg_queues.values():
            queue.reset()


    @staticmethod
    def genReduceNodeID(message_id:MessageID, reduce_method:ReduceMethod):
        return f"_{message_id}.{reduce_method}", f"{message_id}.{reduce_method}"

    def buildGraph(self):

        # 处理所有多pub对一个消息的情况
        self._adjust_message_route()

        # 建立消息发送图
        self._build_message_graph()

        # 屏蔽所有无效节点，遍历辅助接点的后续节点，并将enabled设置为false
        self._disable_no_source_components()

        # 将目前使能的节点与边拷贝到active_graph中
        self._build_active_graph()

        # 检查输入输出形状是否正确
        self._check_active_graph_message_compatible()

        # 检查某个消息是否存在环路，如果存在环路需要打印并报错
        self._check_active_graph_loop()

        # 调整所有元件的路由属性:pubs, subs
        self._update_components_route_ref()

        # 由active_graph生成执行图
        self._build_execuation_graph()

        # 使用拓扑排序确定执行顺序并打印
        self.execution_order = self._group_topological_sort(self.execution_graph)


    def execuate(self):
        # 找到图中的源并执行 / 直接执行CUDA Graph
        raise NotImplementedError

    def _group_topological_sort(self, G:nx.DiGraph):
        # 初始化
        in_degree = dict(G.in_degree())
        zero_in_degree = [n for n,d in in_degree.items() if d == 0]
        groups = []
        seen = set()

        while zero_in_degree:
            # 当前超步的节点
            current_group = []

            # 处理当前所有入度为0的节点
            for node in zero_in_degree:
                if node not in seen:
                    current_group.append(node)
                    seen.add(node)

            # 更新图,重新计算入度
            next_zero_in_degree = []
            for node in current_group:
                # 更新其邻居节点的入度
                for neighbor in G.neighbors(node):
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        next_zero_in_degree.append(neighbor)

            groups.append(current_group)
            zero_in_degree = next_zero_in_degree

        return groups


    def _check_active_graph_loop(self):
        try:
            cycles = nx.find_cycle(self.active_graph)
            cycle_str = " -> ".join(str(node) for node in cycles)
            raise ValueError(f"Circular dependency detected in message graph: {cycle_str}")
        except nx.NetworkXNoCycle:
            pass


    def _check_active_graph_message_compatible(self):
        for message_id, (pubs, subs) in self.message_routes.items():
            if not pubs or not subs:
                continue

            pub_shape = pubs[0].shape
            for pub in pubs[1:]:
                if pub.shape != pub_shape:
                    raise ValueError(f"Inconsistent shapes for publishers of message {message_id}")

            for sub in subs:
                if sub.shape != pub_shape:
                    raise ValueError(f"Shape mismatch for subscriber of message {message_id}")


    def _adjust_message_route(self):
        new_message_route = self.message_routes.copy()

        # 处理多pub对一个消息的情况
        for message_id in self.message_routes:
            pubs, subs = self.message_routes[message_id]
            reduce_nodes: Set[ComponentID] = set()

            for sub in subs:
                if (pubs is None) or (len(pubs) <= 1):
                    continue
                # 多个发布者情况
                if(sub.reduce_method == ReduceMethod.STACK):
                    # 修改维度信息
                    sub.stack_dim = 1  # 默认在第一维度堆叠
                    sub.stack_order = [pub.component_id for pub in pubs]
                else:
                    # 非堆叠选项：创建reduce组件
                    reduce_node_id, new_message_id = MessageBus.genReduceNodeID(message_id=message_id,
                                                                reduce_method=sub.reduce_method)
                    if reduce_node_id not in reduce_nodes:
                        # 创建reduce组件
                        reduce_component = ReducerCom(
                            component_id=reduce_node_id,
                            message_id=message_id,
                            new_message_id=new_message_id,
                            shape=sub.shape,
                            reduce_method=sub.reduce_method,
                            msgbus=self,
                        )
                        # 挂载该组件
                        reduce_component.onRegister()

                        reduce_nodes.add(reduce_node_id)
                        self.components[reduce_node_id] = reduce_component

                        # 清空原有该消息的pub路径
                        new_message_route[new_message_id][0].extend(pubs)
                        new_message_route[message_id][0].clear()

                    # 更改subscriber的接收消息id
                    sub.message_id = new_message_id
                    # 更改原有的消息的sub路径
                    new_message_route[new_message_id][1].append(sub)
                    new_message_route[message_id][1].remove(sub)

        self.message_routes = new_message_route


    def _update_components_route_ref(self):
        """
        Updates the publisher/subscriber cross-references in existing pubs and subs
        based on current message routes.
        """
        # Go through all components
        for component in self.components.values():
            # Update publishers' subscribers references
            for pub in component.pubs:
                # Clear existing subscribers
                pub.subscribers = []
                # Find all subscribers for this message_id
                _, subscribers = self.message_routes[pub.message_id]
                # Add all subscribers that are enabled
                pub.subscribers.extend([sub for sub in subscribers if sub.is_enabled])

            # Update subscribers' publishers references
            for sub in component.subs:
                # Clear existing publishers
                sub.publishers = []
                # Find all publishers for this message_id
                publishers, _ = self.message_routes[sub.message_id]
                # Add all publishers that are enabled
                sub.publishers.extend([pub for pub in publishers if pub.is_enabled])


    def _build_message_graph(self):
        self.message_graph.clear()
        # 添加辅助节点，所有没有发布者的消息都连接到该节点
        self.message_graph.add_node("no_pub", obj=None)

        # 添加所有其他节点
        for com_id, com in self.components.items():
            self.message_graph.add_node(com_id, obj=com)

        # 建立消息依赖关系
        for message_id in self.message_routes:
            pubs, subs = self.message_routes[message_id]
            for sub in subs:
                if not pubs:
                    # 不存在发布者时，将消息连接到辅助节点
                    self.message_graph.add_edge("no_pub", sub.component_id, message_id)
                else:
                    for pub in pubs:
                        self.message_graph.add_edge(pub.component_id, sub.component_id, message_id)


    def _build_active_graph(self):
        self.active_graph = nx.MultiDiGraph()
        for node in self.message_graph:
            if node == "no_pub":
                continue
            if isinstance(node, ComponentID) and not self.components[node].enabled:
                continue
            self.active_graph.add_node(node)

        for u, v in self.message_graph.edges():
            if u == "no_pub" or v == "no_pub":
                continue
            if isinstance(u, ComponentID)  and not self.components[u].enabled:
                continue
            if isinstance(v, ComponentID)  and not self.components[v].enabled:
                continue
            self.active_graph.add_edge(u, v)


    def _build_execuation_graph(self):
        # 由active生成execgraph执行依赖图
        self.execution_graph = nx.DiGraph()
        for com_id, exec_id in self.component_graph_ids.items():
            if not com_id in self.active_graph:
                continue
            if not exec_id in self.execution_graph:
                self.execution_graph.add_node(exec_id)

        # 为相同执行图的节点添加边
        for u, v in self.active_graph.edges():
            if isinstance(u, ComponentID)  or not isinstance(v, ComponentID) :
                continue
            u_exec = self.component_graph_ids[u]
            v_exec = self.component_graph_ids[v]
            if u_exec != v_exec:
                self.execution_graph.add_edge(u_exec, v_exec)


    def _disable_no_source_components(self):
        for _, neighbor in nx.bfs_edges(self.message_graph, "no_pub"):
            if neighbor in self.components:
                self.components[neighbor].setEnabled(False)


    def _initialize_message_queues(self, ctx:MessageBusContext):
        """Initialize message queues for all publishers based on subscriber requirements."""
        for message_id, (publishers, subscribers) in self.message_routes.items():
            # Find maximum history offset required by subscribers
            max_history = 0
            for sub in subscribers:
                max_history = max(max_history, sub.history_offset)

            # Create queue for each publisher
            for pub in publishers:
                queue_key = (pub.component_id, message_id)
                ctx.msg_queues[queue_key] = MessageQueue(
                    max_history_len=max_history + 1,  # +1 for current message
                )


    def _trigger_component_execution(self, mbctx: MessageBusContext, component_id: ComponentID):
        """
        触发组件执行的方法。检查组件的所有输入是否就绪，如果就绪则执行组件。

        Args:
            mbctx: MessageBus上下文
            component_id: 要触发的组件ID
        """
        component = self.components[component_id]
        if not component.enabled:
            return

        # 收集所有输入数据
        input_data = {}
        all_inputs_ready = True

        # 检查每个订阅者的输入是否就绪
        for subscriber in component.subs:
            if not subscriber.is_enabled:
                continue

            # 获取该订阅者的发布者数据
            assert len(subscriber.publishers) == 1, "Each subscriber should have exactly one publisher after graph building"
            pub = subscriber.publishers[0]
            queue_key = (pub.component_id, subscriber.message_id)
            queue = mbctx.msg_queues[queue_key]

            if subscriber.history_offset == 0:
                # 使用当前数据时，必须等待数据就绪
                if queue.valid_count == 0:
                    return
                tensor = queue.get_history(0)
            else:
                # 使用历史数据时，检查是否就绪或是否接受无效输入
                if queue.valid_count < subscriber.history_offset + 1:
                    if not subscriber.accept_invalid_history:
                        all_inputs_ready = False
                        break
                    # 使用padding值
                    tensor = subscriber.history_padding_val
                else:
                    # 获取历史数据
                    tensor = queue.get_history(subscriber.history_offset)

            input_data[subscriber.message_id] = tensor

        # 所有输入就绪时执行组件
        if all_inputs_ready:
            context = mbctx.ctx_manager.get_context(component.context_id)
            component.onExecuate(context, input_data)

class GameEngineManager:
    # buses           : Mapping[str, TensorMessageBus]
    componenets     : Mapping[ComponentID, Component]

    def registerComponent(self):
        # TODO. 注册所有的Component
        raise NotImplementedError

    def execuate(self, bus):
        raise NotImplementedError

