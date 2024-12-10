import networkx as nx
from collections import deque, defaultdict
from common_types import *
from components import Component
from message_bus_base import MessageBusBase
from mesage_handler import IPublish, Publisher, Subscriber
from reducer_com import ReducerCom

@dataclass
class MessageQueue(deque[Tensor]):
    history: Deque[Tensor]
    max_elem: int = 0
    valid_count: int = 0


@dataclass
class MessageBusContext:
    msg_queues : Mapping[MessageID, MessageQueue]


class MessageBus(MessageBusBase, IPublish):
    components          : Dict[ComponentID, 'Component'] = dict()
    execution_groups    : Dict[ComponentID, GraphID] = dict()
    message_routes      : Dict[MessageID, Tuple[List[Publisher],List[Subscriber]]] = defaultdict(lambda :(list(), list()))
    reducer_nodes       : Dict[MessageID, Dict[ReduceMethod, ComponentID]] = dict()
    message_graph       : nx.MultiGraph = nx.MultiGraph()
    active_graph        : nx.MultiGraph = nx.MultiGraph()
    execution_graph     : nx.Graph = nx.Graph()

    # 分成不同的执行子图，如state_step,
    def registerNode(self, component: Component, exec_graph: GraphID):
        com_id = component.id

        # 检查当前component是否在
        assert com_id in self.components

        self.components[com_id] = component
        self.execution_groups[com_id] = component.exec_graph
        # component_id
        # return super().registerNode(component_id, exec_graph)


    def createPublisher(
        self,
        component_id: ComponentID,
        tensor_id: MessageID,
        shape: TensorShape,
        max_history_len: int,
        history_padding_val: Tensor
    ) -> Publisher:
        pub = Publisher(
            component_id=component_id,
            message_id=tensor_id,
            shape=shape,
            history_padding_val=history_padding_val,
            publish_if=self
        )

        self.message_routes[tensor_id][0].append(pub)

        return pub


    def createSubscriber(
        self,
        component_id: ComponentID,
        tensor_id: MessageID,
        shape: TensorShape,
        reduce_method: ReduceMethod,
        history_offset: int
    ) -> Subscriber:
        sub = Subscriber(
            component_id=component_id,
            message_id=tensor_id,
            shape=shape,
            reduce_method=reduce_method,
            history_offset=history_offset,
        )

        self.message_routes[tensor_id][1].append(sub)

        return sub

    @staticmethod
    def reduceNodeName(msg_id:MessageID, reduce_method:ReduceMethod):
        return ComponentID(f"{msg_id}.{str(ReduceMethod)}")

    def buildGraph(self):
        # 添加辅助节点，所有没有发布者的消息都连接到该节点
        self.message_graph.add_node("no_pub", obj=None)

        # 添加所有其他节点
        for com_id, com in self.components.items():
            self.message_graph.add_node(com_id, obj=com)

        # 建立消息依赖关系
        for msg_id in self.message_routes:
            pubs, subs = self.message_routes[msg_id]
            reduce_nodes: Mapping[ReduceMethod, ComponentID] = dict()

            for sub in subs:
                if not pubs:
                    # 不存在发布者时，将消息连接到辅助节点
                    self.message_graph.add_edge("no_pub", sub.component_id)
                    continue

                if len(pubs) == 1:
                    # 单个发布者情况：直接建立连接
                    self.message_graph.add_edge(pubs[0].component_id, sub.component_id)
                    sub.publishers = [pubs[0]]
                    continue

                # 多个发布者情况
                if sub.reduce_method == ReduceMethod.STACK:
                    # 堆叠选项：更新subscriber的堆叠属性
                    sub.stack_dim = 1  # 默认在第一维度堆叠
                    sub.stack_order = [pub.component_id for pub in pubs]
                    sub.publishers = pubs
                    for pub in pubs:
                        self.message_graph.add_edge(pub.component_id, sub.component_id)
                else:
                    # 非堆叠选项：创建reduce组件
                    reduce_node_id = MessageBus.reduceNodeName(msg_id=msg_id,
                                                                reduce_method=sub.reduce_method)
                    if reduce_node_id not in reduce_nodes:
                        # 创建reduce组件
                        reduce_component = ReducerCom(
                            component_id=reduce_node_id,
                            msg_id=msg_id,
                            shape=sub.shape,
                            reduce_method=sub.reduce_method,
                            msgbus=self
                        )
                        reduce_nodes[sub.reduce_method] = reduce_node_id
                        self.components[reduce_node_id] = reduce_component
                        self.message_graph.add_node(reduce_node_id, obj=reduce_component)

                        # 连接所有发布者到reduce组件
                        for pub in pubs:
                            self.message_graph.add_edge(pub.component_id, reduce_node_id)

                    # 连接reduce组件到订阅者
                    self.message_graph.add_edge(reduce_node_id, sub.component_id)

        # 屏蔽所有无效节点，遍历辅助接点的后续节点，并将enabled设置为false
        for _, neighbor in nx.bfs_edges(self.message_graph, "no_pub"):
            if neighbor in self.components:
                self.components[neighbor].setEnabled(False)

        # 将目前使能的节点与边拷贝到valid_msg_graph中
        self.active_graph = nx.MultiGraph()
        for node in self.message_graph:
            if node == "no_pub":
                continue
            if type(node) == type(ComponentID("")) and not self.components[node].enabled:
                continue
            self.active_graph.add_node(node)

        for u, v in self.message_graph.edges():
            if u == "no_pub" or v == "no_pub":
                continue
            if type(u) == type(ComponentID("")) and not self.components[u].enabled:
                continue
            if type(v) == type(ComponentID("")) and not self.components[v].enabled:
                continue
            self.active_graph.add_edge(u, v)

        # 检查某个消息是否存在环路，如果存在环路需要打印并报错
        try:
            cycles = nx.find_cycle(self.active_graph)
            cycle_str = " -> ".join(str(node) for node in cycles)
            raise ValueError(f"Circular dependency detected in message graph: {cycle_str}")
        except nx.NetworkXNoCycle:
            pass

        # 检查输入输出形状是否正确
        for msg_id, (pubs, subs) in self.message_routes.items():
            if not pubs or not subs:
                continue

            pub_shape = pubs[0].shape
            for pub in pubs[1:]:
                if pub.shape != pub_shape:
                    raise ValueError(f"Inconsistent shapes for publishers of message {msg_id}")

            for sub in subs:
                if sub.shape != pub_shape:
                    raise ValueError(f"Shape mismatch for subscriber of message {msg_id}")

        # 由valid_msg_graph生成execgraph执行依赖图
        self.execution_graph = nx.Graph()
        for com_id, exec_id in self.execution_groups.items():
            if not com_id in self.active_graph:
                continue
            if not exec_id in self.execution_graph:
                self.execution_graph.add_node(exec_id)

        # 为相同执行图的节点添加边
        for u, v in self.active_graph.edges():
            if type(u) == type(ComponentID("")) or not type(v) == type(ComponentID("")):
                continue
            u_exec = self.execution_groups[u]
            v_exec = self.execution_groups[v]
            if u_exec != v_exec:
                self.execution_graph.add_edge(u_exec, v_exec)

        # 使用拓扑排序确定执行顺序并打印
        try:
            execution_order = list(nx.topological_sort(self.execution_graph))
            print("Execution order of graphs:", " -> ".join(map(str, execution_order)))
        except nx.NetworkXUnfeasible:
            raise ValueError("Circular dependency detected in execution graphs")

    def execCPU(self):
        pass

    def execuate(self):
        # 找到图中的源并执行 / 直接执行CUDA Graph
        raise NotImplementedError


class GameEngineManager:
    # buses           : Mapping[str, TensorMessageBus]
    componenets     : Mapping[ComponentID, Component]

    def registerComponent(self):
        # TODO. 注册所有的Component
        raise NotImplementedError

    def execuate(self, bus):
        raise NotImplementedError

