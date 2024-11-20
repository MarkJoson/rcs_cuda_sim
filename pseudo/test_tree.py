import unittest
from typing import List, Optional

from mytree import *

class TestTree(unittest.TestCase):
    def setUp(self):
        # 创建测试用的树结构
        self.tree = Tree[str]()
        self.root_node = TreeNode[str]("root data")
        self.root_node.set_name("root")
        self.tree.add_child(self.root_node)
        
        self.child1 = TreeNode[str]("child1 data")
        self.child1.set_name("child1")
        self.root_node.add_child(self.child1)
        
        self.child2 = TreeNode[str]("child2 data")
        self.child2.set_name("child2")
        self.root_node.add_child(self.child2)
        
        self.grandchild = TreeNode[str]("grandchild data")
        self.grandchild.set_name("grandchild")
        self.child1.add_child(self.grandchild)

    def test_basic_operations(self):
        # 测试基本的树操作
        self.assertEqual(self.root_node.get_name(), "root")
        self.assertEqual(len(self.root_node.get_children()), 2)
        self.assertEqual(self.child1.get_parent(), self.root_node)
        self.assertEqual(self.child1.get_data(), "child1 data")

    def test_path_operations(self):
        # 测试路径相关操作
        self.assertEqual(self.grandchild.get_path(), "/root/child1/grandchild")
        self.assertEqual(self.child2.get_path(), "/root/child2")
        
        # 测试相对路径
        relative_path = self.child1.get_relative_path(self.child2)
        self.assertEqual(relative_path, "../child2")
        
        relative_path = self.grandchild.get_relative_path(self.child2)
        self.assertEqual(relative_path, "../../child2")

    def test_find_operations(self):
        # 测试查找操作
        found_node = self.tree.find_by_path("/root/child1/grandchild")
        self.assertEqual(found_node, self.grandchild)
        
        # 测试条件查找
        found_node = self.tree.find(lambda node: node.get_data() == "child2 data")
        self.assertEqual(found_node, self.child2)

    def test_path_utils(self):
        # 测试路径工具类
        self.assertEqual(PathUtils.get_parent_path("/root/child1"), "/root")
        self.assertEqual(PathUtils.get_last_segment("/root/child1"), "child1")
        self.assertEqual(PathUtils.normalize("/root/./child1/../child2"), "/root/child2")
        self.assertEqual(PathUtils.join_paths("/root", "child1"), "/root/child1")

class TestTreeObserver(ITreeObserver[str]):
    def __init__(self):
        self.added_nodes: List[ITreeNode[str]] = []
        self.removed_nodes: List[ITreeNode[str]] = []
        self.changed_nodes: List[ITreeNode[str]] = []

    def on_node_added(self, node: ITreeNode[str]) -> None:
        self.added_nodes.append(node)

    def on_node_removed(self, node: ITreeNode[str]) -> None:
        self.removed_nodes.append(node)

    def on_node_changed(self, node: ITreeNode[str]) -> None:
        self.changed_nodes.append(node)

class TestTreeVisitor(ITreeVisitor[str]):
    def __init__(self):
        self.visited_nodes: List[str] = []

    def visit_enter(self, node: ITreeNode[str]) -> None:
        self.visited_nodes.append(f"enter:{node.get_name()}")

    def visit_leave(self, node: ITreeNode[str]) -> None:
        self.visited_nodes.append(f"leave:{node.get_name()}")

class TestObserverAndVisitor(unittest.TestCase):
    def setUp(self):
        self.tree = Tree[str]()
        self.root_node = TreeNode[str]("root")
        self.root_node.set_name("root")
        self.observer = TestTreeObserver()
        self.root_node.observers.append(self.observer)

    def test_observer(self):
        # 测试观察者模式
        child = TreeNode[str]("child")
        child.set_name("child")
        self.root_node.add_child(child)
        self.assertEqual(len(self.observer.added_nodes), 1)
        
        self.root_node.remove_child(child)
        self.assertEqual(len(self.observer.removed_nodes), 1)

    def test_visitor(self):
        # 测试访问者模式
        child = TreeNode[str]("child")
        child.set_name("child")
        self.root_node.add_child(child)
        
        visitor = TestTreeVisitor()
        self.root_node.traverse(visitor)
        
        expected_visits = [
            "enter:root",
            "enter:child",
            "leave:child",
            "leave:root"
        ]
        self.assertEqual(visitor.visited_nodes, expected_visits)

if __name__ == '__main__':
    unittest.main()