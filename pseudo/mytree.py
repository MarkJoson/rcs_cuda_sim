from abc import ABC, abstractmethod
from typing import Generic, TypeVar, List, Callable, Optional
from dataclasses import dataclass
from enum import Enum

__all__ = ['NotifyType', 'ITreeObserver', 'ITreeVisitor', 'ITreeNode', 'ITree', 'TreeNode', 'Tree', 'PathUtils']

T = TypeVar('T')


class PathType(Enum):
    ABS_PATH = 0    # 绝对路径
    REL_PATH = 1    # 相对路径
    SPEC_PATH = 2   # 前导为. 或.. 的相对路径

class PathUtils:
    @staticmethod
    def get_parent_path(path: str) -> str:
        if not path or path == "/":
            return "/"
        parent = "/".join(path.split("/")[:-1])
        return parent if parent else "/"

    @staticmethod
    def get_last_segment(path: str) -> str:
        if not path or path == "/":
            return ""
        return path.split("/")[-1]

    @staticmethod
    def join_paths(path1: str, path2: str) -> str:
        if not path1.endswith("/"):
            path1 += "/"
        if path2.startswith("/"):
            path2 = path2[1:]
        return path1 + path2

    @staticmethod
    def is_abs_path(path: str) -> bool:
        return path[0] == '/'

    @staticmethod
    def parse(path: str) -> List[str]:
        return [p for p in path.split("/") if p]

    @staticmethod
    def parse_relative_path(path: str, current_path: str) -> str:
        """
        解析相对路径，结合当前路径得到绝对路径
        
        Args:
            path: 相对路径，可以包含 . 或 ..
            current_path: 当前路径（必须是绝对路径）
        
        Returns:
            解析后的绝对路径
            
        Examples:
            parse_relative_path("../b", "/a/c") -> "/a/b"
            parse_relative_path("./b", "/a/c") -> "/a/c/b"
            parse_relative_path("b", "/a/c") -> "/a/c/b"
            parse_relative_path("../../b", "/a/c/d") -> "/a/b"
        """
        # 如果是绝对路径，直接返回normalize后的结果
        if path.startswith('/'):
            return PathUtils.normalize(path)
            
        # 确保current_path是绝对路径
        if not current_path.startswith('/'):
            raise ValueError("current_path must be an absolute path")
            
        # 如果path为空，返回当前路径
        if not path:
            return current_path
            
        # 如果path是 "." 开头，移除开头的 "./"
        if path.startswith('./'):
            path = path[2:]
        
        # 将当前路径和相对路径组合
        if current_path.endswith('/'):
            full_path = current_path + path
        else:
            full_path = current_path + '/' + path
            
        # 使用normalize处理 .. 和 . 
        return PathUtils.normalize(full_path)

    @staticmethod
    def combine(parts: List[str]) -> str:
        return "/".join(parts)

    @staticmethod
    def normalize(path: str) -> str:
        parts = [p for p in path.split("/") if p and p != "."]
        result = []
        for part in parts:
            if part == "..":
                if result:
                    result.pop()
            else:
                result.append(part)
        return "/" + "/".join(result)
    
    @staticmethod
    def get_relative_path(target_path:str, rel_path:str):
        rel_parts = PathUtils.parse(rel_path)
        target_parts = PathUtils.parse(target_path)

        # 找到共同祖先的深度
        common_prefix_len = 0
        for i in range(min(len(rel_parts), len(target_parts))):
            if rel_parts[i] != target_parts[i]:
                break
            common_prefix_len = i + 1

        # 计算需要往上走几级
        up_levels = len(rel_parts) - common_prefix_len

        # 构建相对路径
        relative_parts = ['..'] * up_levels
        relative_parts.extend(target_parts[common_prefix_len:])

        return '/'.join(relative_parts)



class NotifyType(Enum):
    NODE_ADDED = 1
    NODE_REMOVED = 2
    NODE_CHANGED = 3

class ITreeObserver(Generic[T], ABC):
    @abstractmethod
    def on_node_added(self, node: 'ITreeNode[T]') -> None:
        pass

    @abstractmethod
    def on_node_removed(self, node: 'ITreeNode[T]') -> None:
        pass

    @abstractmethod
    def on_node_changed(self, node: 'ITreeNode[T]') -> None:
        pass



class ITreeVisitor(Generic[T], ABC):
    @abstractmethod
    def visit_enter(self, node: 'ITreeNode[T]') -> None:
        pass

    @abstractmethod
    def visit_leave(self, node: 'ITreeNode[T]') -> None:
        pass



class ITreeNode(Generic[T], ABC):
    @abstractmethod
    def get_name(self) -> str:
        pass

    @abstractmethod
    def set_name(self, name: str) -> None:
        pass

    @abstractmethod
    def get_parent(self) -> Optional['ITreeNode[T]']:
        pass

    @abstractmethod
    def get_children(self) -> List['ITreeNode[T]']:
        pass

    @abstractmethod
    def get_data(self) -> Optional[T]:
        pass

    @abstractmethod
    def add_child(self, node: 'ITreeNode[T]') -> None:
        pass

    @abstractmethod
    def remove_child(self, node: 'ITreeNode[T]') -> None:
        pass

    @abstractmethod
    def find_child(self, predicate: Callable[['ITreeNode[T]'], bool]) -> Optional['ITreeNode[T]']:
        pass

    @abstractmethod
    def get_path(self) -> str:
        """获取从根节点到当前节点的路径"""
        pass

    @abstractmethod
    def find(self, predicate: Callable[['ITreeNode[T]'], bool]) -> Optional['ITreeNode[T]']:
        pass

    @abstractmethod
    def traverse(self, visitor: ITreeVisitor[T]) -> None:
        pass

    @abstractmethod
    def find_by_path(self, path: str) -> Optional['ITreeNode[T]']:
        pass


class ITree(Generic[T], ITreeNode[T]):
    # TODO. Tree就是名称为为'/'的根节点，或者名称为''的节点
    pass



class TreeNode(ITreeNode[T]):
    def __init__(self, data: Optional[T]):
        self.name: str = ""
        self.parent: Optional[ITreeNode[T]] = None
        self.children: List[ITreeNode[T]] = []
        self.data: Optional[T] = data
        self.observers: List[ITreeObserver[T]] = []

    def get_name(self) -> str:
        return self.name

    def set_name(self, name: str) -> None:
        self.name = name

    def get_parent(self) -> Optional[ITreeNode[T]]:
        return self.parent

    def get_children(self) -> List[ITreeNode[T]]:
        return self.children

    def get_data(self) -> Optional[T]:
        return self.data

    def add_child(self, node: ITreeNode[T]) -> None:
        if isinstance(node, TreeNode):
            node.parent = self
            self.children.append(node)
            self.notify_observers(NotifyType.NODE_ADDED, node)

    def remove_child(self, node: ITreeNode[T]) -> None:
        if node in self.children:
            self.children.remove(node)
            if isinstance(node, TreeNode):
                node.parent = None
            self.notify_observers(NotifyType.NODE_REMOVED, node)

    def find_child(self, predicate: Callable[[ITreeNode[T]], bool]) -> Optional[ITreeNode[T]]:
        for child in self.children:
            if predicate(child):
                return child
        return None

    def notify_observers(self, notify_type: NotifyType, node: 'ITreeNode[T]') -> None:
        for observer in self.observers:
            if notify_type == NotifyType.NODE_ADDED:
                observer.on_node_added(node)
            elif notify_type == NotifyType.NODE_REMOVED:
                observer.on_node_removed(node)
            elif notify_type == NotifyType.NODE_CHANGED:
                observer.on_node_changed(node)

    def get_path(self) -> str:
        """
        获取从根节点到当前节点的完整路径
        返回格式如: /root/parent/current
        """
        path_parts = []
        current: Optional[ITreeNode[T]] = self

        # 从当前节点往上遍历到根节点，收集所有节点名称
        while current:
            path_parts.insert(0, current.get_name())
            current = current.get_parent()
        
        # 根节点的名称为""
        return PathUtils.combine(path_parts)

    # 可选：添加一个辅助方法来获取相对路径
    def get_relative_path(self, target: 'ITreeNode[T]') -> str:
        """计算从当前节点到目标节点的相对路径"""
        if self == target:
            return "."

        return PathUtils.get_relative_path(
            target_path=target.get_path(),
            rel_path=self.get_path()
        )
    
    def find(self, predicate: Callable[[ITreeNode[T]], bool]) -> Optional[ITreeNode[T]]:
        def _find(node: ITreeNode[T]) -> Optional[ITreeNode[T]]:
            if predicate(node):
                return node
            for child in node.get_children():
                result = _find(child)
                if result:
                    return result
            return None

        return _find(self)

    def traverse(self, visitor: ITreeVisitor[T]) -> None:
        def _traverse(node: ITreeNode[T]) -> None:
            visitor.visit_enter(node)
            for child in node.get_children():
                _traverse(child)
            visitor.visit_leave(node)

        return _traverse(self)

    def find_by_path(self, path: str) -> Optional[ITreeNode[T]]:
        if not path:
            return None

        path_parts = PathUtils.parse(path=path)

        current = self              # 此处指向 '/'

        for part in path_parts:
            found = False
            for child in current.get_children():
                if child.get_name() == part:
                    current = child
                    found = True
                    break
            if not found:
                return None

        return current


class Tree(TreeNode[T]):
    def __init__(self, data:Optional[T] = None):
        super().__init__(data)
        self.set_name("")

