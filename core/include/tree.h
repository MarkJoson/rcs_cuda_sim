#ifndef TREE_H
#define TREE_H

#include <string>
#include <vector>
#include <functional>
#include <memory>
#include <optional>

enum class PathType {
    ABS_PATH,   // 绝对路径
    REL_PATH,   // 相对路径
    SPEC_PATH   // 前导为. 或.. 的相对路径
};

enum class NotifyType {
    NODE_ADDED,
    NODE_REMOVED,
    NODE_CHANGED
};

class PathUtils {
public:
    static std::string getParentPath(const std::string& path);
    static std::string getLastSegment(const std::string& path);
    static std::string joinPaths(const std::string& path1, const std::string& path2);
    static bool isAbsPath(const std::string& path);
    static std::vector<std::string> parse(const std::string& path);
    static std::string parseRelativePath(const std::string& path, const std::string& currentPath);
    static std::string combine(const std::vector<std::string>& parts);
    static std::string normalize(const std::string& path);
    static std::string getRelativePath(const std::string& targetPath, const std::string& relPath);
};

template<typename T>
class ITreeNode;

template<typename T>
class ITreeObserver {
public:
    virtual ~ITreeObserver() = default;
    virtual void onNodeAdded(ITreeNode<T>* node) = 0;
    virtual void onNodeRemoved(ITreeNode<T>* node) = 0;
    virtual void onNodeChanged(ITreeNode<T>* node) = 0;
};

template<typename T>
class ITreeVisitor {
public:
    virtual ~ITreeVisitor() = default;
    virtual void visitEnter(ITreeNode<T>* node) = 0;
    virtual void visitLeave(ITreeNode<T>* node) = 0;
};

template<typename T>
class ITreeNode {
public:
    virtual ~ITreeNode() = default;
    virtual std::string getName() const = 0;
    virtual void setName(const std::string& name) = 0;
    virtual ITreeNode<T>* getParent() const = 0;
    virtual std::vector<ITreeNode<T>*> getChildren() const = 0;
    virtual std::optional<T> getData() const = 0;
    virtual void addChild(ITreeNode<T>* node) = 0;
    virtual void removeChild(ITreeNode<T>* node) = 0;
    virtual ITreeNode<T>* findChild(const std::function<bool(ITreeNode<T>*)>& predicate) = 0;
    virtual std::string getPath() const = 0;
    virtual ITreeNode<T>* find(const std::function<bool(ITreeNode<T>*)>& predicate) = 0;
    virtual void traverse(ITreeVisitor<T>* visitor) = 0;
    virtual ITreeNode<T>* findByPath(const std::string& path) = 0;
};

template<typename T>
class TreeNode : public ITreeNode<T> {
protected:
    std::string name_;
    ITreeNode<T>* parent_;
    std::vector<ITreeNode<T>*> children_;
    std::optional<T> data_;
    std::vector<ITreeObserver<T>*> observers_;

public:
    explicit TreeNode(const std::optional<T>& data = std::nullopt);
    virtual ~TreeNode();

    std::string getName() const override;
    void setName(const std::string& name) override;
    ITreeNode<T>* getParent() const override;
    std::vector<ITreeNode<T>*> getChildren() const override;
    std::optional<T> getData() const override;
    void addChild(ITreeNode<T>* node) override;
    void removeChild(ITreeNode<T>* node) override;
    ITreeNode<T>* findChild(const std::function<bool(ITreeNode<T>*)>& predicate) override;
    std::string getPath() const override;
    ITreeNode<T>* find(const std::function<bool(ITreeNode<T>*)>& predicate) override;
    void traverse(ITreeVisitor<T>* visitor) override;
    ITreeNode<T>* findByPath(const std::string& path) override;

    std::string getRelativePath(ITreeNode<T>* target) const;
    void addObserver(ITreeObserver<T>* observer);
    void removeObserver(ITreeObserver<T>* observer);

protected:
    void notifyObservers(NotifyType type, ITreeNode<T>* node);
};

template<typename T>
class Tree : public TreeNode<T> {
public:
    explicit Tree(const std::optional<T>& data = std::nullopt);
};

template<typename T>
TreeNode<T>::TreeNode(const std::optional<T>& data)
    : name_(), parent_(nullptr), data_(data) {}

template<typename T>
TreeNode<T>::~TreeNode() {
    for (auto child : children_) {
        delete child;
    }
}

template<typename T>
std::string TreeNode<T>::getName() const {
    return name_;
}

template<typename T>
void TreeNode<T>::setName(const std::string& newName) {
    name_ = newName;
}

template<typename T>
ITreeNode<T>* TreeNode<T>::getParent() const {
    return parent_;
}

template<typename T>
std::vector<ITreeNode<T>*> TreeNode<T>::getChildren() const {
    return children_;
}

template<typename T>
std::optional<T> TreeNode<T>::getData() const {
    return data_;
}

template<typename T>
void TreeNode<T>::addChild(ITreeNode<T>* node) {
    auto treeNode = dynamic_cast<TreeNode<T>*>(node);
    if (treeNode) {
        treeNode->parent_ = this;
        children_.push_back(node);
        notifyObservers(NotifyType::NODE_ADDED, node);
    }
}

template<typename T>
void TreeNode<T>::removeChild(ITreeNode<T>* node) {
    auto it = std::find(children_.begin(), children_.end(), node);
    if (it != children_.end()) {
        children_.erase(it);
        auto treeNode = dynamic_cast<TreeNode<T>*>(node);
        if (treeNode) {
            treeNode->parent_ = nullptr;
        }
        notifyObservers(NotifyType::NODE_REMOVED, node);
    }
}
 
template<typename T>
ITreeNode<T>* TreeNode<T>::findChild(const std::function<bool(ITreeNode<T>*)>& predicate) {
    for (auto child : children_) {
        if (predicate(child)) {
            return child;
        }
    }
    return nullptr;
}

template<typename T>
void TreeNode<T>::notifyObservers(NotifyType type, ITreeNode<T>* node) {
    for (auto observer : observers_) {
        switch (type) {
            case NotifyType::NODE_ADDED:
                observer->onNodeAdded(node);
                break;
            case NotifyType::NODE_REMOVED:
                observer->onNodeRemoved(node);
                break;
            case NotifyType::NODE_CHANGED:
                observer->onNodeChanged(node);
                break;
        }
    }
}

template<typename T>
std::string TreeNode<T>::getPath() const {
    std::vector<std::string> pathParts;
    const ITreeNode<T>* current = this;
    
    while (current->getParent()) {
        pathParts.insert(pathParts.begin(), current->getName());
        current = current->getParent();
    }
    
    return PathUtils::combine(pathParts);
}

template<typename T>
ITreeNode<T>* TreeNode<T>::find(const std::function<bool(ITreeNode<T>*)>& predicate) {
    if (predicate(this)) return this;
    
    for (auto child : children_) {
        if (auto result = child->find(predicate)) {
            return result;
        }
    }
    
    return nullptr;
}

template<typename T>
void TreeNode<T>::traverse(ITreeVisitor<T>* visitor) {
    visitor->visitEnter(this);
    for (auto child : children_) {
        child->traverse(visitor);
    }
    visitor->visitLeave(this);
}

template<typename T>
ITreeNode<T>* TreeNode<T>::findByPath(const std::string& path) {
    if (path.empty()) return nullptr;
    
    auto pathParts = PathUtils::parse(path);
    ITreeNode<T>* current = this;
    
    for (const auto& part : pathParts) {
        bool found = false;
        for (auto child : current->getChildren()) {
            if (child->getName() == part) {
                current = child;
                found = true;
                break;
            }
        }
        if (!found) return nullptr;
    }
    
    return current;
}

template<typename T>
void TreeNode<T>::addObserver(ITreeObserver<T>* observer) {
    observers_.push_back(observer);
}

template<typename T>
void TreeNode<T>::removeObserver(ITreeObserver<T>* observer) {
    auto it = std::find(observers_.begin(), observers_.end(), observer);
    if (it != observers_.end()) {
        observers_.erase(it);
    }
}

template<typename T>
std::string TreeNode<T>::getRelativePath(ITreeNode<T>* target) const {
    if (this == target) return ".";
    return PathUtils::getRelativePath(target->getPath(), this->getPath());
}

template<typename T>
Tree<T>::Tree(const std::optional<T>& data) : TreeNode<T>(data) {
    this->setName("");
}

#endif // TREE_H