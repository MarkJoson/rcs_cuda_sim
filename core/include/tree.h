#ifndef TREE_H
#define TREE_H

#include <string>
#include <vector>
#include <functional>
#include <memory>
#include <optional>


enum class NotifyType
{
    NODE_ADDED,
    NODE_REMOVED,
    NODE_CHANGED
};

class PathUtils
{
public:
    static std::string getParentPath(const std::string &path);
    static std::string getLastSegment(const std::string &path);
    static std::string joinPaths(const std::string &path1, const std::string &path2);
    static bool isAbsPath(const std::string &path);
    static std::vector<std::string> parse(const std::string &path);
    static std::string parseRelativePath(const std::string &path, const std::string &currentPath);
    static std::string combine(const std::vector<std::string> &parts);
    static std::string normalize(const std::string &path);
    static std::string getRelativePath(const std::string &targetPath, const std::string &relPath);
};

class ITreeNode;

class ITreeObserver
{
public:
    virtual ~ITreeObserver() = default;
    virtual void onNodeAdded(ITreeNode *node) = 0;
    virtual void onNodeRemoved(ITreeNode *node) = 0;
    virtual void onNodeChanged(ITreeNode *node) = 0;
};

class ITreeVisitor
{
public:
    virtual ~ITreeVisitor() = default;
    virtual void visitEnter(ITreeNode *node) = 0;
    virtual void visitLeave(ITreeNode *node) = 0;
};

class ITreeNode
{
public:
    virtual ~ITreeNode() = default;
    virtual std::string getName() const = 0;
    virtual void setName(const std::string &name) = 0;
    virtual ITreeNode *getParent() const = 0;
    virtual std::vector<ITreeNode *> getChildren() const = 0;
    virtual void addChild(ITreeNode *node) = 0;
    virtual void removeChild(ITreeNode *node) = 0;
    virtual ITreeNode *findChild(const std::function<bool(ITreeNode *)> &predicate) = 0;
    virtual std::string getPath() const = 0;
    virtual ITreeNode *find(const std::function<bool(ITreeNode *)> &predicate) = 0;
    virtual void traverse(ITreeVisitor *visitor) = 0;
    virtual ITreeNode *findByPath(const std::string &path) = 0;
};

template <typename T>
class TreeNode : public ITreeNode
{
protected:
    std::string name_;
    ITreeNode *parent_;
    std::vector<ITreeNode *> children_;
    std::vector<ITreeObserver *> observers_;

public:
    explicit TreeNode()
        : name_(), parent_(nullptr) {}

    std::string getName() const override
    {
        return name_;
    }
    
    void setName(const std::string &name) override
    {
        name_ = name;
    }
    
    ITreeNode *getParent() const override
    {
        return parent_;
    }

    std::vector<ITreeNode *> getChildren() const override
    {
        return children_;
    }

    void addChild(ITreeNode *node) override
    {
        auto treeNode = dynamic_cast<TreeNode *>(node);
        if (treeNode)
        {
            treeNode->parent_ = this;
            children_.push_back(node);
            notifyObservers(NotifyType::NODE_ADDED, node);
        }
    }
    
    void removeChild(ITreeNode *node) override
    {
        auto it = std::find(children_.begin(), children_.end(), node);
        if (it != children_.end())
        {
            children_.erase(it);
            auto treeNode = dynamic_cast<TreeNode *>(node);
            if (treeNode)
            {
                treeNode->parent_ = nullptr;
            }
            notifyObservers(NotifyType::NODE_REMOVED, node);
        }
    }
    
    ITreeNode *findChild(const std::function<bool(ITreeNode *)> &predicate) override
    {
        for (auto child : children_)
        {
            if (predicate(child))
            {
                return child;
            }
        }
        return nullptr;
    }
    
    std::string getPath() const override
    {
        std::vector<std::string> pathParts;
        const ITreeNode *current = this;

        while (current->getParent())
        {
            pathParts.insert(pathParts.begin(), current->getName());
            current = current->getParent();
        }

        return PathUtils::combine(pathParts);
    }
    
    ITreeNode *find(const std::function<bool(ITreeNode *)> &predicate) override
    {
        if (predicate(this))
            return this;

        for (auto child : children_)
        {
            if (auto result = child->find(predicate))
            {
                return result;
            }
        }

        return nullptr;
    }
    
    void traverse(ITreeVisitor *visitor) override
    {
        visitor->visitEnter(this);
        for (auto child : children_)
        {
            child->traverse(visitor);
        }
        visitor->visitLeave(this);
    }
    
    ITreeNode *findByPath(const std::string &path) override
    {
        if (path.empty())
            return nullptr;

        auto pathParts = PathUtils::parse(path);
        ITreeNode *current = this;

        for (const auto &part : pathParts)
        {
            bool found = false;
            for (auto child : current->getChildren())
            {
                if (child->getName() == part)
                {
                    current = child;
                    found = true;
                    break;
                }
            }
            if (!found)
                return nullptr;
        }

        return current;
    }

    std::string getRelativePath(ITreeNode *target) const
    {
        if (this == target)
            return ".";
        return PathUtils::getRelativePath(target->getPath(), this->getPath());
    }

    void addObserver(ITreeObserver *observer)
    {
        observers_.push_back(observer);
    }
    
    void removeObserver(ITreeObserver *observer)
    {
        auto it = std::find(observers_.begin(), observers_.end(), observer);
        if (it != observers_.end())
        {
            observers_.erase(it);
        }
    }

protected:
    void notifyObservers(NotifyType type, ITreeNode *node)
    {
        for (auto observer : observers_)
        {
            switch (type)
            {
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
};

template <typename T>
class Tree : public TreeNode
{
public:
    explicit Tree(const std::optional &data = std::nullopt) : TreeNode(data)
    {
        this->setName("");
    }
};

#endif // TREE_H