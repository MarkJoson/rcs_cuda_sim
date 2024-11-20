#include <gtest/gtest.h>
#include "tree.h"
#include <string>

class TestObserver : public ITreeObserver<std::string> {
public:
    bool nodeAdded = false;
    bool nodeRemoved = false;
    bool nodeChanged = false;
    
    void onNodeAdded(ITreeNode<std::string>* node) override {
        nodeAdded = true;
    }
    
    void onNodeRemoved(ITreeNode<std::string>* node) override {
        nodeRemoved = true;
    }
    
    void onNodeChanged(ITreeNode<std::string>* node) override {
        nodeChanged = true;
    }
    
    void reset() {
        nodeAdded = false;
        nodeRemoved = false;
        nodeChanged = false;
    }
};

class TestVisitor : public ITreeVisitor<std::string> {
public:
    std::vector<std::string> visitedNodes;
    
    void visitEnter(ITreeNode<std::string>* node) override {
        visitedNodes.push_back("enter:" + node->getName());
    }
    
    void visitLeave(ITreeNode<std::string>* node) override {
        visitedNodes.push_back("leave:" + node->getName());
    }
};

// PathUtils测试
TEST(PathUtilsTest, BasicOperations) {
    EXPECT_EQ(PathUtils::getParentPath("/a/b/c"), "/a/b");
    EXPECT_EQ(PathUtils::getParentPath("/"), "/");
    EXPECT_EQ(PathUtils::getParentPath(""), "/");
    
    EXPECT_EQ(PathUtils::getLastSegment("/a/b/c"), "c");
    EXPECT_EQ(PathUtils::getLastSegment("/"), "");
    
    EXPECT_EQ(PathUtils::joinPaths("/a/b", "c"), "/a/b/c");
    EXPECT_EQ(PathUtils::joinPaths("/a/b/", "/c"), "/a/b/c");
    
    EXPECT_TRUE(PathUtils::isAbsPath("/a/b"));
    EXPECT_FALSE(PathUtils::isAbsPath("a/b"));
    
    auto parts = PathUtils::parse("/a/b/c");
    EXPECT_EQ(parts.size(), 3);
    EXPECT_EQ(parts[0], "a");
    EXPECT_EQ(parts[1], "b");
    EXPECT_EQ(parts[2], "c");
}

TEST(PathUtilsTest, NormalizeAndRelativePath) {
    EXPECT_EQ(PathUtils::normalize("/a/b/../c"), "/a/c");
    EXPECT_EQ(PathUtils::normalize("/a/./b/./c"), "/a/b/c");
    EXPECT_EQ(PathUtils::normalize("/a/b/../../c"), "/c");
    
    EXPECT_EQ(PathUtils::parseRelativePath("../b", "/a/c"), "/a/b");
    EXPECT_EQ(PathUtils::parseRelativePath("./b", "/a/c"), "/a/c/b");
    EXPECT_EQ(PathUtils::parseRelativePath("b", "/a/c"), "/a/c/b");
    
    EXPECT_EQ(PathUtils::getRelativePath("/a/b/c", "/a"), "b/c");
    EXPECT_EQ(PathUtils::getRelativePath("/a/d", "/a/b/c"), "../../d");
}

// Tree基本操作测试
class TreeTest : public ::testing::Test {
protected:
    Tree<std::string>* tree;
    TestObserver* observer;
    
    void SetUp() override {
        tree = new Tree<std::string>();
        observer = new TestObserver();
        tree->addObserver(observer);
    }
    
    void TearDown() override {
        delete tree;
        delete observer;
    }
};

TEST_F(TreeTest, BasicOperations) {
    // 测试初始状态
    EXPECT_EQ(tree->getName(), "");
    EXPECT_EQ(tree->getParent(), nullptr);
    EXPECT_TRUE(tree->getChildren().empty());
    
    // 添加子节点
    auto child1 = new TreeNode<std::string>("child1");
    child1->setName("child1");
    tree->addChild(child1);
    
    EXPECT_TRUE(observer->nodeAdded);
    EXPECT_EQ(tree->getChildren().size(), 1);
    EXPECT_EQ(child1->getParent(), tree);
    
    // 移除子节点
    tree->removeChild(child1);
    EXPECT_TRUE(observer->nodeRemoved);
    EXPECT_TRUE(tree->getChildren().empty());
}

TEST_F(TreeTest, PathOperations) {
    auto child1 = new TreeNode<std::string>("child1");
    child1->setName("child1");
    auto child2 = new TreeNode<std::string>("child2");
    child2->setName("child2");
    
    tree->addChild(child1);
    child1->addChild(child2);
    
    EXPECT_EQ(child2->getPath(), "/child1/child2");
    EXPECT_EQ(tree->findByPath("/child1/child2"), child2);
    EXPECT_EQ(tree->findByPath("/nonexistent"), nullptr);
    
    EXPECT_EQ(child1->getRelativePath(child2), "child2");
    EXPECT_EQ(child2->getRelativePath(child1), "..");
}

TEST_F(TreeTest, TraversalAndSearch) {
    auto child1 = new TreeNode<std::string>("child1");
    child1->setName("child1");
    auto child2 = new TreeNode<std::string>("child2");
    child2->setName("child2");
    
    tree->addChild(child1);
    child1->addChild(child2);
    
    TestVisitor visitor;
    tree->traverse(&visitor);
    
    EXPECT_EQ(visitor.visitedNodes.size(), 6); // 3 nodes * 2 (enter/leave)
    EXPECT_EQ(visitor.visitedNodes[0], "enter:");
    EXPECT_EQ(visitor.visitedNodes[1], "enter:child1");
    EXPECT_EQ(visitor.visitedNodes[2], "enter:child2");
    
    // 测试查找
    auto found = tree->find([](ITreeNode<std::string>* node) {
        return node->getName() == "child2";
    });
    EXPECT_EQ(found, child2);
    
    found = tree->find([](ITreeNode<std::string>* node) {
        return node->getName() == "nonexistent";
    });
    EXPECT_EQ(found, nullptr);
}

TEST_F(TreeTest, DataOperations) {
    auto child = new TreeNode<std::string>("data1");
    child->setName("child");
    
    EXPECT_TRUE(child->getData().has_value());
    EXPECT_EQ(child->getData().value(), "data1");
    
    tree->addChild(child);
    EXPECT_FALSE(tree->getData().has_value());
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}