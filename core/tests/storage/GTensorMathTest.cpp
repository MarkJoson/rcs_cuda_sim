#include <gtest/gtest.h>
#include "storage/GTensor.h"
#include "storage/TensorRegistryManager.h"

namespace test {

class GTensorMathTest : public ::testing::Test {
protected:
    void SetUp() override {
        registry = RSG_SIM::TensorRegistryManager::createRegistry(2);
    }

    RSG_SIM::TensorRegistry registry;
};

// 基础数学操作测试
TEST_F(GTensorMathTest, BasicMathOperations) {
    auto* tensor = registry.createTensor<float>("test", {2, 2});
    float fill_value = -2.0f;
    tensor->fill(&fill_value);
    
    tensor->toCPU();

    // 测试abs
    tensor->abs();
    float* data = static_cast<float*>(tensor->data());
    for (int i = 0; i < 8; ++i) {
        EXPECT_FLOAT_EQ(data[i], 2.0f);
    }
    
    // 测试clip
    float min_val = 1.0f, max_val = 1.5f;
    tensor->clip(&min_val, &max_val);
    data = static_cast<float*>(tensor->data());
    for (int i = 0; i < 8; ++i) {
        EXPECT_FLOAT_EQ(data[i], 1.5f);
    }
}

// 统计操作测试
TEST_F(GTensorMathTest, StatisticalOperations) {
    auto* tensor = registry.createTensor<float>("test", {2, 2});
    float fill_value = 2.0f;
    tensor->fill(&fill_value);

    tensor->toCPU();
    
    // 测试sum
    auto sum = tensor->sum();
    EXPECT_FLOAT_EQ(*static_cast<float*>(sum->data()), 16.0f); // 2*2*2*2
    
    // 测试mean
    auto mean = tensor->mean();
    EXPECT_FLOAT_EQ(*static_cast<float*>(mean->data()), 2.0f);
    
    // 测试维度上的统计
    auto dim_sum = tensor->sum(1);
    EXPECT_EQ(dim_sum->shape(), std::vector<int64_t>({2, 2}));
}

// 比较操作测试
TEST_F(GTensorMathTest, ComparisonOperations) {
    auto* tensor1 = registry.createTensor<float>("test1", {2, 2});
    auto* tensor2 = registry.createTensor<float>("test2", {2, 2});
    
    float val1 = 1.0f, val2 = 2.0f;
    tensor1->fill(&val1);
    tensor2->fill(&val2);
    
    tensor1->lt(*tensor2);

    tensor1->toCPU();

    // if(tensor1->isTypeMatch())

    bool* data = (bool*)(tensor1->data());
    for (int i = 0; i < 8; ++i) {
        EXPECT_FLOAT_EQ(data[i], 1.0f); // true in PyTorch is represented as 1.0f
    }
}

} // namespace test