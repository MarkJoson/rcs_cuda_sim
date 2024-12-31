#include <gtest/gtest.h>
#include "storage/GTensor.h"
#include "storage/TensorRegistryManager.h"

namespace test {

class GTensorTypeSafetyTest : public ::testing::Test {
protected:
    void SetUp() override {
        registry = cuda_simulator::TensorRegistryManager::createRegistry(2);
    }

    cuda_simulator::TensorRegistry registry;
};

// 类型检查测试
TEST_F(GTensorTypeSafetyTest, TypeChecking) {
    auto* base_tensor = registry.createTensor<float>("test", {2, 2});

    EXPECT_TRUE(base_tensor->isTypeMatch(typeid(float)));
    EXPECT_FALSE(base_tensor->isTypeMatch(typeid(int)));

    auto* typed_tensor = dynamic_cast<cuda_simulator::GTensor<float>*>(base_tensor);
    EXPECT_NE(typed_tensor, nullptr);

    auto* wrong_type = dynamic_cast<cuda_simulator::GTensor<int>*>(base_tensor);
    EXPECT_EQ(wrong_type, nullptr);
}

// 类型安全操作测试
TEST_F(GTensorTypeSafetyTest, TypeSafeOperations) {
    auto* tensor = registry.createTensor<float>("test", {2, 2});
    auto* typed_tensor = dynamic_cast<cuda_simulator::GTensor<float>*>(tensor);
    ASSERT_NE(typed_tensor, nullptr);

    // 使用类型安全的方法
    float value = 1.5f;
    typed_tensor->fillValue(value);

    typed_tensor->toCPU();

    const float* data = typed_tensor->data();
    for (int i = 0; i < 8; ++i) {
        EXPECT_FLOAT_EQ(data[i], 1.5f);
    }
}

} // namespace test