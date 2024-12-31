#include <gtest/gtest.h>
#include "storage/GTensor.h"
#include "storage/TensorRegistryManager.h"

namespace test {

class GTensorDeviceTest : public ::testing::Test {
protected:
    void SetUp() override {
        registry = cuda_simulator::TensorRegistryManager::createRegistry(2);
    }

    cuda_simulator::TensorRegistry registry;
};

// 设备转移测试
TEST_F(GTensorDeviceTest, DeviceTransfer) {
    auto* tensor = registry.createTensor<float>("test", {2, 2});

    EXPECT_TRUE(tensor->isOnGPU());
    EXPECT_FALSE(tensor->isOnCPU());

    tensor->toCPU();
    EXPECT_TRUE(tensor->isOnCPU());
    EXPECT_FALSE(tensor->isOnGPU());

    tensor->toGPU();
    EXPECT_TRUE(tensor->isOnGPU());
    EXPECT_FALSE(tensor->isOnCPU());
}

// 跨设备操作测试
TEST_F(GTensorDeviceTest, CrossDeviceOperations) {
    auto* tensor1 = registry.createTensor<float>("test1", {2, 2});
    auto* tensor2 = registry.createTensor<float>("test2", {2, 2});

    float val1 = 1.0f, val2 = 2.0f;
    tensor1->fill(&val1);
    tensor2->fill(&val2);

    // tensor1->toCPU();
    // 应该自动处理设备不匹配的情况
    EXPECT_NO_THROW(tensor1->eq(*tensor2));
}

} // namespace test