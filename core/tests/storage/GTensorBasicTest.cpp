#include <gtest/gtest.h>
#include "../../include/storage/GTensor.h"
#include "../../include/storage/TensorRegistryManager.h"

namespace test {

class GTensorTest : public ::testing::Test {
protected:
    void SetUp() override {
        registry = RSG_SIM::TensorRegistryManager::createRegistry(2); // 使用小的env_count便于测试
    }

    RSG_SIM::TensorRegistry registry;
};

// 基础创建和元信息测试
TEST_F(GTensorTest, Creation) {
    auto* tensor = registry.createTensor<float>("test", {2, 3});
    ASSERT_NE(tensor, nullptr);
    
    const auto& meta = tensor->meta();
    EXPECT_EQ(meta.name, "test");
    EXPECT_EQ(meta.shape, std::vector<int64_t>({2, 2, 3})); // 包含env_count
    EXPECT_EQ(meta.dtype, RSG_SIM::TensorDataType::kFloat32);
    EXPECT_EQ(meta.type_size, sizeof(float));
    EXPECT_EQ(meta.type_info, std::type_index(typeid(float)));
}

// 数据操作测试
TEST_F(GTensorTest, DataOperations) {
    auto* tensor = registry.createTensor<float>("test", {2, 2});
    
    // 测试zero
    tensor->zero();
    tensor->toCPU();
    float* data = static_cast<float*>(tensor->ptr());
    for (int i = 0; i < 8; ++i) { // 2(env) * 2 * 2
        EXPECT_FLOAT_EQ(data[i], 0.0f);
    }
    
    // 测试fill
    float fill_value = 1.5f;
    tensor->fill(&fill_value);
    for (int i = 0; i < 8; ++i) {
        EXPECT_FLOAT_EQ(data[i], 1.5f);
    }
}

// 形状操作测试
TEST_F(GTensorTest, ShapeOperations) {
    auto* tensor = registry.createTensor<float>("test", {2, 3});
    
    // 测试reshape
    tensor->reshape({2, 2, 3});
    EXPECT_EQ(tensor->shape(), std::vector<int64_t>({2, 2, 3}));
    
    // 测试flatten
    tensor->flatten();
    EXPECT_EQ(tensor->shape(), std::vector<int64_t>({12}));
    
    // 测试unsqueeze
    tensor->unsqueeze(0);
    EXPECT_EQ(tensor->shape(), std::vector<int64_t>({1, 12}));
}

// 切片操作测试
TEST_F(GTensorTest, SliceOperations) {
    auto* tensor = registry.createTensor<float>("test", {4, 4});
    float fill_value = 1.0f;
    tensor->fill(&fill_value);
    
    auto slice = tensor->slice(1, 0, 2);
    auto shape = slice->shape();
    for(int i=0; i<shape.size(); i++)
        std::cerr<< i << ",";
    std::cerr << std::endl;
    EXPECT_EQ(slice->shape(), std::vector<int64_t>({2, 2, 4}));
}

} // namespace test

