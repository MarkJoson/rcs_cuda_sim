// test_gtensor_thrust.cpp

#include <gtest/gtest.h>
#include "GTensor.h"
#include "storage/TensorRegistry.h"
#include "storage/TensorRegistryManager.h"
#include <thrust/functional.h>
#include <random>
#include <algorithm>
#include <string>
#include <cstdlib>
#include <memory>

namespace RSG_SIM {
namespace testing {

class GTensorThrustTest : public ::testing::Test {
protected:
    void SetUp() override {
        TensorRegistryManager::initialize();
        registry_ = TensorRegistryManager::createRegistry(1);
    }
    
    void TearDown() override {
        TensorRegistryManager::shutdown();
    }
    
    // 辅助函数：创建随机数据的tensor
    template<typename T>
    GTensor<T>* createRandomTensor(const std::string& name, 
                                 const std::vector<int64_t>& shape,
                                 T min_val = 0,
                                 T max_val = 100) {
        auto* tensor = registry_.createTensor<T>(name, shape);
        
        size_t size = tensor->elemCount();
        std::vector<T> host_data(size);
        
        std::random_device rd;
        std::mt19937 gen(rd());
        
        if constexpr (std::is_integral_v<T>) {
            std::uniform_int_distribution<T> dist(min_val, max_val);
            std::generate(host_data.begin(), host_data.end(),
                [&]() { return dist(gen); });
        } else {
            std::uniform_real_distribution<T> dist(min_val, max_val);
            std::generate(host_data.begin(), host_data.end(),
                [&]() { return dist(gen); });
        }
        
        std::memcpy(tensor->data(), host_data.data(), size * sizeof(T));
        return tensor;
    }
    
    TensorRegistry registry_;
};

// 基本操作测试
TEST_F(GTensorThrustTest, BasicOperations) {
    auto* tensor = createRandomTensor<float>("test", {100});
    
    // 转换到device vector
    auto device_vec = tensor->to_device_vector();
    ASSERT_EQ(device_vec.size(), tensor->elemCount());
    
    // 转换到host vector
    auto host_vec = tensor->to_host_vector();
    ASSERT_EQ(host_vec.size(), tensor->elemCount());
    
    // 比较数据
    std::vector<float> direct_data(tensor->elemCount());
    std::memcpy(direct_data.data(), tensor->data(), tensor->elemCount() * sizeof(float));
    
    for (size_t i = 0; i < tensor->elemCount(); ++i) {
        ASSERT_FLOAT_EQ(host_vec[i], direct_data[i]);
    }
}

// 排序测试
TEST_F(GTensorThrustTest, SortOperations) {
    auto* tensor = createRandomTensor<float>("test", {100});
    
    // 升序排序
    tensor->sort();
    auto host_vec = tensor->to_host_vector();
    ASSERT_TRUE(std::is_sorted(host_vec.begin(), host_vec.end()));
    
    // 降序排序
    tensor->sort_descending();
    host_vec = tensor->to_host_vector();
    ASSERT_TRUE(std::is_sorted(host_vec.begin(), host_vec.end(), 
        std::greater<float>()));
}

// Transform操作测试
TEST_F(GTensorThrustTest, TransformOperations) {
    auto* tensor = createRandomTensor<float>("test", {100});
    auto original_data = tensor->to_host_vector();
    
    // 一元操作
    tensor->transform(thrust::negate<float>());
    auto transformed_data = tensor->to_host_vector();
    
    for (size_t i = 0; i < tensor->elemCount(); ++i) {
        ASSERT_FLOAT_EQ(transformed_data[i], -original_data[i]);
    }
    
    // 二元操作
    auto* tensor2 = createRandomTensor<float>("test2", {100});
    auto original_data2 = tensor2->to_host_vector();
    
    tensor->transform(*tensor2, thrust::plus<float>());
    transformed_data = tensor->to_host_vector();
    
    for (size_t i = 0; i < tensor->elemCount(); ++i) {
        ASSERT_FLOAT_EQ(transformed_data[i], 
            (-original_data[i] + original_data2[i]));
    }
}

// Reduction操作测试
TEST_F(GTensorThrustTest, ReductionOperations) {
    auto* tensor = createRandomTensor<float>("test", {100});
    auto host_vec = tensor->to_host_vector();
    
    // Sum
    float thrust_sum = tensor->sum();
    float std_sum = std::accumulate(host_vec.begin(), host_vec.end(), 0.0f);
    ASSERT_FLOAT_EQ(thrust_sum, std_sum);
    
    // Product
    float thrust_product = tensor->product();
    float std_product = std::accumulate(host_vec.begin(), host_vec.end(), 
        1.0f, std::multiplies<float>());
    ASSERT_FLOAT_EQ(thrust_product, std_product);
}

// Scan操作测试
TEST_F(GTensorThrustTest, ScanOperations) {
    auto* tensor = createRandomTensor<float>("test", {100});
    auto original_data = tensor->to_host_vector();
    
    // Inclusive scan
    tensor->inclusive_scan();
    auto scanned_data = tensor->to_host_vector();
    
    float running_sum = 0.0f;
    for (size_t i = 0; i < tensor->elemCount(); ++i) {
        running_sum += original_data[i];
        ASSERT_FLOAT_EQ(scanned_data[i], running_sum);
    }
    
    // Exclusive scan
    tensor = createRandomTensor<float>("test2", {100});
    original_data = tensor->to_host_vector();
    
    tensor->exclusive_scan();
    scanned_data = tensor->to_host_vector();
    
    running_sum = 0.0f;
    for (size_t i = 0; i < tensor->elemCount(); ++i) {
        ASSERT_FLOAT_EQ(scanned_data[i], running_sum);
        running_sum += original_data[i];
    }
}

// 最大最小值测试
TEST_F(GTensorThrustTest, MinMaxOperations) {
    auto* tensor = createRandomTensor<float>("test", {100});
    auto host_vec = tensor->to_host_vector();
    
    float min_val = tensor->min_element();
    float max_val = tensor->max_element();
    
    ASSERT_FLOAT_EQ(min_val, *std::min_element(host_vec.begin(), host_vec.end()));
    ASSERT_FLOAT_EQ(max_val, *std::max_element(host_vec.begin(), host_vec.end()));
    
    auto [thrust_min, thrust_max] = tensor->minmax_element();
    ASSERT_FLOAT_EQ(thrust_min, min_val);
    ASSERT_FLOAT_EQ(thrust_max, max_val);
}

// 重复元素测试
TEST_F(GTensorThrustTest, DuplicateDetection) {
    // 创建包含重复元素的tensor
    std::vector<float> data = {1, 2, 2, 3, 3, 3, 4, 5, 5};
    auto* tensor = registry_.createTensor<float>("test", {9});
    std::memcpy(tensor->data(), data.data(), data.size() * sizeof(float));
    
    auto duplicates = tensor->find_duplicates();
    std::vector<size_t> expected_duplicates = {2, 3, 4, 8};
    
    ASSERT_EQ(duplicates.size(), expected_duplicates.size());
    for (size_t i = 0; i < duplicates.size(); ++i) {
        ASSERT_EQ(duplicates[i], expected_duplicates[i]);
    }
}

// Unique操作测试
TEST_F(GTensorThrustTest, UniqueOperation) {
    std::vector<float> data = {1, 2, 2, 3, 3, 3, 4, 5, 5};
    auto* tensor = registry_.createTensor<float>("test", {9});
    std::memcpy(tensor->data(), data.data(), data.size() * sizeof(float));
    
    tensor->unique();
    auto result = tensor->to_host_vector();
    std::vector<float> expected = {1, 2, 3, 4, 5};
    
    ASSERT_EQ(result.size(), expected.size());
    for (size_t i = 0; i < result.size(); ++i) {
        ASSERT_FLOAT_EQ(result[i], expected[i]);
    }
}

// 自定义操作测试
TEST_F(GTensorThrustTest, CustomOperations) {
    auto* tensor = createRandomTensor<float>("test", {100});
    auto original_data = tensor->to_host_vector();
    
    // 自定义transform
    tensor->transform([] __device__ (float x) {
        return x * x + 1.0f;
    });
    
    auto transformed_data = tensor->to_host_vector();
    for (size_t i = 0; i < tensor->elemCount(); ++i) {
        ASSERT_FLOAT_EQ(transformed_data[i], 
            original_data[i] * original_data[i] + 1.0f);
    }
}

// 边界情况测试
TEST_F(GTensorThrustTest, EdgeCases) {
    // 空tensor
    auto* empty_tensor = registry_.createTensor<float>("empty", {0});
    ASSERT_EQ(empty_tensor->elemCount(), 0);
    ASSERT_NO_THROW(empty_tensor->sort());
    ASSERT_NO_THROW(empty_tensor->unique());
    
    // 单元素tensor
    auto* single_tensor = registry_.createTensor<float>("single", {1});
    ASSERT_NO_THROW(single_tensor->sort());
    ASSERT_NO_THROW(single_tensor->unique());
    
    // 大size tensor
    auto* large_tensor = createRandomTensor<float>("large", {1000000});
    ASSERT_NO_THROW(large_tensor->sort());
    ASSERT_NO_THROW(large_tensor->sum());
}

// 错误处理测试
TEST_F(GTensorThrustTest, ErrorHandling) {
    auto* tensor1 = createRandomTensor<float>("test1", {100});
    auto* tensor2 = createRandomTensor<float>("test2", {50});
    
    // 形状不匹配的transform
    ASSERT_THROW(tensor1->transform(*tensor2, thrust::plus<float>()), 
        std::runtime_error);
    
    // 无效的device vector转换
    tensor1->toCPU();
    ASSERT_THROW(tensor1->to_device_vector(), std::runtime_error);
}

// 内存泄漏测试
TEST_F(GTensorThrustTest, MemoryLeakCheck) {
    for (int i = 0; i < 1000; ++i) {
        auto* tensor = createRandomTensor<float>(
            "test_" + std::to_string(i), {1000});
        tensor->sort();
        tensor->unique();
        tensor->transform(thrust::negate<float>());
        registry_.removeTensor(tensor->name());
    }
}

} // namespace testing
} // namespace RSG_SIM
