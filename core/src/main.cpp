#include "storage.h"
#include <iostream>

class GTensorTest {

public:
    void SetUp() {
        registry = RSG_SIM::TensorRegistryManager::createRegistry(2); // 使用小的env_count便于测试
    }

    void test() {
            auto* tensor = registry.createTensor<float>("test", {4, 4});
        float fill_value = 1.0f;
        tensor->fill(&fill_value);
        
        auto slice = tensor->slice(1, 0, 2);
        auto shape = tensor->shape();
        for(int i=0; i<shape.size(); i++)
            std::cerr<< shape[i] << ",";
        std::cerr << std::endl;
    }

    RSG_SIM::TensorRegistry registry;
};

int main(int argc, char** args)
{
    GTensorTest a;
    a.SetUp();
    a.test();

}