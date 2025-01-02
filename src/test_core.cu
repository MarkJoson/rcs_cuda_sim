#include <iostream>
#include <cassert>
#include "core/MessageBus.hh"
#include "core/SimulatorContext.hh"
#include "core/storage/GTensor.hh"
#include "core/storage/TensorRegistryManager.hh"

using namespace cuda_simulator::core;

// Test components
class SourceComponent : public ComponentBase {
public:
    SourceComponent() : ComponentBase("source") {}

    void onRegister(SimulatorContext* context) override {
        auto* msg_bus = context->getMessageBus();
        std::vector<int64_t> shape = {1};
        MessageShape msg_shape(shape);
        msg_bus->registerOutput(this, "source_output", msg_shape);
    }

    void onEnvironGroupInit(SimulatorContext*) override {}

    void onExecute(
        SimulatorContext* context,
        const std::unordered_map<MessageNameRef, TensorHandle>& input,
        const std::unordered_map<MessageNameRef, TensorHandle>& output) override {
        auto* tensor = static_cast<GTensor<float>*>(output.at("source_output"));
        printf("source_output: %lx\n", tensor->data());
        tensor->data()[0] = 42.0f;
    }

    void onReset(TensorHandle, std::unordered_map<MessageNameRef, TensorHandle>&) override {}
};

class ProcessComponent : public ComponentBase {
public:
    ProcessComponent() : ComponentBase("process") {}
    float received_value = 0;

    void onRegister(SimulatorContext* context) override {
        auto* msg_bus = context->getMessageBus();
        std::vector<int64_t> shape = {1};
        MessageShape msg_shape(shape);
        msg_bus->registerInput(this, "source_output", msg_shape);
        msg_bus->registerOutput(this, "process_output", msg_shape);
    }

    void onEnvironGroupInit(SimulatorContext*) override {}

    void onExecute(
        SimulatorContext* context,
        const std::unordered_map<MessageNameRef, TensorHandle>& input,
        const std::unordered_map<MessageNameRef, TensorHandle>& output) override {
        auto* in_tensor = static_cast<GTensor<float>*>(input.at("source_output"));
        received_value = in_tensor->data()[0];

        auto* out_tensor = static_cast<GTensor<float>*>(output.at("process_output"));
        out_tensor->data()[0] = received_value * 2;
    }

    void onReset(TensorHandle, std::unordered_map<MessageNameRef, TensorHandle>&) override {}
};

class SinkComponent : public ComponentBase {
public:
    SinkComponent() : ComponentBase("sink") {}
    float received_value = 0;

    void onRegister(SimulatorContext* context) override {
        auto* msg_bus = context->getMessageBus();
        std::vector<int64_t> shape = {1};
        MessageShape msg_shape(shape);
        msg_bus->registerInput(this, "process_output", msg_shape);
    }

    void onEnvironGroupInit(SimulatorContext*) override {}

    void onExecute(
        SimulatorContext* context,
        const std::unordered_map<MessageNameRef, TensorHandle>& input,
        const std::unordered_map<MessageNameRef, TensorHandle>&) override {
        auto* in_tensor = static_cast<GTensor<float>*>(input.at("process_output"));
        received_value = in_tensor->data()[0];
    }

    void onReset(TensorHandle, std::unordered_map<MessageNameRef, TensorHandle>&) override {}
};

void testBasicMessagePassing() {
    std::cout << "\n=== Testing Basic Message Passing ===\n";

    auto context = std::make_unique<SimulatorContext>();
    auto* msg_bus = context->getMessageBus();

    auto source = std::make_unique<SourceComponent>();
    auto process = std::make_unique<ProcessComponent>();
    auto sink = std::make_unique<SinkComponent>();

    msg_bus->registerComponent(source.get());
    msg_bus->registerComponent(process.get());
    msg_bus->registerComponent(sink.get());
    msg_bus->addTrigger("default");
    msg_bus->buildGraph();
    msg_bus->trigger("default");

    assert(process->received_value == 42.0f && "Process should receive value 42.0");
    assert(sink->received_value == 84.0f && "Sink should receive value 84.0");

    std::cout << "Process received: " << process->received_value << std::endl;
    std::cout << "Sink received: " << sink->received_value << std::endl;
    std::cout << "Basic message passing test passed!\n";
}

// 测试带历史记录的消息传递
class HistoryComponent : public ComponentBase {
public:
    HistoryComponent() : ComponentBase("history_consumer") {}
    std::vector<float> received_values;

    void onRegister(SimulatorContext* context) override {
        auto* msg_bus = context->getMessageBus();
        std::vector<int64_t> shape = {1};
        MessageShape msg_shape(shape);
        // 注册一个需要历史消息的输入
        msg_bus->registerInput(this, "source_output", msg_shape, 2);  // 历史长度为2
    }

    void onEnvironGroupInit(SimulatorContext*) override {}

    void onExecute(
        SimulatorContext* context,
        const std::unordered_map<MessageNameRef, TensorHandle>& input,
        const std::unordered_map<MessageNameRef, TensorHandle>&) override {
        auto* in_tensor = static_cast<GTensor<float>*>(input.at("source_output"));
        received_values.push_back(in_tensor->data()[0]);
    }

    void onReset(TensorHandle, std::unordered_map<MessageNameRef, TensorHandle>&) override {}
};

void testMessageHistory() {
    std::cout << "\n=== Testing Message History ===\n";

    auto context = std::make_unique<SimulatorContext>();
    auto* msg_bus = context->getMessageBus();

    auto source = std::make_unique<SourceComponent>();
    auto history = std::make_unique<HistoryComponent>();

    msg_bus->registerComponent(source.get());
    msg_bus->registerComponent(history.get());
    msg_bus->addTrigger("default");
    msg_bus->buildGraph();

    // 触发多次，测试历史记录
    for(int i = 0; i < 3; i++) {
        msg_bus->trigger("default");
    }

    for(size_t i = 0; i < history->received_values.size(); i++) {
        std::cout << "History value " << i << ": " << history->received_values[i] << std::endl;
    }

    std::cout << "Message history test completed!\n";
}

// 测试多输入融合
class FusionComponent : public ComponentBase {
public:
    FusionComponent() : ComponentBase("fusion") {}
    ~FusionComponent() override = default;
    std::vector<float> received_values;

    void onRegister(SimulatorContext* context) override {
        auto* msg_bus = context->getMessageBus();
        std::vector<int64_t> shape = {1};
        MessageShape msg_shape(shape);
        // 使用STACK方法注册多个输入源
        msg_bus->registerInput(this, "source_output", msg_shape, 0, ReduceMethod::STACK);
    }

    void onEnvironGroupInit(SimulatorContext*) override {}

    void onExecute(
        SimulatorContext* context,
        const std::unordered_map<MessageNameRef, TensorHandle>& input,
        const std::unordered_map<MessageNameRef, TensorHandle>&) override {
        auto* in_tensor = static_cast<GTensor<float>*>(input.at("source_output"));
        for(size_t i = 0; i < in_tensor->elemCount(); i++) {
            received_values.push_back(in_tensor->data()[i]);
        }
    }

    void onReset(TensorHandle, std::unordered_map<MessageNameRef, TensorHandle>&) override {}
};

void testMultiSourceFusion() {
    std::cout << "\n=== Testing Multi-Source Fusion ===\n";

    auto context = std::make_unique<SimulatorContext>();
    auto* msg_bus = context->getMessageBus();

    auto source1 = std::make_unique<SourceComponent>();
    auto source2 = std::make_unique<SourceComponent>();
    auto fusion = std::make_unique<FusionComponent>();

    msg_bus->registerComponent(source1.get());
    msg_bus->registerComponent(source2.get());
    msg_bus->registerComponent(fusion.get());
    msg_bus->addTrigger("default");
    msg_bus->buildGraph();

    msg_bus->trigger("default");

    std::cout << "Fusion received values: ";
    for(float val : fusion->received_values) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    std::cout << "Multi-source fusion test completed!\n";
}

int main() {
    try {
        // 初始化tensor系统
        TensorRegistryManager::initialize();

        testBasicMessagePassing();
        testMessageHistory();
        testMultiSourceFusion();

        // 清理tensor系统
        TensorRegistryManager::shutdown();

        std::cout << "\nAll tests passed successfully!\n";
    }
    catch(const std::exception& e) {
        std::cerr << "Test failed with error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}