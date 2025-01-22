#include <iostream>
#include <cassert>
#include "core/MessageBus.hh"
#include "core/SimulatorContext.hh"
#include "core/core_types.hh"
#include "core/storage/GTensorConfig.hh"
#include "core/storage/TensorRegistry.hh"

#include "core/console_style.h"

#include "geometry/GeometryManager.cuh"

using namespace cuda_simulator::core;

// Test components
class SourceComponent : public CountableComponent<SourceComponent> {
public:
    SourceComponent(float start = 42) : CountableBase("source"), value(start) {}
    float value;

    void onRegister(SimulatorContext* context) override {
        auto* msg_bus = context->getMessageBus();
        std::vector<int64_t> shape = {1};
        MessageShape msg_shape(shape);
        msg_bus->registerOutput(this, "source_output", msg_shape);
    }

    void onEnvironGroupInit(SimulatorContext*) override {}

    void onExecute(
        SimulatorContext*,
        const NodeExecInputType&,
        NodeExecOutputType& output
    ) override {
        auto& tensor = output.at("source_output");
        tensor = value;
        printf(BG_GREEN "Source Component Sending %f..." FG_DEFAULT LINE_ENDL, value);
        value += 1.0f;
    }

    void onReset(const GTensor&, NodeExecStateType&) override {}
};

class ProcessComponent : public CountableComponent<ProcessComponent> {
public:
    ProcessComponent() : CountableBase("process") {}
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
        SimulatorContext*,
        const NodeExecInputType& input,
        NodeExecOutputType& output
    ) override {
        const auto& in_tensor = input.at("source_output").at(0);
        auto& out_tensor = output.at("process_output");

        std::cout << "Process Component Executing...\n";
        std::cout << "Received value: " << in_tensor << std::endl;

        received_value = in_tensor.item<float>();
        out_tensor = in_tensor;
    }

    void onReset(const GTensor&, NodeExecStateType&) override {}
};

class SinkComponent : public CountableComponent<SinkComponent> {
public:
    SinkComponent() : CountableBase("sink") {}
    float received_value = 0;

    void onRegister(SimulatorContext* context) override {
        auto* msg_bus = context->getMessageBus();
        std::vector<int64_t> shape = {1};
        MessageShape msg_shape(shape);
        msg_bus->registerInput(this, "process_output", msg_shape);
    }

    void onEnvironGroupInit(SimulatorContext*) override {}

    void onExecute(
        SimulatorContext*,
        const NodeExecInputType& input,
        NodeExecOutputType&
    ) override {

        auto in_tensor = input.at("process_output")[0];
        received_value = in_tensor.item<float>();
        std::cout << "Sink Component Executing...\n";
        std::cout << "Received value: " << received_value << std::endl;
        received_value = in_tensor.item<float>() * 2;
        std::cout << "Output value: " << in_tensor.item<float>() << std::endl;

    }

    void onReset(const GTensor&, NodeExecStateType&) override {}
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
class HistoryComponent : public Component {
public:
    HistoryComponent() : Component("history_consumer") {}
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
        SimulatorContext*,
        const NodeExecInputType& input,
        NodeExecOutputType&
    ) override {
        auto in_tensor = input.at("source_output");
        printf(BG_BLUE "History Component Receive:%f ..." FG_DEFAULT LINE_ENDL, in_tensor[0].item<float>());
        received_values.push_back(in_tensor[0].item<float>());
    }

    void onReset(const GTensor&, NodeExecStateType&) override {}
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
    for(int i = 0; i < 5; i++) {
        msg_bus->resetExecuteOrder();
        msg_bus->trigger("default");
    }

    for(size_t i = 0; i < history->received_values.size(); i++) {
        std::cout << "History value " << i << ": " << history->received_values[i] << std::endl;
    }

    std::cout << "Message history test completed!\n";
}

// 测试多输入融合
class FusionComponent : public Component {
public:
    FusionComponent() : Component("fusion") {}
    ~FusionComponent() override = default;
    std::vector<float> received_values;

    void onRegister(SimulatorContext* context) override {
        auto* msg_bus = context->getMessageBus();
        std::vector<int64_t> shape = {1};
        MessageShape msg_shape(shape);
        // 使用STACK方法注册多个输入源
        msg_bus->registerInput(this, "source_output", msg_shape, 0, ReduceMethod::MIN);
    }

    void onEnvironGroupInit(SimulatorContext*) override {}

    void onExecute(
        SimulatorContext*,
        const NodeExecInputType& input,
        NodeExecOutputType&) override {
        auto in_tensor = input.at("source_output");
        for(size_t i = 0; i < in_tensor.size(); i++) {
            received_values.push_back(in_tensor[i].item<float>());
        }
    }

    void onReset(const GTensor&, NodeExecStateType&) override {}
};

void testMultiSourceFusion() {
    std::cout << "\n=== Testing Multi-Source Fusion ===\n";

    auto context = std::make_unique<SimulatorContext>();
    auto* msg_bus = context->getMessageBus();

    auto source1 = std::make_unique<SourceComponent>(42);
    auto source2 = std::make_unique<SourceComponent>(43);
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
    auto& registry = TensorRegistry::getInstance();
    testBasicMessagePassing();
    testMessageHistory();
    testMultiSourceFusion();

    std::cout << "\nAll tests passed successfully!\n";
    // try {
    //     // 使用TensorRegistry替代之前的初始化
    //     auto& registry = TensorRegistry::getInstance();

    //     testBasicMessagePassing();
    //     testMessageHistory();
    //     testMultiSourceFusion();

    //     std::cout << "\nAll tests passed successfully!\n";
    // }
    // catch(const std::exception& e) {
    //     std::cerr << "Test failed with error: " << e.what() << std::endl;
    //     return 1;
    // }
    return 0;
}