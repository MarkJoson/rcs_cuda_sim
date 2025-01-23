#include "core/Component.hh"
#include "core/EnvGroupManager.cuh"
#include "core/ExecuteNode.hh"
#include "core/MessageQueue.hh"
#include "core/MessageBus.hh"

#include "core/SimulatorContext.hh"
#include "core/storage/GTensorConfig.hh"
#include "core/storage/ITensor.hh"
#include "core/storage/Scalar.hh"
#include "core/storage/TensorRegistry.hh"


#include "pybind11/detail/common.h"
#include "torch/extension.h"

#include <memory>

using namespace cuda_simulator::core;

template <typename NodeBase> class PyExecuteNode : public NodeBase {
public:
  using NodeBase::NodeBase;
  void onNodeInit() override {
    PYBIND11_OVERLOAD_PURE(void, NodeBase, onNodeInit);
  }
  void onNodeStart() override {
    PYBIND11_OVERLOAD_PURE(void, NodeBase, onNodeStart);
  }
  void onNodeExecute(const NodeExecInputType &input, NodeExecOutputType &output,
                     NodeExecStateType &state) override {
    PYBIND11_OVERLOAD_PURE(void, NodeBase, onNodeExecute, input, output, state);
  }
  void onNodeReset(const GTensor &reset_flags, NodeExecStateType &state) override {
    PYBIND11_OVERLOAD_PURE(void, NodeBase, onNodeReset, reset_flags, state);
  }

  using NodeBase::addInput;
  using NodeBase::addOutput;
  using NodeBase::addState;

  using NodeBase::input_info_;
  using NodeBase::name_;
  using NodeBase::output_info_;
  using NodeBase::state_info_;
  using NodeBase::tag_;
  ~PyExecuteNode() override = default;
};

template <typename ComponentBase> class PyComponent : public PyExecuteNode<ComponentBase> {
public:
  using PyExecuteNode<ComponentBase>::PyExecuteNode;
  void onEnvironGroupInit() override {
    PYBIND11_OVERLOAD_PURE(void, ComponentBase, onEnvironGroupInit);
  }
  ~PyComponent() override = default;

  using ComponentBase::dependences_;
};

PYBIND11_MODULE(cudasim_pycore, m) {
  m.doc() = "CUDA Simulator Core Module";

  py::enum_<NumericalDataType>(m, "NumericalDataType")
      .value("kFloat32", NumericalDataType::kFloat32)
      .value("kFloat64", NumericalDataType::kFloat64)
      .value("kInt32", NumericalDataType::kInt32)
      .value("kInt64", NumericalDataType::kInt64)
      .value("kUInt8", NumericalDataType::kUInt8)
      .value("kUInt32", NumericalDataType::kUInt32)
      .export_values();

  py::enum_<DeviceType>(m, "DeviceType")
      .value("kCUDA", DeviceType::kCUDA)
      .value("kCPU", DeviceType::kCPU)
      .value("kNumDevices", DeviceType::kNumDevices)
      .export_values();

  py::class_<Scalar>(m, "Scalar")
      .def(py::init<float>())
      .def(py::init<double>())
      .def(py::init<int32_t>())
      .def(py::init<int64_t>())
      .def(py::init<uint8_t>())
      .def(py::init<uint32_t>())
      .def("toFloat", &Scalar::toFloat)
      .def("toDouble", &Scalar::toDouble)
      .def("toInt32", &Scalar::toInt32)
      .def("toInt64", &Scalar::toInt64)
      .def("toUInt8", &Scalar::toUInt8)
      .def("toUInt32", &Scalar::toUInt32);

  py::class_<GTensor>(m, "GTensor")
      .def(py::init<>())
      .def(py::init<const TensorShape &, NumericalDataType, DeviceType>(), py::arg("shape"),
           py::arg("dtype"), py::arg("device_type"))
      .def(py::init<const Scalar &, DeviceType>(), py::arg("scalar"), py::arg("device_type"))
      .def("shape", &GTensor::shape)
      //--
      .def("elemCount", &GTensor::elemCount)
      .def("elemSize", &GTensor::elemSize)
      .def("numel", &GTensor::numel)
      .def("dim", &GTensor::dim)
      .def("dtype", &GTensor::dtype)
      .def("device", &GTensor::device)
      .def("isContiguous", &GTensor::isContiguous)
      //--
      .def("copyFrom", &GTensor::copyFrom, py::arg("other"))
      .def("copyTo", &GTensor::copyTo, py::arg("other"))
      .def("clone", &GTensor::clone)
      //--
      .def_static("zeros", GTensor::zeros, py::arg("shape"),
                  py::arg("dtype") = NumericalDataType::kFloat32,
                  py::arg("device_type") = DeviceType::kCUDA)
      .def_static("rands", GTensor::rands, py::arg("shape"),
                  py::arg("dtype") = NumericalDataType::kFloat32,
                  py::arg("device_type") = DeviceType::kCUDA)
      .def("zerosLike", &GTensor::zerosLike)
      .def("randsLike", &GTensor::randsLike)
      .def("print", static_cast<void (GTensor::*)() const>(&GTensor::print))
      .def("__str__", &GTensor::toString)
      .def("__repr__", &GTensor::toString)
      //--
      .def("zero", &GTensor::zero)
      .def("fill", &GTensor::fill, py::arg("value"))
      //--
      .def("expand", &GTensor::expand, py::arg("new_shape"))
      .def("reshape", &GTensor::reshape, py::arg("shape"))
      .def("squeeze", &GTensor::squeeze, py::arg("dim"))
      .def("unsqueeze", &GTensor::unsqueeze, py::arg("dim"))
      //--
      .def_static("setTensorDefaultDeviceId", GTensor::setTensorDefaultDeviceId,
                  py::arg("device_id"))
      .def_static("setSeed", GTensor::setSeed, py::arg("seed"))
      .def("getTorchTensor", &GTensor::getTorchTensor, py::return_value_policy::reference);

  py::class_<TensorRegistry, std::unique_ptr<TensorRegistry, py::nodelete>>(m, "TensorRegistry")
      .def_static("getInstance", TensorRegistry::getInstance, py::return_value_policy::reference)
      .def("createTensor",
           static_cast<GTensor *(TensorRegistry::*)(const std::string &, const TensorShape &,
                                                    NumericalDataType, DeviceType)>(
               &TensorRegistry::createTensor),
           py::arg("uri"), py::arg("shape"), py::arg("dtype"), py::arg("device"),
           py::return_value_policy::reference)
      .def("getTensor",
           static_cast<GTensor *(TensorRegistry::*)(const std::string &)>(
               &TensorRegistry::getTensor),
           py::return_value_policy::reference, py::arg("uri"), py::return_value_policy::reference)
      .def("removeTensor", &TensorRegistry::removeTensor, py::arg("uri"))
      .def("getTensorsByPrefix", &TensorRegistry::getTensorsByPrefix, py::arg("prefix"),
           py::return_value_policy::reference) // TODO. 检查这里使用ref的合理性
      .def("removeTensorsByPrefix", &TensorRegistry::removeTensorsByPrefix, py::arg("prefix"))
      .def("clear", &TensorRegistry::clear)
      .def("size", &TensorRegistry::size)
      .def("getAllTensorUri", &TensorRegistry::getAllTensorUri);

  py::enum_<ReduceMethod>(m, "ReduceMethod")
      .value("STACK", ReduceMethod::STACK)
      .value("SUM", ReduceMethod::SUM)
      .value("MAX", ReduceMethod::MAX)
      .value("MIN", ReduceMethod::MIN)
      .export_values();

  py::class_<ExecuteNode::NodeInputInfo>(m, "NodeInputInfo")
      .def(py::init([](const MessageName &name, const MessageShape &shape, int history_offset,
                       ReduceMethod reduce_method) {
             return ExecuteNode::NodeInputInfo{name, shape, history_offset, reduce_method};
           }),
           py::arg("name"), py::arg("shape"), py::arg("history_offset"), py::arg("reduce_method"))
      .def_readonly("name", &ExecuteNode::NodeInputInfo::name)
      .def_readonly("shape", &ExecuteNode::NodeInputInfo::shape)
      .def_readwrite("history_offset", &ExecuteNode::NodeInputInfo::history_offset)
      .def_readwrite("reduce_method", &ExecuteNode::NodeInputInfo::reduce_method);

  py::class_<ExecuteNode::NodeOutputInfo>(m, "NodeOutputInfo")
      .def(py::init([](const MessageName &name, const MessageShape &shape, NumericalDataType dtype,
                       std::optional<GTensor> history_padding_val) {
             return ExecuteNode::NodeOutputInfo{name, shape, dtype, history_padding_val};
           }),
           py::arg("name"), py::arg("shape"), py::arg("dtype"), py::arg("history_padding_val"))
      .def_readonly("name", &ExecuteNode::NodeOutputInfo::name)
      .def_readonly("shape", &ExecuteNode::NodeOutputInfo::shape)
      .def_readwrite("dtype", &ExecuteNode::NodeOutputInfo::dtype)
      .def_readwrite("history_padding_val", &ExecuteNode::NodeOutputInfo::history_padding_val);

  py::class_<ExecuteNode::NodeStateInfo>(m, "NodeStateInfo")
      .def(py::init([](const MessageName &name, const MessageShape &shape, NumericalDataType dtype,
                       std::optional<GTensor> init_val) {
             return ExecuteNode::NodeStateInfo{name, shape, dtype, init_val};
           }),
           py::arg("name"), py::arg("shape"), py::arg("dtype"), py::arg("init_val"))
      .def_readonly("name", &ExecuteNode::NodeStateInfo::name)
      .def_readonly("shape", &ExecuteNode::NodeStateInfo::shape)
      .def_readwrite("dtype", &ExecuteNode::NodeStateInfo::dtype)
      .def_readwrite("init_val", &ExecuteNode::NodeStateInfo::init_val);

  py::class_<ExecuteNode, PyExecuteNode<ExecuteNode>>(m, "ExecuteNode")
      .def(py::init<const NodeName &, const NodeTag &>(), py::arg("name"), py::arg("tag"))
      .def("getName", &ExecuteNode::getName)
      .def("getTag", &ExecuteNode::getTag)
      .def("onNodeInit", &ExecuteNode::onNodeInit)
      .def("onNodeStart", &ExecuteNode::onNodeStart)
      .def("onNodeExecute", &ExecuteNode::onNodeExecute, py::arg("input"), py::arg("output"),
           py::arg("state"), py::return_value_policy::reference)
      .def("onNodeReset", &ExecuteNode::onNodeReset, py::arg("reset_flags"), py::arg("state"),
           py::return_value_policy::reference)
      .def("getInputInfo", &ExecuteNode::getInputInfo, py::arg("message_name"),
           py::return_value_policy::reference)
      .def("getOutputInfo", &ExecuteNode::getOutputInfo, py::arg("message_name"),
           py::return_value_policy::reference)
      .def("getStateInfo", &ExecuteNode::getStateInfo, py::arg("state_name"),
           py::return_value_policy::reference)
      .def("getInputs", &ExecuteNode::getInputs)
      .def("getOutputs", &ExecuteNode::getOutputs)
      .def("getStates", &ExecuteNode::getStates)
      .def("addInput", &PyExecuteNode<ExecuteNode>::addOutput, py::arg("input"))
      .def("addOutput", &PyExecuteNode<ExecuteNode>::addOutput, py::arg("output"))
      .def("addState", &PyExecuteNode<ExecuteNode>::addState, py::arg("state"))
      .def_readonly("name_", &PyExecuteNode<ExecuteNode>::name_)
      .def_readonly("tag_", &PyExecuteNode<ExecuteNode>::tag_)
      .def_readonly("input_info_", &PyExecuteNode<ExecuteNode>::input_info_)
      .def_readonly("output_info_", &PyExecuteNode<ExecuteNode>::output_info_)
      .def_readonly("state_info_", &PyExecuteNode<ExecuteNode>::state_info_);

  py::class_<Component, ExecuteNode, PyComponent<Component>>(m, "Component")
      .def(py::init<const NodeName &, const NodeTag &>(), py::arg("name"), py::arg("tag"))
      .def("onEnvironGroupInit", &Component::onEnvironGroupInit)
      .def("addDependence", &Component::addDependence, py::arg("dependence"))
      .def("getDependences", &Component::getDependences, py::return_value_policy::reference)
      .def_readwrite("dependences_", &PyComponent<Component>::dependences_);

  py::class_<TensorItemHandle<float>>(m, "TensorItemHandle")
      .def("groupAt", static_cast<GTensor (TensorItemHandle<float>::*)(int64_t)>(
                              &TensorItemHandle<float>::groupAt<>), py::arg("group_id"))
      .def("activeGroupAt", static_cast<GTensor (TensorItemHandle<float>::*)(int64_t)>(
                              &TensorItemHandle<float>::activeGroupAt<>), py::arg("group_id"));

  py::class_<EnvGroupManager, std::unique_ptr<EnvGroupManager, py::nodelete>>(m, "EnvGroupManager")
      .def("registerConfigTensor", &EnvGroupManager::registerConfigTensor<float>, py::arg("name"),
           py::arg("shape"), py::return_value_policy::reference)
      .def("sampleActiveGroupIndices", &EnvGroupManager::sampleActiveGroupIndices)
      .def("syncToDevice", &EnvGroupManager::syncToDevice)
      .def("getNumGroup", &EnvGroupManager::getNumGroup)
      .def("getNumActiveGroup", &EnvGroupManager::getNumActiveGroup)
      .def("getNumEnvPerGroup", &EnvGroupManager::getNumEnvPerGroup);

  py::class_<MessageQueue, std::unique_ptr<MessageQueue, py::nodelete>>(m, "MessageQueue")
      .def("getHistoryGTensor", &MessageQueue::getHistoryGTensor, py::arg("offset"),
           py::return_value_policy::reference)
      .def("getWriteTensorRef", &MessageQueue::getWriteTensorRef, py::return_value_policy::reference)
      .def("resetEnvData", &MessageQueue::resetEnvData, py::arg("env_group_id"), py::arg("env_id"))
      .def("reset", &MessageQueue::reset);



  py::class_<MessageBus, std::unique_ptr<MessageBus, py::nodelete>>(m, "MessageBus")
      .def(py::init())
      .def("registerComponent", &MessageBus::registerComponent, py::arg("node"))
      .def("getMessageShape", &MessageBus::getMessageShape, py::arg("message_name"),
           py::return_value_policy::reference)
      .def("registerInput", &MessageBus::registerInput, py::arg("component"), py::arg("info"))
      .def("registerOutput", &MessageBus::registerOutput, py::arg("component"), py::arg("info"))
      .def("registerState", &MessageBus::registerState, py::arg("component"), py::arg("info"))
      .def("getMessageQueue", &MessageBus::getMessageQueue, py::arg("node_name"),
           py::arg("message_name"), py::return_value_policy::reference)
      .def("addTrigger", &MessageBus::addTrigger, py::arg("trigger_tag"))
      .def("buildGraph", &MessageBus::buildGraph)
      .def("clearAll", &MessageBus::clearAll)
      .def("trigger", &MessageBus::trigger, py::arg("trigger_tag"))
      .def("resetExecuteOrder", &MessageBus::resetExecuteOrder)
      .def("getCurrentExecuteOrder", &MessageBus::getCurrentExecuteOrder)
      .def("getNodeOrder", &MessageBus::getNodeOrder, py::arg("node_id"));

  // py::class_<SimulatorContext>(m, "SimulatorContext")

  m.def("getTensorRegistry", &TensorRegistry::getInstance, py::return_value_policy::reference);
}