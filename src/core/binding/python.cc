#include "core/storage/ITensor.hh"
#include "core/storage/Scalar.hh"
#include "core/storage/TensorRegistry.hh"
#include "pybind11/detail/common.h"
#include "torch/extension.h"

#include "core/storage/GTensorConfig.hh"
#include <memory>

using namespace cuda_simulator::core;

// namespace py = pybind11;

void tensor_add(at::Tensor &a, at::Tensor &b) {
  a.add_(b);
}

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
      .def("device", &GTensor::device)
      .def("elemCount", &GTensor::elemCount)
      .def("elemSize", &GTensor::elemSize)
      .def("dim", &GTensor::dim)
      .def("zero", &GTensor::zero)
      .def("fill", &GTensor::fill, py::arg("value"))
      .def("zerosLike", &GTensor::zerosLike)
      .def("randsLike", &GTensor::randsLike)
      .def("print", static_cast<void (GTensor::*)() const>(&GTensor::print))
      .def("copyFrom", &GTensor::copyFrom, py::arg("other"))
      .def("copyTo", &GTensor::copyTo, py::arg("other"))
      .def("expand", &GTensor::expand, py::arg("new_shape"))
      .def("reshape", &GTensor::reshape, py::arg("shape"))
      .def("squeeze", &GTensor::squeeze, py::arg("dim"))
      .def("unsqueeze", &GTensor::unsqueeze, py::arg("dim"))
      .def("getTorchTensor", &GTensor::getTorchTensor)
      .def_static("zeros", GTensor::zeros, py::arg("shape"),
                  py::arg("dtype") = NumericalDataType::kFloat32,
                  py::arg("device_type") = DeviceType::kCUDA)
      .def_static("rands", GTensor::rands, py::arg("shape"),
                  py::arg("dtype") = NumericalDataType::kFloat32,
                  py::arg("device_type") = DeviceType::kCUDA);

  py::class_<TensorRegistry, std::unique_ptr<TensorRegistry, py::nodelete>>(m, "TensorRegistry")
      .def_static("getInstance", TensorRegistry::getInstance)
      .def("createTensor",
           static_cast<GTensor *(TensorRegistry::*)(const std::string &, const TensorShape &,
                                                    NumericalDataType, DeviceType)>(
               &TensorRegistry::createTensor),
           py::arg("uri"), py::arg("shape"), py::arg("dtype"), py::arg("device"))
      .def("getTensor",
           static_cast<GTensor *(TensorRegistry::*)(const std::string &)>(
               &TensorRegistry::getTensor),
           py::return_value_policy::reference, py::arg("uri"))
      .def("removeTensor", &TensorRegistry::removeTensor, py::arg("uri"));

  m.def("tensor_add", &tensor_add, py::arg("a"), py::arg("b"));
}