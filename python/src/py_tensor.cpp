/**
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include "py_tensor.hpp"

#include <core/hip_assert.h>
#include <dlpack/dlpack.h>

#include "py_helpers.hpp"

DLManagedTensor* createDLManagedTensor(std::shared_ptr<roccv::Tensor> tensor, std::shared_ptr<PyTensor> ctx) {
    auto tensorData = tensor->exportData<roccv::TensorDataStrided>();
    DLManagedTensor* dlTensor = new DLManagedTensor();
    dlTensor->dl_tensor.data = tensorData.basePtr();
    dlTensor->dl_tensor.byte_offset = 0;
    dlTensor->dl_tensor.ndim = tensor->rank();
    dlTensor->dl_tensor.device = RoccvDeviceToDLDevice(tensor->device());
    dlTensor->dl_tensor.dtype = RoccvTypeToDLType(tensor->dtype().etype());

    // Copy shape and stride data into the DLPack tensor
    dlTensor->dl_tensor.strides = new int64_t[tensor->rank()];
    dlTensor->dl_tensor.shape = new int64_t[tensor->rank()];
    for (int i = 0; i < tensor->rank(); i++) {
        dlTensor->dl_tensor.shape[i] = tensor->shape(i);
        // DLTensor strides are element-wise. Convert from byte-wise to element-wise strides.
        dlTensor->dl_tensor.strides[i] = tensorData.stride(i) / tensor->dtype().size();
    }

    // Keep the tensor in the DLTensor's context to ensure its destructor is only called when all references to it fall
    // out of scope.
    dlTensor->manager_ctx = new std::shared_ptr<PyTensor>(ctx);

    dlTensor->deleter = [](DLManagedTensor* mt) {
        delete[] mt->dl_tensor.shape;
        delete[] mt->dl_tensor.strides;
        delete static_cast<std::shared_ptr<PyTensor>*>(mt->manager_ctx);
        delete mt;
    };

    return dlTensor;
}

PyTensor::PyTensor(std::vector<int64_t> shape, eTensorLayout layout, eDataType dtype, eDeviceType device) {
    roccv::TensorShape tShape(roccv::TensorShape(roccv::TensorLayout(layout), shape));
    m_tensor = std::make_shared<roccv::Tensor>(tShape, roccv::DataType(dtype), device);
}

PyTensor::PyTensor(std::shared_ptr<roccv::Tensor> tensor) : m_tensor(tensor) {}

PyTensor::PyTensor(std::shared_ptr<roccv::Tensor> tensor, DLManagedTensor* managedTensor)
    : m_tensor(tensor), m_managedTensor(managedTensor) {}

PyTensor::~PyTensor() {
    // If we are a consumer of a DLManagedTensor, ensure that we call its deleter.
    if (m_managedTensor && m_managedTensor->deleter) {
        m_managedTensor->deleter(m_managedTensor);
    }
}

std::shared_ptr<PyTensor> PyTensor::copyTo(eDeviceType device) {
    std::shared_ptr<roccv::Tensor> copiedTensor =
        std::make_shared<roccv::Tensor>(m_tensor->shape(), m_tensor->dtype(), device);

    auto dstTensorData = copiedTensor->exportData<roccv::TensorDataStrided>();
    auto srcTensorData = m_tensor->exportData<roccv::TensorDataStrided>();
    size_t dataSize = m_tensor->shape().size() * m_tensor->dtype().size();

    const std::map<std::pair<eDeviceType, eDeviceType>, hipMemcpyKind> devicePairToMemcpyKind = {
        {{eDeviceType::CPU, eDeviceType::CPU}, hipMemcpyHostToHost},
        {{eDeviceType::CPU, eDeviceType::GPU}, hipMemcpyHostToDevice},
        {{eDeviceType::GPU, eDeviceType::CPU}, hipMemcpyDeviceToHost},
        {{eDeviceType::GPU, eDeviceType::GPU}, hipMemcpyDeviceToDevice}};

    HIP_VALIDATE_NO_ERRORS(hipMemcpy(dstTensorData.basePtr(), srcTensorData.basePtr(), dataSize,
                                     devicePairToMemcpyKind.at({m_tensor->device(), device})));

    return std::make_shared<PyTensor>(copiedTensor);
}

std::shared_ptr<PyTensor> PyTensor::fromDLPack(pybind11::object src, eTensorLayout layout) {
    if (!py::hasattr(src, "__dlpack__")) {
        throw std::runtime_error("Provided object does not support the DLPack protocol.");
    }

    py::capsule dlpackCapsule = src.attr("__dlpack__")();
    if (!PyCapsule_IsValid(dlpackCapsule.ptr(), "dltensor")) {
        throw std::runtime_error("Invalid DLPack capsule.");
    }
    DLManagedTensor* dlManagedTensor = static_cast<DLManagedTensor*>(dlpackCapsule.get_pointer());
    DLTensor dlTensor = dlManagedTensor->dl_tensor;

    // Mark this capsule as consumed, so that the deleter will not free underlying data.
    dlpackCapsule.set_name("used_dltensor");

    // Copy shape data
    std::vector<int64_t> shapeData(dlTensor.ndim);
    for (int i = 0; i < dlTensor.ndim; i++) {
        shapeData[i] = dlTensor.shape[i];
    }

    // Create a non-owning roccv::Tensor based on the received data
    roccv::TensorShape shape(roccv::TensorLayout(layout), shapeData);
    eDeviceType device = DLDeviceToRoccvDevice(dlTensor.device);
    roccv::TensorRequirements reqs =
        roccv::Tensor::CalcRequirements(shape, roccv::DataType(DLTypeToRoccvType(dlTensor.dtype)), device);
    auto data = std::make_shared<roccv::TensorStorage>(dlTensor.data, device, eOwnership::VIEW);
    auto tensor = std::make_shared<roccv::Tensor>(reqs, data);

    return std::make_shared<PyTensor>(tensor, dlManagedTensor);
}

py::capsule PyTensor::toDLPack(py::object stream) {
    // Stream parameter is intentionally left unused to support Pytorch DLPack device conversions

    DLManagedTensor* dlTensor = createDLManagedTensor(m_tensor, shared_from_this());

    py::capsule capsule(dlTensor, "dltensor", [](PyObject* self) {
        if (PyCapsule_IsValid(self, "used_dltensor")) {
            return;  // Do nothing if the capsule has been consumed
        }

        DLManagedTensor* managed = static_cast<DLManagedTensor*>(PyCapsule_GetPointer(self, "dltensor"));
        if (managed == nullptr) {
            PyErr_WriteUnraisable(self);
            return;
        }

        // The DLManagedTensor deleter can be null if there is no way for the caller to provide a reasonable destructor.
        if (managed->deleter) {
            managed->deleter(managed);
        }
    });

    return capsule;
}

std::vector<int64_t> PyTensor::getStrides() {
    auto tensorData = m_tensor->exportData<roccv::TensorDataStrided>();
    std::vector<int64_t> strides(m_tensor->rank());
    for (int i = 0; i < m_tensor->rank(); i++) {
        strides[i] = tensorData.stride(i);
    }
    return strides;
}

std::vector<int64_t> PyTensor::getShape() {
    std::vector<int64_t> shape(m_tensor->rank());
    for (int i = 0; i < m_tensor->rank(); i++) {
        shape[i] = m_tensor->shape(i);
    }
    return shape;
}

int PyTensor::getRank() { return m_tensor->rank(); }

eDataType PyTensor::getDataType() { return m_tensor->dtype().etype(); }

eTensorLayout PyTensor::getLayout() { return m_tensor->layout().elayout(); }

eDeviceType PyTensor::getDevice() { return m_tensor->device(); }

std::shared_ptr<roccv::Tensor> PyTensor::getTensor() { return m_tensor; }

py::tuple PyTensor::getDLDevice() {
    DLDevice device = RoccvDeviceToDLDevice(m_tensor->device());
    return py::make_tuple(py::int_(static_cast<int>(device.device_type)), py::int_(static_cast<int>(device.device_id)));
}

std::shared_ptr<PyTensor> PyTensor::reshape(std::vector<int64_t> newShape, eTensorLayout layout) {
    roccv::TensorShape newTensorShape(roccv::TensorLayout(layout), newShape);
    auto newTensor = std::make_shared<roccv::Tensor>(m_tensor->reshape(newTensorShape));
    DLManagedTensor* dlTensor = createDLManagedTensor(newTensor, shared_from_this());
    return std::make_shared<PyTensor>(newTensor, dlTensor);
}

void PyTensor::Export(pybind11::module& m) {
    using namespace pybind11::literals;

    pybind11::class_<PyTensor, std::shared_ptr<PyTensor>> tensor(m, "Tensor");
    tensor
        .def(pybind11::init<std::vector<int64_t>, eTensorLayout, eDataType, eDeviceType>(), "shape"_a, "layout"_a,
             "dtype"_a, "device"_a = eDeviceType::GPU, "Constructs a tensor object.")
        .def("copy_to", &PyTensor::copyTo, "device"_a,
             "Returns a deep copy of the tensor with data copied to a specified device type.")
        .def("__dlpack__", &PyTensor::toDLPack, "stream"_a = py::none(),
             "Creates a DLPack compatible tensor from this tensor.")
        .def("strides", &PyTensor::getStrides, "Returns a list representing tensor strides.")
        .def("shape", &PyTensor::getShape, "Returns a list representing the tensor shape.")
        .def("layout", &PyTensor::getLayout, "Returns the layout for this tensor.")
        .def("device", &PyTensor::getDevice, "Returns the device this tensor is on.")
        .def("ndim", &PyTensor::getRank, "Returns the number of dimensions of the tensor.")
        .def("dtype", &PyTensor::getDataType, "Returns the data type of the tensor.")
        .def("__dlpack_device__", &PyTensor::getDLDevice,
             "Returns a tuple containing the DLPack device and device id for the tensor.")
        .def("reshape", &PyTensor::reshape, "new_shape"_a, "layout"_a,
             "Creates a new tensor with the specified shape.");
    m.def("from_dlpack", &PyTensor::fromDLPack, "buffer"_a, "layout"_a,
          "Wraps a DLPack supported tensor in a rocpycv tensor.");
}