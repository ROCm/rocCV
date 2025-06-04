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

#pragma once

#include <dlpack/dlpack.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <core/tensor.hpp>
#include <memory>

namespace py = pybind11;

class PyTensor : public std::enable_shared_from_this<PyTensor> {
   public:
    /**
     * @brief Constructs a new PyTensor object, as well as the underlying roccv::Tensor which this tensor container will
     * wrap.
     *
     * @param shape The shape of the tensor.
     * @param layout The layout of the tensor.
     * @param dtype The data type of the tensor.
     * @param device The device of the tensor.
     */
    PyTensor(std::vector<int64_t> shape, eTensorLayout layout, eDataType dtype, eDeviceType device);

    /**
     * @brief Wraps an existing roccv::Tensor inside of a newly constructed PyTensor.
     *
     * @param tensor A shared pointer of the roccv::Tensor to wrap.
     */
    PyTensor(std::shared_ptr<roccv::Tensor> tensor);

    /**
     * @brief Constructs a new tensor wrapper which wraps a roccv::Tensor and a corresponding DLManagedTensor. This is
     * usually called when we consume a DLManagedTensor from another framework. Ensure that the underlying roccv::Tensor
     * has released ownership so that it does not free itself upon destruction. This should be handled by the producer's
     * provided DLManagedTensor.delete function, called by the destructor of this tensor once it falls out of scope
     * (assuming such a deleter exists).
     *
     * @param tensor A shared pointer of the roccv::Tensor to wrap.
     * @param managedTensor The DLManagedTensor provided by a producer.
     */
    PyTensor(std::shared_ptr<roccv::Tensor> tensor, DLManagedTensor* managedTensor);

    /**
     * @brief Destroys the PyTensor object and calls the deleter on the internal DLManagedTensor, assuming it exists and
     * is a valid pointer.
     *
     */
    ~PyTensor();

    /**
     * @brief Creates a copy of this tensor on the specified device.
     *
     * @param device The device of the new tensor.
     * @return std::shared_ptr<PyTensor>
     */
    std::shared_ptr<PyTensor> copyTo(eDeviceType device);

    /**
     * @brief Creates a new tensor given a capsule containing a DLManagedTensor.
     *
     * @param src A capsule containing a DLManagedTensor.
     * @param layout The layout for the new tensor.
     * @return std::shared_ptr<PyTensor>
     */
    static std::shared_ptr<PyTensor> fromDLPack(pybind11::object src, eTensorLayout layout);

    /**
     * @brief Creates a DLManagedTensor contained in a capsule from the tensor.
     *
     * @param stream Optional stream pointer value (used for pytorch).
     *
     * @return py::capsule
     */
    py::capsule toDLPack(py::object stream);

    /**
     * @brief Gets the strides of the tensor as a python list.
     *
     * @return std::vector<int64_t>
     */
    std::vector<int64_t> getStrides();

    /**
     * @brief Gets the shape of the tensor as a python list.
     *
     * @return std::vector<int64_t>
     */
    std::vector<int64_t> getShape();

    /**
     * @brief Gets the number of dimensions of this tensor.
     *
     * @return int
     */
    int getRank();

    /**
     * @brief Gets the data type of this tensor.
     *
     * @return eDataType
     */
    eDataType getDataType();

    /**
     * @brief Gets the layout of this tensor.
     *
     * @return eTensorLayout
     */
    eTensorLayout getLayout();

    /**
     * @brief Gets the device of this tensor.
     *
     * @return eDeviceType
     */
    eDeviceType getDevice();

    /**
     * @brief Gets the underlying roccv::Tensor that this tensor container wraps.
     *
     * @return std::shared_ptr<roccv::Tensor>
     */
    std::shared_ptr<roccv::Tensor> getTensor();

    /**
     * @brief Returns a tuple containing information regarding the DLPack device this tensor uses.
     *
     * @return A python tuple with the first index corresponding to the device type, and the second index corresponding
     * to the device id.
     */
    py::tuple getDLDevice();

    /**
     * @brief Exports this class in the provided module.
     *
     * @param m The python module to export this class to.
     */
    static void Export(py::module& m);

    std::shared_ptr<PyTensor> reshape(std::vector<int64_t> newShape, eTensorLayout layout);

   private:
    std::shared_ptr<roccv::Tensor> m_tensor;

    // TODO: This DLManagedTensor object is to be deprecated in future versions of DLPack. Ensure that this gets updated
    // before then.
    DLManagedTensor* m_managedTensor = nullptr;
};