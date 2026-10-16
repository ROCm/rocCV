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

/**
 * @brief A Python-facing wrapper for roccv::Tensor, providing DLPack interoperability and Pybind11 integration.
 *
 * The PyTensor class serves as a container around roccv::Tensor, exposing its functionality to Python. It supports
 * construction from shape, data type, layout, and device information, as well as wrapping existing tensors or
 * external DLPack-managed tensors. The class ensures correct lifetime handling for external resources (such as those
 * transferred via DLPack) and enables seamless data movement between devices (CPU/GPU). PyTensor also supplies
 * utilities to export to and import from DLPack capsules, making it suitable for zero-copy interoperability with
 * frameworks like PyTorch, NumPy, or TensorFlow.
 *
 * Usage scenarios include direct creation from Python, import/export to DLPack, device copying, and serving as a
 * type-safe bridge between Python and C++ tensor operations.
 */
class PyTensor : public std::enable_shared_from_this<PyTensor> {
   public:
    /**
     * @brief Constructs a new PyTensor object, as well as the underlying roccv::Tensor which this tensor container will
     * wrap.
     *
     * @param shape The shape of the tensor.
     * @param dtype The data type of the tensor.
     * @param layout The layout of the tensor.
     * @param device The device of the tensor.
     */
    PyTensor(std::vector<int64_t> shape, eDataType dtype, eTensorLayout layout, eDeviceType device);

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
     * @brief Creates a new PyTensor by consuming an external DLPack capsule.
     *
     * This static method constructs a PyTensor that wraps a new roccv::Tensor allocated
     * from the contents of a DLPack capsule (i.e., an object supporting the __dlpack__ protocol),
     * typically exported from other frameworks such as PyTorch, NumPy, or TVM.
     * The shape and datatype are taken from the capsule's DLTensor metadata, while the layout
     * must be specified explicitly, since DLPack does not encode layout.
     *
     * Ownership of the underlying DLPack-managed memory is transferred to the returned PyTensor,
     * such that when the PyTensor is destroyed (and no Python references remain), the deleter
     * from the DLPack capsule is called, ensuring correct cross-framework resource management.
     *
     * @param src   Python object supporting the __dlpack__() method. This can be a capsule returned by
     *              __dlpack__() or any object exposing the DLPack consumer protocol.
     * @param layout The tensor layout to use for the new roccv::Tensor (e.g., NHWC, NCHW, etc.).
     * @return      A std::shared_ptr<PyTensor> wrapping a new tensor that shares memory with the DLPack object.
     *
     * @throws std::runtime_error if the object does not support the DLPack protocol,
     *         if the capsule is invalid or missing, or if conversion fails in any other way.
     *
     * @note The resulting tensor will have the same shape, datatype, device, and (if present) strides
     *       as encoded in the DLPack capsule, but the layout must be provided by the caller.
     */
    static std::shared_ptr<PyTensor> fromDLPack(pybind11::object src, eTensorLayout layout);

    /**
     * @brief Exports this tensor as a DLPack-compatible capsule.
     *
     * This method creates a DLPack DLManagedTensor wrapper for the current PyTensor,
     * encapsulates it in a Python capsule with the "dltensor" name, and returns it.
     * The resulting capsule can be consumed by any framework supporting the DLPack protocol
     * for zero-copy tensor data sharing.
     *
     * The optional @p stream argument is present to satisfy the Pytorch DLPack consumer interface,
     * but is ignored by this implementation currently.
     *
     * @param stream Optional stream pointer (typically unused, present for PyTorch DLPack compliance).
     *
     * @return py::capsule  A Python capsule containing the DLPack DLManagedTensor.
     *
     * @note The caller is responsible for transferring or managing ownership of the data
     * (according to DLPack conventions), including calling the capsule consumer and ensuring
     * that the deleter in DLManagedTensor is invoked only once.
     */
    pybind11::capsule toDLPack(pybind11::object stream);

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
     * @brief Returns the address of the tensor's underlying data buffer as an
     * unsigned integer. For GPU tensors this is a HIP device address; for CPU
     * tensors it is a host address. Use ``device()`` to disambiguate.
     *
     * The pointer is non-owning. The caller is responsible for ensuring this
     * PyTensor remains alive for as long as the pointer is used; otherwise the
     * underlying buffer may be freed and the pointer left dangling.
     *
     * Intended for zero-copy interop with frameworks that accept a raw
     * pointer + shape + dtype (e.g. ``migraphx.argument_from_pointer``).
     *
     * @return uintptr_t
     */
    uintptr_t getDataPtr();

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
    pybind11::tuple getDLDevice();

    /**
     * @brief Exports this class in the provided module.
     *
     * @param m The python module to export this class to.
     */
    static void Export(pybind11::module& m);

    /**
     * @brief Returns a new PyTensor with a reshaped tensor according to the specified shape and layout.
     *
     * Creates and returns a new PyTensor whose underlying tensor is a view or copy of this tensor,
     * with the shape specified by newShape and the layout specified by layout. The number of elements
     * in newShape must match the number of elements in the original tensor.
     *
     * @param newShape The new shape for the tensor.
     * @param layout The new layout to use for the reshaped tensor.
     * @return std::shared_ptr<PyTensor> A new PyTensor with the reshaped tensor.
     *
     * @throws std::runtime_error if the total number of elements does not match.
     */
    std::shared_ptr<PyTensor> reshape(std::vector<int64_t> newShape, eTensorLayout layout);

   private:
    std::shared_ptr<roccv::Tensor> m_tensor;

    // TODO: This DLManagedTensor object is to be deprecated in future versions of DLPack. Ensure that this gets updated
    // before then.
    DLManagedTensor* m_managedTensor = nullptr;
};