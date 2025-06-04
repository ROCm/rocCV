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

#include <memory>

#include "tensor_data.hpp"
#include "tensor_requirements.hpp"
#include "tensor_storage.hpp"

namespace roccv {

class ImageFormat;
struct Size2D;
class TensorShape;
class TensorLayout;

class Tensor {
   public:
    using Requirements = TensorRequirements;

    /**
     * @brief Constructs a Tensor object given a list of requirements. Creating
     * a tensor through this constructor will automatically allocate the
     * required amount of space on either the device or host.
     *
     * @param[in] reqs An object representing the requirements for this tensor.
     */
    explicit Tensor(const TensorRequirements &reqs);

    /**
     * @brief Constructs a Tensor object given a list of requirements and the underlying data as a TensorStorage
     * pointer. This constructor will not automatically allocate data.
     *
     * @param[in] reqs An object representing the requirements for this tensor.
     * @param[in] data A TensorStorage object for the tensor's underlying data.
     */
    explicit Tensor(const TensorRequirements &reqs, std::shared_ptr<TensorStorage> data);

    /**
     * @brief Constructs a tensor object and allocates the appropriate amount of memory on the specified device.
     *
     * @param[in] shape The shape describing the tensor.
     * @param[in] dtype The underlying datatype of the tensor.
     * @param[in] device The device the tensor should be allocated on.
     */
    explicit Tensor(const TensorShape &shape, DataType dtype, const eDeviceType device = eDeviceType::GPU);

    /**
     * @brief Constructs a tensor using image-based requirements and allocates the appropriate amount of memory on the
     * specified device.
     *
     * @param[in] num_images The number of images in the batch.
     * @param[in] image_size The size for images in the batch.
     * @param[in] fmt The format of the underlying image data.
     * @param[in] device The device the tensor should be allocated on.
     */
    explicit Tensor(int num_images, Size2D image_size, ImageFormat fmt, eDeviceType device = eDeviceType::GPU);

    Tensor(const Tensor &other) = delete;
    Tensor(Tensor &&other);

    /**
     * @brief Returns the rank of the tensor (i.e. the number of dimensions)
     *
     * @return An integer representing the rank of the tensor
     */
    int rank() const;

    /**
     * @brief Returns the location (device or host) of the tensor data.
     *
     * @return The location of the tensor data.
     */
    const eDeviceType device() const;

    /**
     * @brief Returns the shape of the tensor
     *
     * @return Shape of the tensor
     */
    const TensorShape &shape() const;

    /**
     * @brief Retrieves a specific dimension size from the tensor shape.
     *
     * @param[in] d The index of the dimension.
     * @return The size of the specified dimension.
     */
    const int64_t shape(int d) const &;

    /**
     * @brief Returns the data type of the tensor
     *
     * @return Data type of the tensor
     */
    const DataType &dtype() const;

    /**
     * @brief Returns the layout of the tensor
     *
     * @return Layout of the tensor
     */
    const TensorLayout &layout() const;

    /**
     * @brief Exports the tensor data of the tensor
     *
     * @return Tensor data of the tensor
     */
    TensorData exportData() const;

    /**
     * @brief Exports tensor data and casts it to a specified tensor data object
     *
     * @tparam The tensor data object to cast this tensor's data to
     * @return The tensor data casted to the tensor data object specified
     */
    template <typename DerivedTensorData>
    DerivedTensorData exportData() const {
        TensorData data = exportData();
        std::optional<DerivedTensorData> derived_tensor = data.cast<DerivedTensorData>();
        if (!derived_tensor.has_value()) {
            throw std::bad_cast();
        }

        return derived_tensor.value();
    }

    /**
     * @brief Creates a view of this tensor with a new shape and layout
     *
     * @param[in] new_shape the new shape of the tensor
     * @return Tensor
     */
    Tensor reshape(const TensorShape &new_shape) const;

    /**
     * @brief Creates a vew of this tensor with a new shape, layout, and data type. The number of bytes allocated must
     * match the original tensor.
     *
     * @param new_shape The new tensor shape.
     * @param new_dtype The new data type of the underlying tensor data.
     * @return Tensor
     */
    Tensor reshape(const TensorShape &new_shape, const DataType &new_dtype) const;

    Tensor &operator=(const Tensor &other);

    /**
     * @brief Calculates tensor requirements. This essentially wraps the
     * provided parameters into a TensorRequirements object.
     *
     * @param[in] shape The desired shape of the tensor.
     * @param[in] dtype The desired data type of the tensor's raw data.
     * @param[in] device The deivce the tensor data should belong to.
     * @return A TensorRequirements object representing this tensor's
     * requirements.
     */
    static Requirements CalcRequirements(const TensorShape &shape, DataType dtype,
                                         const eDeviceType device = eDeviceType::GPU);

    /**
     * @brief Calculates tensor requirements using image-based parameters.
     *
     * @param[in] num_images The number of images in the batch.
     * @param[in] image_size The size for images in the batch.
     * @param[in] fmt The format of the underlying image data.
     * @param[in] device The deivce the tensor data should belong to.
     * @return A TensorRequirements object representing the tensor's requirements.
     */
    static Requirements CalcRequirements(int num_images, Size2D image_size, ImageFormat fmt,
                                         eDeviceType device = eDeviceType::GPU);

   private:
    TensorRequirements m_requirements;      // Tensor metadata
    std::shared_ptr<TensorStorage> m_data;  // Stores raw tensor data
};

/**
 * @brief Wraps TensorData object into a Tensor object.
 *
 * @param[in] data The tensor data to wrap.
 * @return The resulting Tensor with the provided TensorData.
 */
extern Tensor TensorWrapData(const TensorData &tensor_data);

}  // namespace roccv