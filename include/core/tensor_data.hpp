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

#include <stdint.h>

#include <optional>

#include "core/data_type.hpp"
#include "core/tensor_buffer.hpp"
#include "core/tensor_shape.hpp"
#include "core/util_enums.h"

namespace roccv {

enum class TensorBufferType {
    TENSOR_BUFFER_NONE,         // Default/invalid buffer type. Used when no buffer type is specified.
    TENSOR_BUFFER_STRIDED_HIP,  // GPU-accessible buffer with strided access.
    TENSOR_BUFFER_STRIDED_HOST  // Host accessible buffer with strided access.
};

/**
 * @brief Holds the underlying tensor data alongside metadata (shape, layout,
 * datatype). Non-strided tensor data is not supported for use right now, use
 * TensorDataStrided to use strided tensor data instead.
 *
 */
class TensorData {
   public:
    TensorData() = delete;
    virtual ~TensorData() = default;

    /**
     * @brief Returns the rank (the number of dimensions) of the tensor data.
     *
     * @return the rank of the tensor
     */
    virtual int rank() const;

    /**
     * @brief Returns the shape of the tensor.
     *
     * @return const TensorShape&
     */
    virtual const TensorShape &shape() const &;

    /**
     * @brief Retrieves a specific dimension size from the tensor shape.
     *
     * @param[in] d The index of the dimension.
     * @return The size of the specified dimension.
     */
    virtual const int64_t shape(int d) const &;

    /**
     * @brief Retrieves the data type of the tensor's elements.
     *
     * @return The data type of the tensor's elements.
     */
    virtual const DataType &dtype() const;

    /**
     * @brief Retrieves the location where the tensor data is allocated, either
     * on the device or the host.
     *
     * @return An enum representing the data location of this tensor data.
     */
    virtual const eDeviceType device() const;

    template <typename Derived>
    std::optional<Derived> cast() const {
        static_assert(std::is_base_of<TensorData, Derived>::value, "Cannot cast TensorData to an unrelated type.");
        static_assert(sizeof(Derived) == sizeof(TensorData), "Derived type must not add any additional data members.");
        return std::optional(Derived(m_shape, m_dtype, m_buffer, m_deviceType));
    }

   protected:
    TensorData(const TensorShape &tshape, const DataType &dtype, const TensorBuffer &buffer,
               const eDeviceType device = eDeviceType::GPU);

    TensorShape m_shape;
    DataType m_dtype;
    eDeviceType m_deviceType;
    TensorBufferType m_bufferType;
    TensorBuffer m_buffer;
};

/**
 * @brief Holds the underlying tensor data alongside tensor metadata. This
 * particular tensor data type is used to store strided data, and contains
 * additional methods for handling strided data.
 *
 */
class TensorDataStrided : public TensorData {
   public:
    using Buffer = TensorBufferStrided;

    TensorDataStrided(const TensorShape &shape, const DataType &dtype, const TensorBuffer &buffer,
                      eDeviceType device = eDeviceType::GPU);

    /**
     * @brief Constructs a TensorDataStrided object.
     *
     * @param[in] tshape The shape of the tensor.
     * @param[in] dtype The datatype of the tensor.
     * @param[in] buffer The buffer containing the tensor's underlying data.
     * @param[in] data_location The data location of the tensor (default:
     * DATA_ON_DEVICE).
     */
    TensorDataStrided(const TensorShape &tshape, const DataType &dtype, const TensorBufferStrided &buffer,
                      const eDeviceType device = eDeviceType::GPU);

    /**
     * @brief Returns the base pointer of the tensor data in memory.
     *
     * @return A pointer to the tensor data in memory.
     */
    void *basePtr() const;

    /**
     * @brief Returns the stride at a given dimension.
     *
     * @param[in] d The specified dimension of the tensor.
     * @return The stride for the given dimension.
     */
    const int64_t stride(int d) const;
};
}  // namespace roccv