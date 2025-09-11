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

#include <span>

#include "tensor_layout.hpp"

namespace roccv {
/**
 * @brief TensorShape class.
 *
 */
class TensorShape {
   public:
    TensorShape() = delete;

    /**
     * @brief Constructs a tensor shape.
     *
     * @param[in] layout The layout type for the tensor shape. The layout type must match with the size
     * of the list given as a shape.
     * @param[in] shape A list describing the dimensions of the tensor.
     */
    TensorShape(const TensorLayout &layout, const std::span<const int64_t> shape);

    /**
     * @brief Constructs a tensor shape.
     *
     * @param[in] layout The layout type for the tensor shape. The layout type must match with the size
     * of the list given as a shape.
     * @param[in] shape An initializer list describing the dimensions of the tensor.
     */
    TensorShape(const TensorLayout &layout, const std::initializer_list<const int64_t> shape);

    /**
     * @brief Constructs a tensor shape.
     *
     * @param[in] shape A list describing the dimensions of the tensor.
     * @param[in] layoutDesc A string describing the layout type of the tensor. The layout type must match with the size
     * of the list given as a shape.
     */
    TensorShape(const std::span<const int64_t> shape, const std::string &layoutDesc);

    /**
     * @brief Constructs a tensor shape.
     *
     * @param[in] shape An initializer list describing the dimensions of the tensor.
     * @param[in] layoutDesc A string describing the layout type of the tensor. The layout type must match with the size
     * of the list given as a shape.
     */
    TensorShape(const std::initializer_list<const int64_t> shape, const std::string &layoutDesc);

    /**
     * @brief Retrieves the layout of the tensor.
     *
     * @return Layout of the tensor.
     */
    const TensorLayout &layout() const;

    /**
     * @brief Returns the size (total number of elements) of the tensor.
     *
     * @return The size of the tensor.
     */
    int size() const;

    // Operators
    int64_t operator[](int32_t i) const;
    TensorShape &operator=(const TensorShape &other);
    bool operator==(const TensorShape &rhs) const;
    bool operator!=(const TensorShape &rhs) const;

   private:
    std::array<int64_t, ROCCV_TENSOR_MAX_RANK> m_shape;
    TensorLayout m_layout;
    size_t m_size;
};
}  // namespace roccv