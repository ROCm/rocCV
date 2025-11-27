/*
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#pragma once

#include <stdint.h>

namespace roccv {

/**
 * @class MemAlignment
 * @brief Class for specifying memory alignment constraints for buffer allocations.
 *
 * The MemAlignment class allows you to specify the memory alignment in bytes that should
 * be used when allocating or working with device or host memory buffers. Proper alignment
 * is important for performance reasons and hardware compatibility, especially for GPU operations.
 *
 * There are two types of alignment constraints:
 *  - Base address alignment: Alignment requirements for the starting address of the buffer.
 *  - Row address alignment: Alignment requirements for the starting address of each row, which is
 *    important for multi-dimensional data (e.g., images or tensors).
 *
 * Both alignment values default to 0, which implies that the system or device default alignment will be used.
 *
 * Example usage:
 * @code
 *   roccv::MemAlignment align;
 *   align.baseAddr(256).rowAddr(128);
 * @endcode
 *
 * @see roccv::Tensor
 */
class MemAlignment {
   public:
    MemAlignment() = default;

    /**
     * @brief Returns the base address alignment.
     *
     * @return The base address alignment.
     */
    int32_t baseAddr() const;

    /**
     * @brief Returns the row address alignment.
     *
     * @return The row address alignment.
     */
    int32_t rowAddr() const;

    /**
     * @brief Sets the base address alignment.
     *
     * @param[in] alignment Alignment in bytes.
     * @return A reference to this object, with the base address set.
     */
    MemAlignment& baseAddr(int32_t alignment);

    /**
     * @brief Sets the row address alignment.
     *
     * @param[in] alignment Alignment in bytes.
     * @return A reference to this object, with the row address set.
     */
    MemAlignment& rowAddr(int32_t alignment);

   private:
    int32_t m_baseAddrAlignment = 0;
    int32_t m_rowAddrAlignment = 0;
};
}  // namespace roccv