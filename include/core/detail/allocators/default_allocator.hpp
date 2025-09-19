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

#include "core/detail/allocators/i_allocator.hpp"

namespace roccv {
class DefaultAllocator final : public IAllocator {
   public:
    /**
     * @brief Allocate memory on the host with a specified alignment.
     *
     * @param size The size in bytes of the allocation.
     * @param alignment The memory alignment of the allocation.
     * @return A pointer to the allocated memory.
     */
    void* allocHostMem(size_t size, int32_t alignment = 0) const override;

    /**
     * @brief Frees memory allocated on the host.
     *
     * @param ptr A pointer to allocate memory.
     */
    void freeHostMem(void* ptr) const noexcept override;

    /**
     * @brief Allocates pinned host memory.
     *
     * @param size The size in bytes of the allocation.
     * @return A pointer to the allocated pinned host memory.
     */
    void* allocHostPinnedMem(size_t size) const override;

    /**
     * @brief Frees pinned host memory.
     *
     * @param ptr A pointer to pinned host memory.
     */
    void freeHostPinnedMem(void* ptr) const noexcept override;

    /**
     * @brief Allocates memory on the device.
     *
     * @param size The size in bytes to allocate.
     * @return A pointer to allocated device memory.
     */
    void* allocHipMem(size_t size) const override;

    /**
     * @brief Free device allocated memory.
     *
     * @param ptr A pointer to device allocated memory.
     */
    void freeHipMem(void* ptr) const noexcept override;
};
}  // namespace roccv