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

#include <stdlib.h>

#include "core/detail/allocators/default_allocator.hpp"
#include "core/util_enums.h"

namespace roccv {
/**
 * @brief Stores the underlying data of a tensor and is responsible for allocation/freeing of tensor memory. Agnostic to
 * the tensor's metadata (shape, datatype, etc.)
 *
 */
class TensorStorage {
   public:
    /**
     * @brief Creates a new TensorStorage object and takes ownership of the data pointer.
     *
     * @param data A pointer to allocated memory.
     * @param device The device which the allocated memory is on.
     * @param ownership Whether this object should own <data> (responsible for freeing it once it goes out of scope).
     * Default is eOwnership::OWNING.
     */
    explicit TensorStorage(void* data, eDeviceType device, eOwnership ownership = eOwnership::OWNING);
    explicit TensorStorage(void* data, eDeviceType device, const IAllocator& alloc,
                           eOwnership ownership = eOwnership::OWNING);

    /**
     * @brief Creates a new TensorStorage object and allocates the requested number of bytes.
     *
     * @param bytes Number of bytes to allocate.
     * @param device The device to allocate the memory on.
     */
    explicit TensorStorage(size_t bytes, eDeviceType device);
    explicit TensorStorage(size_t bytes, eDeviceType device, const IAllocator& alloc);

    ~TensorStorage();

    /**
     * @brief Retrieves a raw pointer to the underlying tensor data.
     *
     * @return void*
     */
    void* data() const;

    /**
     * @brief Retrieves the device that the tensor data is allocated on.
     *
     * @return const eDeviceType
     */
    const eDeviceType device() const;

   private:
    eDeviceType m_device;
    eOwnership m_ownership;
    void* m_data;
    const IAllocator& m_allocator;
};

}  // namespace roccv