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

#include <functional>

#include "core/detail/allocators/i_allocator.hpp"
#include "core/util_enums.h"

namespace roccv {

/**
 * @brief Cleanup function invoked on the raw data pointer when the owning TensorStorage is destroyed. Used to delegate
 * how (and whether) the underlying memory is freed.
 */
using TensorStorageCleanupFunc = std::function<void(void*)>;

/**
 * @brief Stores the underlying data of a tensor and is responsible for freeing of tensor memory when a cleanup function
 * is provided. Agnostic to the tensor's metadata (shape, datatype, etc.)
 *
 */
class TensorStorage {
   public:
    /**
     * @brief Creates a new TensorStorage object wrapping an existing data pointer. Whether the memory is freed on
     * destruction is determined entirely by the provided cleanup function. If no cleanup function is provided, this
     * storage is a non-owning view and the underlying memory is left untouched on destruction.
     *
     * @param data A pointer to existing memory.
     * @param cleanup An optional cleanup function invoked with <data> when this object is destroyed. Defaults to an
     * empty function (non-owning view).
     */
    explicit TensorStorage(void* data, TensorStorageCleanupFunc cleanup = {});

    /**
     * @brief Creates a new TensorStorage object and allocates the requested number of bytes. The allocated memory is
     * owned by this object and freed on destruction.
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

   private:
    void* m_data;
    TensorStorageCleanupFunc m_cleanup;
};

}  // namespace roccv