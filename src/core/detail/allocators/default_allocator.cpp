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

#include "core/detail/allocators/default_allocator.hpp"

#include <hip/hip_runtime.h>

#include "core/exception.hpp"
#include "core/hip_assert.h"

namespace roccv {
void* DefaultAllocator::allocHostMem(size_t size, int32_t alignment) {
    void* ptr = aligned_alloc(alignment, size);
    if (ptr == nullptr) {
        throw Exception("Unable to allocate memory on host", eStatusType::OUT_OF_MEMORY);
    }

    return ptr;
}

void DefaultAllocator::freeHostMem(void* ptr) noexcept { free(ptr); }

void* DefaultAllocator::allocHostPinnedMem(size_t size) {
    void* ptr;
    HIP_VALIDATE_NO_ERRORS(hipHostMalloc(&ptr, size));
    return ptr;
}

void DefaultAllocator::freeHostPinnedMem(void* ptr) noexcept { hipHostFree(ptr); }

void* DefaultAllocator::allocHipMem(size_t size) {
    void* ptr;
    HIP_VALIDATE_NO_ERRORS(hipMalloc(&ptr, size));
    return ptr;
}

void DefaultAllocator::freeHipMem(void* ptr) noexcept { hipFree(ptr); }
}  // namespace roccv