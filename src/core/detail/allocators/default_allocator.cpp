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

#include <cstdint>

#include "core/exception.hpp"
#include "core/hip_assert.h"
#include "core/utils.hpp"

namespace roccv {
void* DefaultAllocator::allocHostMem(size_t size, int32_t alignment) const {
    if (alignment == 0) {
        // Use malloc when alignment is set to 0
        void* ptr = malloc(size);
        if (ptr == nullptr) throw Exception("Failure when allocating host memory", eStatusType::OUT_OF_MEMORY);
        return ptr;
    }

    // Ensure alignment is a power of 2
    if ((alignment & (alignment - 1)) != 0)
        throw Exception("Specified alignment for host allocation is not a power of 2", eStatusType::INVALID_VALUE);

    // std::aligned_alloc requires the allocation size to be an integral multiple of the alignment.
    size_t alignedSize = detail::AlignUp(size, alignment);

    void* ptr = std::aligned_alloc(alignment, alignedSize);
    if (ptr == nullptr) throw Exception("Failure when allocating host memory", eStatusType::OUT_OF_MEMORY);
    return ptr;
}

void DefaultAllocator::freeHostMem(void* ptr) const noexcept { free(ptr); }

void* DefaultAllocator::allocHostPinnedMem(size_t size) const {
    void* ptr;
    HIP_VALIDATE_NO_ERRORS(hipHostMalloc(&ptr, size));
    return ptr;
}

void DefaultAllocator::freeHostPinnedMem(void* ptr) const noexcept {
    hipError_t status = hipHostFree(ptr);
    if (status != hipSuccess) {
        std::cerr << "Warning: Error when attempting to free pinned memory: " << hipGetErrorName(status) << std::endl;
    }
}

void* DefaultAllocator::allocHipMem(size_t size, int32_t alignment) const {
    void* ptr;
    HIP_VALIDATE_NO_ERRORS(hipMalloc(&ptr, size));

    // hipMalloc does not accept an explicit alignment, but provides a baseline alignment (at least 256 bytes) which
    // satisfies the texture alignment requirements used by the default alignment strategy. If a stricter alignment was
    // requested and the returned pointer does not meet it, surface the error rather than silently handing back
    // misaligned memory.
    if (alignment != 0 && (reinterpret_cast<uintptr_t>(ptr) % static_cast<uintptr_t>(alignment)) != 0) {
        static_cast<void>(hipFree(ptr));
        throw Exception("Unable to satisfy the requested base address alignment for device memory.",
                        eStatusType::INVALID_VALUE);
    }

    return ptr;
}

void DefaultAllocator::freeHipMem(void* ptr) const noexcept {
    hipError_t status = hipFree(ptr);
    if (status != hipSuccess) {
        std::cerr << "Warning: Error when attempting to free device allocated memory: " << hipGetErrorName(status)
                  << std::endl;
    }
}
}  // namespace roccv