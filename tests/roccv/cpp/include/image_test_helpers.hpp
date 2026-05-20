/*
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
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
#include <stdlib.h>

#include <core/detail/allocators/i_allocator.hpp>
#include <core/image.hpp>
#include <core/image_buffer.hpp>
#include <core/image_data.hpp>
#include <core/image_format.hpp>

namespace roccv {
namespace tests {

// Opaque sentinel pointers used by image-layer tests. ImageData / ImageBatchData
// carry pointers but never dereference them — the buffer is a metadata snapshot
// only — so tests use these to verify values flow through without needing real
// allocations.
inline void* const FAKE_PTR_A = reinterpret_cast<void*>(0xAAAAAAAAull);
inline void* const FAKE_PTR_B = reinterpret_cast<void*>(0xBBBBBBBBull);
inline void* const FAKE_PTR_C = reinterpret_cast<void*>(0xCCCCCCCCull);

/**
 * @brief Test allocator that backs every allocation kind with malloc and tallies
 * how many times each entry point is invoked. Pure host-backed; no actual GPU
 * dependency on the returned pointers — callers that exercise the Hip/pinned
 * paths must only inspect metadata, never dereference device memory.
 *
 * `lastAllocBytes` is updated from every alloc path (hip, host, pinned), so
 * callers may assert on the most recent allocation regardless of kind.
 */
class CountingAllocator : public IAllocator {
   public:
    mutable int hipAllocs = 0;
    mutable int hipFrees = 0;
    mutable int hostAllocs = 0;
    mutable int hostFrees = 0;
    mutable int pinnedAllocs = 0;
    mutable int pinnedFrees = 0;
    mutable size_t lastAllocBytes = 0;

    void* allocHipMem(size_t size) const override {
        ++hipAllocs;
        lastAllocBytes = size;
        return std::malloc(size);
    }
    void freeHipMem(void* ptr) const noexcept override {
        ++hipFrees;
        std::free(ptr);
    }

    void* allocHostMem(size_t size, int32_t /*alignment*/ = 0) const override {
        ++hostAllocs;
        lastAllocBytes = size;
        return std::malloc(size);
    }
    void freeHostMem(void* ptr) const noexcept override {
        ++hostFrees;
        std::free(ptr);
    }

    void* allocHostPinnedMem(size_t size) const override {
        ++pinnedAllocs;
        lastAllocBytes = size;
        return std::malloc(size);
    }
    void freeHostPinnedMem(void* ptr) const noexcept override {
        ++pinnedFrees;
        std::free(ptr);
    }
};

// Single-plane packed-row buffer descriptor around `basePtr`. The pointer is
// never dereferenced by the consumers (ImageData / ImageBatchVarShape).
inline ImageBufferStrided MakeSinglePlaneBuffer(int32_t width, int32_t height, int64_t rowStride, void* basePtr) {
    ImageBufferStrided buf{};
    buf.numPlanes = 1;
    buf.planes[0] = {width, height, rowStride, basePtr};
    return buf;
}

// Single-plane GPU-resident ImageData snapshot with packed-row stride implied
// by `fmt`. For tests that need an ImageData but won't touch the pixels.
inline ImageDataStridedHip MakeFakeHipData(int32_t width, int32_t height, void* basePtr, ImageFormat fmt = FMT_RGB8) {
    return ImageDataStridedHip(fmt, MakeSinglePlaneBuffer(width, height, static_cast<int64_t>(width * fmt.channels()),
                                                          basePtr));
}

// Host counterpart of MakeFakeHipData.
inline ImageDataStridedHost MakeFakeHostData(int32_t width, int32_t height, void* basePtr, ImageFormat fmt = FMT_RGB8) {
    return ImageDataStridedHost(fmt, MakeSinglePlaneBuffer(width, height, static_cast<int64_t>(width * fmt.channels()),
                                                           basePtr));
}

// Single-plane GPU-resident Image wrapping a sentinel pointer via ImageWrapData.
// Use for batch tests where pushBack only reads the descriptor.
inline Image MakeFakeGpuImage(int32_t width, int32_t height, void* basePtr, ImageFormat fmt = FMT_RGB8) {
    return ImageWrapData(MakeFakeHipData(width, height, basePtr, fmt));
}

// Host counterpart of MakeFakeGpuImage.
inline Image MakeFakeHostImage(int32_t width, int32_t height, void* basePtr, ImageFormat fmt = FMT_RGB8) {
    return ImageWrapData(MakeFakeHostData(width, height, basePtr, fmt));
}

}  // namespace tests
}  // namespace roccv
