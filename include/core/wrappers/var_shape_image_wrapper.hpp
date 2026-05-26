/*
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include <hip/hip_runtime.h>

#include <cassert>
#include <cstdint>

#include "core/detail/type_traits.hpp"
#include "core/image_batch_data.hpp"
#include "core/image_buffer.hpp"

namespace roccv {

/**
 * @brief VarShapeImageWrapper is a non-owning, kernel-friendly view over an ImageBatchVarShape's
 * device-side descriptor table. It satisfies the same wrapper concept as ImageWrapper<T>
 * (ValueType, at(n,h,w,c), width(n), height(n), batches(), channels()) so it composes with
 * BorderWrapper / InterpolationWrapper unchanged.
 *
 * Single-plane interleaved (NHWC-style) only — ImageBatchVarShape rejects multi-plane images
 * at pushBack, and channel count is derived from T via detail::NumElements<T>.
 *
 * Pointer residency: m_imageList is a device pointer for an ImageBatchVarShapeDataStridedHip;
 * every at()/width()/height() call dereferences it, so this wrapper is only safe to use from
 * device code (or host code that has the descriptor table host-side).
 *
 * @tparam T The datatype of an individual pixel (e.g. uchar1, uchar3, uchar4, float1, float4).
 */
template <typename T>
class VarShapeImageWrapper {
   public:
    using ValueType = T;
    using BaseType = detail::BaseType<T>;

    VarShapeImageWrapper() = default;

    /**
     * @brief Creates a VarShapeImageWrapper from a GPU-resident varshape batch data snapshot.
     *
     * @param data The exported descriptor table from ImageBatchVarShape::exportData(stream).
     */
    __host__ VarShapeImageWrapper(const ImageBatchVarShapeDataStridedHip& data)
        : m_imageList(data.imageList()), m_numImages(data.numImages()) {
#ifndef NDEBUG
        // ImageBatchVarShape rejects multi-plane at pushBack; assert here as a belt-and-braces
        // check in case a future producer relaxes that.
        const ImageFormat* formats = data.hostFormatList();
        for (int32_t i = 0; i < m_numImages; ++i) {
            assert(formats[i].channels() == detail::NumElements<T> &&
                   "VarShapeImageWrapper<T>: per-image channel count must match NumElements<T>");
        }
#endif
    }

    /**
     * @brief Returns a reference to data at given image-batch coordinates.
     *
     * @param n Batch index.
     * @param h Row index within image n.
     * @param w Column index within image n.
     * @param c Channel index within the pixel.
     * @return A reference to the underlying pixel-channel value.
     */
    __device__ __host__ T& at(int64_t n, int64_t h, int64_t w, int64_t c) { return *doGetPtr(n, h, w, c); }

    __device__ __host__ const T at(int64_t n, int64_t h, int64_t w, int64_t c) const { return *doGetPtr(n, h, w, c); }

    /**
     * @brief Width of the image at batch index n.
     */
    __device__ __host__ inline int64_t width(int64_t n) const { return m_imageList[n].planes[0].width; }

    /**
     * @brief Height of the image at batch index n.
     */
    __device__ __host__ inline int64_t height(int64_t n) const { return m_imageList[n].planes[0].height; }

    /**
     * @brief Number of images in the batch.
     */
    __device__ __host__ inline int64_t batches() const { return m_numImages; }

    /**
     * @brief Number of channels per pixel. Derived from T, identical across all images in v1.
     */
    __device__ __host__ inline int64_t channels() const { return detail::NumElements<T>; }

   private:
    __device__ __host__ inline T* doGetPtr(int64_t n, int64_t h, int64_t w, int64_t c) const {
        // Single-plane interleaved NHWC layout: pixel stride is sizeof(T), channel stride is
        // sizeof(BaseType). Match ImageWrapper<T>::at semantics — returns a T* offset to (h, w)
        // and additionally shifted by c channels.
        const ImagePlaneStrided& p = m_imageList[n].planes[0];
        unsigned char* addr =
            reinterpret_cast<unsigned char*>(p.basePtr) + h * p.rowStride + w * sizeof(T) + c * sizeof(BaseType);
        return reinterpret_cast<T*>(addr);
    }

    const ImageBufferStrided* m_imageList = nullptr;
    int32_t m_numImages = 0;
};

}  // namespace roccv
