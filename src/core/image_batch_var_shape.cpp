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

#include "core/image_batch_var_shape.hpp"

#include <algorithm>

#include "core/detail/context.hpp"
#include "core/exception.hpp"
#include "core/hip_assert.h"
#include "core/image_batch_buffer.hpp"
#include "core/image_buffer.hpp"

namespace roccv {

ImageBatchVarShape::ImageBatchVarShape(int32_t capacity)
    : ImageBatchVarShape(capacity, GlobalContext().getDefaultAllocator()) {}

ImageBatchVarShape::ImageBatchVarShape(int32_t capacity, const IAllocator& alloc)
    : m_capacity(capacity), m_allocator(alloc) {
    if (capacity <= 0) {
        throw Exception("ImageBatchVarShape capacity must be positive", eStatusType::INVALID_VALUE);
    }

    m_images.reserve(capacity);

    const size_t imagesBytes = sizeof(ImageBufferStrided) * capacity;
    const size_t formatsBytes = sizeof(ImageFormat) * capacity;

    try {
        m_devImagesBuffer = static_cast<ImageBufferStrided*>(m_allocator.allocHipMem(imagesBytes));
        m_devFormatsBuffer = static_cast<ImageFormat*>(m_allocator.allocHipMem(formatsBytes));
        m_hostImagesBuffer = static_cast<ImageBufferStrided*>(m_allocator.allocHostPinnedMem(imagesBytes));
        m_hostFormatsBuffer = static_cast<ImageFormat*>(m_allocator.allocHostPinnedMem(formatsBytes));

        HIP_VALIDATE_NO_ERRORS(hipEventCreateWithFlags(&m_postFence, hipEventDisableTiming));
    } catch (...) {
        if (m_hostFormatsBuffer != nullptr) m_allocator.freeHostPinnedMem(m_hostFormatsBuffer);
        if (m_hostImagesBuffer != nullptr) m_allocator.freeHostPinnedMem(m_hostImagesBuffer);
        if (m_devFormatsBuffer != nullptr) m_allocator.freeHipMem(m_devFormatsBuffer);
        if (m_devImagesBuffer != nullptr) m_allocator.freeHipMem(m_devImagesBuffer);
        throw;
    }
}

ImageBatchVarShape::~ImageBatchVarShape() {
    if (m_fencePending && m_postFence != nullptr) {
        // Drain any in-flight H2D copy before freeing the host mirrors it
        // reads from. (void) — destructors must not throw.
        (void)hipEventSynchronize(m_postFence);
    }
    if (m_postFence != nullptr) {
        (void)hipEventDestroy(m_postFence);
    }
    if (m_hostFormatsBuffer != nullptr) m_allocator.freeHostPinnedMem(m_hostFormatsBuffer);
    if (m_hostImagesBuffer != nullptr) m_allocator.freeHostPinnedMem(m_hostImagesBuffer);
    if (m_devFormatsBuffer != nullptr) m_allocator.freeHipMem(m_devFormatsBuffer);
    if (m_devImagesBuffer != nullptr) m_allocator.freeHipMem(m_devImagesBuffer);
}

ImageBatchVarShape::ImageBatchVarShape(ImageBatchVarShape&& other) noexcept
    : m_capacity(other.m_capacity),
      m_dirtyStartingFromIndex(other.m_dirtyStartingFromIndex),
      m_fencePending(other.m_fencePending),
      m_allocator(other.m_allocator),
      m_images(std::move(other.m_images)),
      m_devImagesBuffer(other.m_devImagesBuffer),
      m_devFormatsBuffer(other.m_devFormatsBuffer),
      m_hostImagesBuffer(other.m_hostImagesBuffer),
      m_hostFormatsBuffer(other.m_hostFormatsBuffer),
      m_postFence(other.m_postFence),
      m_cacheMaxSize(other.m_cacheMaxSize),
      m_cacheUniqueFormat(other.m_cacheUniqueFormat) {
    other.m_capacity = 0;
    other.m_dirtyStartingFromIndex = 0;
    other.m_fencePending = false;
    other.m_devImagesBuffer = nullptr;
    other.m_devFormatsBuffer = nullptr;
    other.m_hostImagesBuffer = nullptr;
    other.m_hostFormatsBuffer = nullptr;
    other.m_postFence = nullptr;
    other.m_cacheMaxSize.reset();
    other.m_cacheUniqueFormat.reset();
}

void ImageBatchVarShape::pushBack(const Image& img) {
    const int32_t n = numImages();
    if (n >= m_capacity) {
        throw Exception("ImageBatchVarShape::pushBack would exceed capacity", eStatusType::OUT_OF_BOUNDS);
    }
    if (img.device() != eDeviceType::GPU) {
        throw Exception("ImageBatchVarShape only accepts GPU-resident images", eStatusType::INVALID_VALUE);
    }

    ImageDataStridedHip data = img.exportData<ImageDataStridedHip>();
    if (data.numPlanes() != 1) {
        throw Exception("ImageBatchVarShape only supports single-plane images", eStatusType::INVALID_VALUE);
    }

    if (m_fencePending) {
        HIP_VALIDATE_NO_ERRORS(hipEventSynchronize(m_postFence));
        m_fencePending = false;
    }

    ImageBufferStrided slot{};
    slot.numPlanes = 1;
    slot.planes[0] = data.plane(0);
    m_hostImagesBuffer[n] = slot;
    m_hostFormatsBuffer[n] = img.format();

    const Size2D imgSize = img.size();
    if (n == 0) {
        // Seed from scratch: an empty-batch query may have populated the
        // cache with sentinels (FMT_NONE, 0×0); replacing avoids merging the
        // first real image into them.
        m_cacheMaxSize = imgSize;
        m_cacheUniqueFormat = img.format();
    } else {
        // popBack invalidates m_cacheMaxSize without rescanning, so make sure
        // both halves of the cache are populated before merging in.
        doUpdateCache();
        m_cacheMaxSize->w = std::max(m_cacheMaxSize->w, imgSize.w);
        m_cacheMaxSize->h = std::max(m_cacheMaxSize->h, imgSize.h);
        if (*m_cacheUniqueFormat != img.format()) {
            m_cacheUniqueFormat = FMT_NONE;
        }
    }

    m_images.push_back(img);
}

void ImageBatchVarShape::popBack(int32_t count) {
    if (count < 0) {
        throw Exception("ImageBatchVarShape::popBack count must be non-negative", eStatusType::INVALID_VALUE);
    }
    if (count > numImages()) {
        throw Exception("ImageBatchVarShape::popBack count exceeds numImages", eStatusType::OUT_OF_BOUNDS);
    }

    m_images.erase(m_images.end() - count, m_images.end());
    m_dirtyStartingFromIndex = std::min(m_dirtyStartingFromIndex, numImages());

    // maxSize can only shrink on pop; force a rescan on next query. uniqueFormat
    // stays — it may now be conservatively FMT_NONE, but never wrong.
    m_cacheMaxSize.reset();
    if (numImages() == 0) {
        m_cacheUniqueFormat.reset();
    }
}

void ImageBatchVarShape::clear() {
    m_images.clear();
    m_dirtyStartingFromIndex = 0;
    m_cacheMaxSize.reset();
    m_cacheUniqueFormat.reset();
}

Size2D ImageBatchVarShape::maxSize() const {
    doUpdateCache();
    return m_cacheMaxSize.value_or(Size2D{0, 0});
}

ImageFormat ImageBatchVarShape::uniqueFormat() const {
    doUpdateCache();
    return m_cacheUniqueFormat.value_or(FMT_NONE);
}

void ImageBatchVarShape::doUpdateCache() const {
    if (m_cacheMaxSize.has_value() && m_cacheUniqueFormat.has_value()) {
        return;
    }
    const int32_t n = static_cast<int32_t>(m_images.size());
    if (n == 0) {
        m_cacheMaxSize = Size2D{0, 0};
        m_cacheUniqueFormat = FMT_NONE;
        return;
    }

    Size2D maxSz{0, 0};
    ImageFormat unique = m_hostFormatsBuffer[0];
    bool heterogeneous = false;
    for (int32_t i = 0; i < n; ++i) {
        const ImagePlaneStrided& p0 = m_hostImagesBuffer[i].planes[0];
        maxSz.w = std::max(maxSz.w, p0.width);
        maxSz.h = std::max(maxSz.h, p0.height);
        if (!heterogeneous && m_hostFormatsBuffer[i] != unique) {
            heterogeneous = true;
        }
    }
    m_cacheMaxSize = maxSz;
    m_cacheUniqueFormat = heterogeneous ? FMT_NONE : unique;
}

void ImageBatchVarShape::doSyncDirtySuffix(hipStream_t stream) {
    const int32_t n = numImages();
    if (m_dirtyStartingFromIndex >= n) {
        return;
    }
    const int32_t dirtyCount = n - m_dirtyStartingFromIndex;

    if (m_fencePending) {
        HIP_VALIDATE_NO_ERRORS(hipStreamWaitEvent(stream, m_postFence, /*flags=*/0));
    }

    HIP_VALIDATE_NO_ERRORS(hipMemcpyAsync(m_devImagesBuffer + m_dirtyStartingFromIndex,
                                          m_hostImagesBuffer + m_dirtyStartingFromIndex,
                                          sizeof(ImageBufferStrided) * dirtyCount, hipMemcpyHostToDevice, stream));
    HIP_VALIDATE_NO_ERRORS(hipMemcpyAsync(m_devFormatsBuffer + m_dirtyStartingFromIndex,
                                          m_hostFormatsBuffer + m_dirtyStartingFromIndex,
                                          sizeof(ImageFormat) * dirtyCount, hipMemcpyHostToDevice, stream));

    HIP_VALIDATE_NO_ERRORS(hipEventRecord(m_postFence, stream));
    m_fencePending = true;
    m_dirtyStartingFromIndex = n;
}

ImageBatchVarShapeDataStridedHip ImageBatchVarShape::exportData(hipStream_t stream) {
    doSyncDirtySuffix(stream);
    doUpdateCache();

    const Size2D maxSz = m_cacheMaxSize.value();
    ImageBatchVarShapeBufferStrided buffer{};
    buffer.uniqueFormat = m_cacheUniqueFormat.value();
    buffer.maxWidth = maxSz.w;
    buffer.maxHeight = maxSz.h;
    buffer.formatList = m_devFormatsBuffer;
    buffer.hostFormatList = m_hostFormatsBuffer;
    buffer.imageList = m_devImagesBuffer;

    return ImageBatchVarShapeDataStridedHip(numImages(), buffer);
}

}  // namespace roccv
