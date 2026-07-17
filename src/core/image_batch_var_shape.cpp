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
#include "core/image_batch_buffer.hpp"
#include "core/image_buffer.hpp"

namespace roccv {

ImageBatchVarShape::ImageBatchVarShape(int32_t capacity, eDeviceType device)
    : ImageBatchVarShape(capacity, GlobalContext().getDefaultAllocator(), device) {}

ImageBatchVarShape::ImageBatchVarShape(int32_t capacity, const IAllocator& alloc, eDeviceType device)
    : m_capacity(capacity), m_table(capacity, device, alloc) {
    m_images.reserve(capacity);
}

ImageBatchVarShape::ImageBatchVarShape(ImageBatchVarShape&& other) noexcept
    : m_capacity(other.m_capacity),
      m_table(std::move(other.m_table)),
      m_images(std::move(other.m_images)),
      m_cacheMaxSize(other.m_cacheMaxSize),
      m_cacheUniqueFormat(other.m_cacheUniqueFormat) {
    other.m_capacity = 0;
    other.m_cacheMaxSize.reset();
    other.m_cacheUniqueFormat.reset();
}

void ImageBatchVarShape::pushBack(const Image& img) {
    const int32_t n = numImages();
    if (n >= m_capacity) {
        throw Exception("ImageBatchVarShape::pushBack would exceed capacity", eStatusType::OUT_OF_BOUNDS);
    }
    if (img.device() != m_table.device()) {
        throw Exception("ImageBatchVarShape only accepts images matching its device", eStatusType::INVALID_VALUE);
    }

    // Export through the strided base so this works for both GPU- and
    // CPU-resident images (a typed cast<...Hip> would reject host images).
    auto strided = img.exportData().cast<ImageDataStrided>();
    if (!strided.has_value()) {
        throw Exception("ImageBatchVarShape requires strided image data", eStatusType::INVALID_VALUE);
    }
    const ImageDataStrided& data = strided.value();
    if (data.numPlanes() != 1) {
        throw Exception("ImageBatchVarShape only supports single-plane images", eStatusType::INVALID_VALUE);
    }

    ImageBufferStrided slot{};
    slot.numPlanes = 1;
    slot.planes[0] = data.plane(0);
    m_table.writeSlot(n, slot, img.format());

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
    m_table.onShrink(numImages());

    // maxSize can only shrink on pop; force a rescan on next query. uniqueFormat
    // stays — it may now be conservatively FMT_NONE, but never wrong.
    m_cacheMaxSize.reset();
    if (numImages() == 0) {
        m_cacheUniqueFormat.reset();
    }
}

void ImageBatchVarShape::clear() {
    m_images.clear();
    m_table.onShrink(0);
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

    const ImageBufferStrided* hostImages = m_table.hostImages();
    const ImageFormat* hostFormats = m_table.hostFormats();

    Size2D maxSz{0, 0};
    ImageFormat unique = hostFormats[0];
    bool heterogeneous = false;
    for (int32_t i = 0; i < n; ++i) {
        const ImagePlaneStrided& p0 = hostImages[i].planes[0];
        maxSz.w = std::max(maxSz.w, p0.width);
        maxSz.h = std::max(maxSz.h, p0.height);
        if (!heterogeneous && hostFormats[i] != unique) {
            heterogeneous = true;
        }
    }
    m_cacheMaxSize = maxSz;
    m_cacheUniqueFormat = heterogeneous ? FMT_NONE : unique;
}

ImageBatchVarShapeDataStrided ImageBatchVarShape::exportData(hipStream_t stream) {
    const auto snap = m_table.sync(stream, numImages());
    doUpdateCache();

    const Size2D maxSz = m_cacheMaxSize.value();
    ImageBatchVarShapeBufferStrided buffer{};
    buffer.uniqueFormat = m_cacheUniqueFormat.value();
    buffer.maxWidth = maxSz.w;
    buffer.maxHeight = maxSz.h;
    buffer.imageList = snap.imageList;
    buffer.formatList = snap.formatList;
    buffer.hostFormatList = snap.hostFormatList;

    if (m_table.device() == eDeviceType::GPU) {
        return ImageBatchVarShapeDataStridedHip(numImages(), buffer);
    }
    return ImageBatchVarShapeDataStridedHost(numImages(), buffer);
}

}  // namespace roccv
