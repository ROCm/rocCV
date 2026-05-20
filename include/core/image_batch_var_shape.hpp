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

#include <hip/hip_runtime.h>
#include <stdint.h>

#include <iterator>
#include <optional>
#include <vector>

#include "core/detail/allocators/i_allocator.hpp"
#include "core/image.hpp"
#include "core/image_batch_data.hpp"
#include "core/image_format.hpp"
#include "exception.hpp"
#include "operator_types.h"

namespace roccv {

/**
 * @brief Producer-side container for a batch of variable-sized images that
 * share a single GPU-resident descriptor table.
 *
 * Holds up to `capacity()` Image handles and maintains a parallel descriptor
 * table that operators can dispatch over without iterating Image-by-Image.
 * Capacity is fixed at construction; pushBack/popBack move within it.
 *
 * The host descriptor mirrors are pinned so the H2D copy in exportData() is a
 * true DMA (no runtime bounce buffer) and so the snapshot can expose the same
 * pinned pointer as both `formatList`'s host shadow and `hostFormatList`.
 *
 * Sync model: pushBack/popBack mutate the host mirrors only; the device
 * descriptor table is brought up to date lazily inside exportData(stream),
 * which copies just the dirty suffix `[dirtyStart, numImages)`. A hipEvent
 * (`m_postFence`) guards the host buffers — if a previous exportData's H2D
 * is still in flight, pushBack hipEventSynchronize's on the CPU before
 * mutating, so the snapshot a consumer is reading never tears.
 *
 * GPU-only in v1. CPU-resident images are rejected on push.
 */
class ImageBatchVarShape {
   public:
    using const_iterator = std::vector<Image>::const_iterator;

    /**
     * @brief Construct an empty batch with `capacity` slots, using the global
     * default allocator.
     */
    explicit ImageBatchVarShape(int32_t capacity);

    /**
     * @brief Construct an empty batch with `capacity` slots, using the supplied
     * allocator. The allocator must outlive the batch.
     */
    explicit ImageBatchVarShape(int32_t capacity, const IAllocator &alloc);

    ~ImageBatchVarShape();

    ImageBatchVarShape(const ImageBatchVarShape &) = delete;
    ImageBatchVarShape &operator=(const ImageBatchVarShape &) = delete;
    ImageBatchVarShape(ImageBatchVarShape &&) noexcept;
    ImageBatchVarShape &operator=(ImageBatchVarShape &&) = delete;

    int32_t capacity() const noexcept { return m_capacity; }
    int32_t numImages() const noexcept { return static_cast<int32_t>(m_images.size()); }

    /**
     * @brief Append an image to the batch. Throws if capacity would be
     * exceeded, the image is CPU-resident, or the image has more than one
     * plane (rocCV is single-plane today).
     */
    void pushBack(const Image &img);

    /**
     * @brief Append a range of images. Strong exception guarantee — if any
     * image fails validation, the batch is rolled back to its pre-call state
     * and the exception is rethrown.
     */
    template <typename It>
    void pushBack(It begin, It end);

    /**
     * @brief Remove the trailing `count` images. Throws if `count` exceeds
     * numImages().
     */
    void popBack(int32_t count = 1);

    /**
     * @brief Drop all images. Buffers are kept; the batch is reusable.
     */
    void clear();

    const Image &operator[](int32_t i) const { return m_images[i]; }

    const_iterator begin() const noexcept { return m_images.cbegin(); }
    const_iterator end() const noexcept { return m_images.cend(); }

    /**
     * @brief Bounding box across all images, in pixels. Returns Size2D{0, 0}
     * for an empty batch.
     */
    Size2D maxSize() const;

    /**
     * @brief The common ImageFormat across all images, or FMT_NONE if formats
     * are heterogeneous or the batch is empty. popBack invalidates the cache
     * so the next call rescans and may return an exact format again.
     */
    ImageFormat uniqueFormat() const;

    /**
     * @brief Build (and return by value) a GPU-resident snapshot of the batch.
     *
     * Synchronizes the dirty suffix of the host mirrors to the device
     * descriptor table on the supplied stream before returning. The returned
     * snapshot's `imageList` and `formatList` are device pointers safe for
     * kernels enqueued on the same stream; `hostFormatList` aliases the pinned
     * host format mirror and is safe to read from host code. The snapshot is
     * a metadata view valid as long as this batch outlives it.
     */
    ImageBatchVarShapeDataStridedHip exportData(hipStream_t stream);

    /**
     * @brief Build a snapshot and down-cast it to a specific subclass. Throws
     * std::bad_cast if the underlying buffer kind doesn't match Derived.
     */
    template <typename Derived>
    Derived exportData(hipStream_t stream);

   private:
    void doSyncDirtySuffix(hipStream_t stream);
    void doUpdateCache() const;

    int32_t m_capacity;
    int32_t m_dirtyStartingFromIndex = 0;
    bool m_fencePending = false;

    const IAllocator &m_allocator;
    std::vector<Image> m_images;

    ImageBufferStrided *m_devImagesBuffer = nullptr;
    ImageFormat *m_devFormatsBuffer = nullptr;
    ImageBufferStrided *m_hostImagesBuffer = nullptr;
    ImageFormat *m_hostFormatsBuffer = nullptr;

    hipEvent_t m_postFence = nullptr;

    mutable std::optional<Size2D> m_cacheMaxSize;
    mutable std::optional<ImageFormat> m_cacheUniqueFormat;
};

template <typename It>
void ImageBatchVarShape::pushBack(It begin, It end) {
    const int32_t incoming = static_cast<int32_t>(std::distance(begin, end));
    if (incoming + numImages() > m_capacity) {
        throw Exception("ImageBatchVarShape::pushBack range would exceed capacity", eStatusType::OUT_OF_BOUNDS);
    }

    const int32_t oldNumImages = numImages();
    const auto oldMaxSize = m_cacheMaxSize;
    const auto oldUniqueFormat = m_cacheUniqueFormat;

    try {
        for (auto it = begin; it != end; ++it) {
            pushBack(*it);
        }
    } catch (...) {
        m_images.erase(m_images.begin() + oldNumImages, m_images.end());
        m_cacheMaxSize = oldMaxSize;
        m_cacheUniqueFormat = oldUniqueFormat;
        throw;
    }
}

template <typename Derived>
Derived ImageBatchVarShape::exportData(hipStream_t stream) {
    ImageBatchVarShapeDataStridedHip data = exportData(stream);
    auto derived = data.cast<Derived>();
    if (!derived.has_value()) {
        throw std::bad_cast();
    }
    return derived.value();
}

}  // namespace roccv
