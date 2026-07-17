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
#include "core/detail/var_shape_descriptor_table.hpp"
#include "core/image.hpp"
#include "core/image_batch_data.hpp"
#include "core/image_format.hpp"
#include "core/size.hpp"
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
 * Residency is fixed at construction (defaults to GPU). A GPU batch keeps a
 * device descriptor table mirrored by pinned host buffers and the lazy H2D sync
 * above; a CPU batch holds a single host-resident descriptor table with no
 * device buffers, no fence, and no sync (exportData hands the host table
 * straight to host kernels). pushBack rejects images whose device doesn't match
 * the batch's.
 */
class ImageBatchVarShape {
   public:
    using const_iterator = std::vector<Image>::const_iterator;

    /**
     * @brief Construct an empty batch with `capacity` slots on `device`, using
     * the global default allocator.
     */
    explicit ImageBatchVarShape(int32_t capacity, eDeviceType device = eDeviceType::GPU);

    /**
     * @brief Construct an empty batch with `capacity` slots on `device`, using
     * the supplied allocator. The allocator must outlive the batch.
     */
    explicit ImageBatchVarShape(int32_t capacity, const IAllocator &alloc, eDeviceType device = eDeviceType::GPU);

    ~ImageBatchVarShape() = default;

    ImageBatchVarShape(const ImageBatchVarShape &) = delete;
    ImageBatchVarShape &operator=(const ImageBatchVarShape &) = delete;
    ImageBatchVarShape(ImageBatchVarShape &&) noexcept;
    ImageBatchVarShape &operator=(ImageBatchVarShape &&) = delete;

    int32_t capacity() const noexcept { return m_capacity; }
    int32_t numImages() const noexcept { return static_cast<int32_t>(m_images.size()); }

    /**
     * @brief The device the batch (and every image it accepts) resides on.
     */
    eDeviceType device() const noexcept { return m_table.device(); }

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
     * @brief Build (and return by value) a snapshot of the batch, residency
     * matching the batch's device.
     *
     * The concrete returned object is an ImageBatchVarShapeDataStridedHip for a
     * GPU batch or an ImageBatchVarShapeDataStridedHost for a CPU batch; both are
     * returned through the common ImageBatchVarShapeDataStrided base, which
     * carries the device/buffer-kind tag so callers can recover the leaf via
     * cast<>() (see the templated overload). The snapshot is a metadata view
     * valid as long as this batch outlives it.
     *
     * GPU: synchronizes the dirty suffix of the host mirrors to the device
     * descriptor table on `stream` first; `imageList`/`formatList` are device
     * pointers safe for kernels enqueued on the same stream, and `hostFormatList`
     * aliases the pinned host format mirror. CPU: `stream` is unused, no sync
     * occurs, and `imageList`/`formatList`/`hostFormatList` are all host pointers
     * (`formatList` and `hostFormatList` alias).
     */
    ImageBatchVarShapeDataStrided exportData(hipStream_t stream);

    /**
     * @brief Build a snapshot and down-cast it to a specific subclass. Throws
     * std::bad_cast if the underlying buffer kind doesn't match Derived.
     */
    template <typename Derived>
    Derived exportData(hipStream_t stream);

   private:
    void doUpdateCache() const;

    int32_t m_capacity;
    detail::VarShapeDescriptorTable m_table;  // owns the descriptor buffers, fence, and sync.
    std::vector<Image> m_images;

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
    ImageBatchVarShapeDataStrided data = exportData(stream);
    auto derived = data.cast<Derived>();
    if (!derived.has_value()) {
        throw std::bad_cast();
    }
    return derived.value();
}

}  // namespace roccv
