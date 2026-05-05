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

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>

#include "core/detail/allocators/i_allocator.hpp"
#include "core/image_buffer.hpp"
#include "core/image_data.hpp"
#include "core/image_format.hpp"
#include "core/util_enums.h"
#include "operator_types.h"

namespace roccv {

class ImageStorage;

/**
 * @brief Cleanup callback signature for ImageWrapData. Invoked when the last
 * Image handle referencing the wrapped buffer is destroyed. Receives the
 * ImageData snapshot that was originally wrapped, so callbacks can free
 * multi-plane buffers or dispatch on format.
 */
using ImageDataCleanupFunc = std::function<void(const ImageData&)>;

/**
 * @brief Per-image allocation spec describing what to allocate for a single
 * variable-sized image. Mirrors NVCVImageRequirements: size, format, per-plane
 * row strides, and base-address alignment. Used as the input to Image's
 * allocating constructors and as the output of CalcRequirements; not stored
 * on the Image instance after construction (m_metadata holds the runtime
 * descriptor in ImageData form).
 *
 * Per-plane row strides are populated only for planes 0..numPlanes(format)-1;
 * remaining slots are unused. Today's interleaved-only ImageFormat means only
 * planeRowStride[0] is populated in practice.
 */
struct ImageRequirements {
    Size2D size;                                     // Width and height in pixels.
    ImageFormat format;                              // Pixel format (dtype + channel count + swizzle).
    int64_t planeRowStride[ROCCV_MAX_IMAGE_PLANES];  // Per-plane row stride in bytes.
    int32_t alignBytes;                              // Required base-address alignment, in bytes.
};

/**
 * @brief A single variable-sized image with device-resident pixel data.
 *
 * Image is the per-element type held by ImageBatchVarShape. It is a handle
 * over a refcounted ImageStorage: copying an Image bumps the refcount and
 * leaves both handles pointing at the same underlying buffer. The buffer is
 * freed when the last handle is destroyed (for owning Images) or when the
 * cleanup callback fires (for ImageWrapData with a callback).
 */
class Image {
   public:
    using Requirements = ImageRequirements;

    /**
     * @brief Compute the requirements (row stride, etc.) for an image of the
     * given dimensions and format.
     */
    static Requirements CalcRequirements(Size2D size, ImageFormat format);

    /**
     * @brief Allocate a new device buffer for an image of the given dimensions
     * and format using the global default allocator.
     */
    explicit Image(Size2D size, ImageFormat format, eDeviceType device = eDeviceType::GPU);

    /**
     * @brief Allocate a new device buffer using a caller-supplied allocator.
     */
    explicit Image(Size2D size, ImageFormat format, const IAllocator& alloc, eDeviceType device = eDeviceType::GPU);

    /**
     * @brief Allocate a new device buffer from precomputed requirements.
     */
    explicit Image(const Requirements& reqs, eDeviceType device = eDeviceType::GPU);
    explicit Image(const Requirements& reqs, const IAllocator& alloc, eDeviceType device = eDeviceType::GPU);

    Image(const Image&) = default;  // refcount bump
    Image(Image&&) noexcept = default;
    Image& operator=(const Image&) = default;  // refcount bump
    Image& operator=(Image&&) noexcept = default;
    ~Image() = default;

    /**
     * @brief Image dimensions in pixels.
     */
    Size2D size() const noexcept;

    /**
     * @brief Pixel format.
     */
    ImageFormat format() const noexcept;

    /**
     * @brief Device the underlying buffer resides on.
     */
    eDeviceType device() const noexcept;

    /**
     * @brief Snapshot of the image's data buffer (pointer, stride, format).
     *
     * The returned ImageData references the same underlying buffer; lifetime
     * is controlled by this Image's refcount, not by the snapshot.
     */
    ImageData exportData() const;

    /**
     * @brief Exports the image's data buffer and casts it to a specified image data object.
     *
     * Throws std::bad_cast if the underlying buffer kind does not match what
     * `Derived` expects (e.g. exportData<ImageDataStridedHip>() on a host-resident
     * image throws std::bad_cast). Convenience wrapper around ImageData::cast<>.
     *
     * @tparam Derived The ImageData subclass to cast to.
     * @return The image data casted to the image data object specified
     */
    template <typename Derived>
    Derived exportData() const {
        ImageData data = exportData();
        std::optional<Derived> derived_data = data.cast<Derived>();
        if (!derived_data.has_value()) {
            throw std::bad_cast();
        }

        return derived_data.value();
    }

   private:
    // Internal ctor used by ImageWrapData and the allocating public ctors via
    // delegation. Stores `metadata` and `storage` verbatim — no allocation.
    Image(ImageData metadata, std::shared_ptr<ImageStorage> storage);

    friend Image ImageWrapData(const ImageData& data, ImageDataCleanupFunc cleanup);

    // m_data is declared first so the allocating ctor can initialize it
    // (allocating the buffer) before m_metadata reads back the pointer.
    std::shared_ptr<ImageStorage> m_data;
    ImageData m_metadata;
};

/**
 * @brief Wrap an externally-owned buffer as an Image without allocating.
 *
 * View-only by default: the wrapped buffer is NOT freed when the returned
 * Image (and any copies) go out of scope. The caller is responsible for
 * keeping the underlying memory alive for as long as any handle survives.
 *
 * Pass a non-null cleanup callback to opt into ownership transfer; the
 * callback runs exactly once, when the last handle is destroyed.
 *
 * @param[in] data Pre-existing image data (pointer, layout, device).
 * @param[in] cleanup Optional callback to free the buffer on last destruction.
 * @return An Image referencing the wrapped buffer.
 */
extern Image ImageWrapData(const ImageData& data, ImageDataCleanupFunc cleanup = nullptr);

}  // namespace roccv
