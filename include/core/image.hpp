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

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <typeinfo>

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
 * allocating constructors and as the output of CalcRequirements; also
 * preserved on the Image itself as the source of truth from which exportData()
 * rebuilds an ImageData snapshot on demand.
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
 *
 * Storage shape: Image holds the buffer pointer (via ImageStorage) plus the
 * "ingredients" describing it (size, format, device, per-plane row strides).
 * It does NOT hold a precomputed ImageData snapshot — exportData() rebuilds
 * one on demand from the ingredients. This keeps a single source of truth for
 * the buffer pointer and aligns with how ImageBatchVarShape produces its
 * own snapshots.
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

    Size2D size() const noexcept { return m_size; }
    ImageFormat format() const noexcept { return m_format; }
    eDeviceType device() const noexcept { return m_device; }

    /**
     * @brief Build and return an ImageData snapshot describing this image.
     *
     * Returned by value (not by reference) — Image stores ingredients, not a
     * precomputed snapshot, so each call constructs a fresh ImageData. The
     * snapshot's plane descriptors point into this Image's buffer; it remains
     * valid as long as any handle to this storage is alive.
     */
    ImageData exportData() const;

    /**
     * @brief Build a snapshot and down-cast it to a specific subclass. Throws
     * std::bad_cast if the underlying buffer kind doesn't match Derived.
     */
    template <typename Derived>
    Derived exportData() const {
        ImageData data = exportData();
        auto derived = data.cast<Derived>();
        if (!derived.has_value()) {
            throw std::bad_cast();
        }
        return derived.value();
    }

   private:
    Image(const Requirements& reqs, eDeviceType device, std::shared_ptr<ImageStorage> storage);

    friend Image ImageWrapData(const ImageData& data, ImageDataCleanupFunc cleanup);

    std::shared_ptr<ImageStorage> m_data;
    Size2D m_size;
    ImageFormat m_format;
    eDeviceType m_device;
    std::array<int64_t, ROCCV_MAX_IMAGE_PLANES> m_planeRowStride;
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
