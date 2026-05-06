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

#include <optional>
#include <type_traits>

#include "core/image_batch_buffer.hpp"
#include "core/image_format.hpp"
#include "core/util_enums.h"
#include "operator_types.h"

namespace roccv {

/**
 * @brief Discriminator for the kind of buffer an ImageBatchData carries. Used
 * by IsCompatibleKind() / cast<>() to perform safe runtime down-casting through
 * the ImageBatchData hierarchy.
 *
 * The hierarchy currently exposes only one concrete buffer kind
 * (variable-shape, strided, GPU-resident); the enum is shaped to grow into
 * additional kinds (e.g. tensor-backed batches, host-resident varshape) without
 * breaking the existing buffer kind values.
 */
enum class ImageBatchBufferType {
    IMAGE_BATCH_BUFFER_NONE,                   // Default/invalid buffer type.
    IMAGE_BATCH_VARSHAPE_BUFFER_STRIDED_HIP,   // GPU-accessible varshape descriptor table.
    IMAGE_BATCH_VARSHAPE_BUFFER_STRIDED_HOST,  // Host-accessible varshape descriptor table.
};

/**
 * @brief Holds the underlying image-batch data alongside metadata
 * (numImages, buffer kind). Non-strided batch data is not supported for use
 * right now; use ImageBatchVarShapeDataStrided to access strided varshape data
 * instead.
 *
 * ImageBatchData is the interchange type for a batch of variable-sized images.
 * It does not own any of the underlying buffers (the descriptor table, the
 * format arrays, or the per-image pixel buffers) — it is a metadata snapshot,
 * valid only as long as the producing batch outlives it.
 *
 * Lazy-sync note: for a GPU-resident batch the producer (ImageBatchVarShape)
 * is responsible for ensuring the device-side descriptor table is up to date
 * with any pushBack/popBack edits before handing out an ImageBatchData. The
 * snapshot itself carries no synchronization state.
 */
class ImageBatchData {
   public:
    ImageBatchData() = delete;
    virtual ~ImageBatchData() = default;

    /**
     * @brief Returns the number of images currently in the batch.
     */
    virtual int32_t numImages() const;

    /**
     * @brief Returns the device the descriptor table (and per-image pixel
     * buffers) reside on.
     */
    virtual eDeviceType device() const;

    /**
     * @brief Attempts to down-cast this ImageBatchData to a more specific
     * subclass. Returns the casted value if the underlying buffer kind matches
     * what Derived expects, or std::nullopt otherwise.
     *
     * @tparam Derived The target subclass to cast to.
     */
    template <typename Derived>
    std::optional<Derived> cast() const {
        static_assert(std::is_base_of<ImageBatchData, Derived>::value,
                      "Cannot cast ImageBatchData to an unrelated type.");
        static_assert(sizeof(Derived) == sizeof(ImageBatchData),
                      "Derived type must not add any additional data members.");

        if (!Derived::IsCompatibleKind(m_bufferType)) {
            return std::nullopt;
        }

        return std::make_optional<Derived>(m_numImages, m_buffer);
    }

    static bool IsCompatibleKind(ImageBatchBufferType bufferType);

   protected:
    ImageBatchData(int32_t numImages, const ImageBatchBuffer& buffer);

    int32_t m_numImages;
    eDeviceType m_deviceType;
    ImageBatchBufferType m_bufferType;
    ImageBatchBuffer m_buffer;
};

/**
 * @brief Image-batch data backed by a variable-shape descriptor table. Adds
 * typed accessors for the per-image format arrays and the bounding box across
 * the batch. Sub-classed by ImageBatchVarShapeDataStrided to discriminate
 * pitch-linear storage; further sub-classed by ImageBatchVarShapeDataStridedHip
 * to tag device residency.
 */
class ImageBatchVarShapeData : public ImageBatchData {
   public:
    using Buffer = ImageBatchVarShapeBufferStrided;

    ImageBatchVarShapeData(int32_t numImages, const ImageBatchBuffer& buffer);

    static bool IsCompatibleKind(ImageBatchBufferType bufferType);

    /**
     * @brief Bounding box across all images in the batch, in pixels. Both
     * dimensions are 0 when the batch is empty. Used by operators to size
     * launch grids without iterating the descriptor table.
     */
    Size2D maxSize() const;

    /**
     * @brief Returns the common ImageFormat across all images, or FMT_NONE if
     * formats are heterogeneous or the batch is empty.
     */
    ImageFormat uniqueFormat() const;

    /**
     * @brief Per-image format array. Residency matches the descriptor table
     * (device for GPU batches). Length == numImages().
     *
     * Prefer hostFormatList() for host-side validation paths to avoid a D->H
     * copy.
     */
    const ImageFormat* formatList() const;

    /**
     * @brief Host-resident mirror of formatList(). Always safe to dereference
     * from host code. Length == numImages().
     */
    const ImageFormat* hostFormatList() const;
};

/**
 * @brief Variable-shape image-batch data backed by a pitch-linear descriptor
 * table. Adds the per-image descriptor accessor on top of
 * ImageBatchVarShapeData.
 */
class ImageBatchVarShapeDataStrided : public ImageBatchVarShapeData {
   public:
    using Buffer = ImageBatchVarShapeBufferStrided;

    ImageBatchVarShapeDataStrided(int32_t numImages, const ImageBatchBuffer& buffer);

    static bool IsCompatibleKind(ImageBatchBufferType bufferType);

    /**
     * @brief Per-image descriptor table. Length == numImages(). Residency
     * matches the enclosing data type — for ImageBatchVarShapeDataStridedHip
     * this is a device pointer; kernels read it directly.
     *
     * Each entry is a full ImageBufferStrided so the per-image shape
     * (multi-plane-capable, per-plane stride and base pointer) matches what a
     * single Image carries.
     */
    const ImageBufferStrided* imageList() const;
};

/**
 * @brief GPU-accessible variable-shape image-batch data.
 */
class ImageBatchVarShapeDataStridedHip : public ImageBatchVarShapeDataStrided {
   public:
    using Buffer = ImageBatchVarShapeBufferStrided;

    ImageBatchVarShapeDataStridedHip(int32_t numImages, const ImageBatchBuffer& buffer);

    /**
     * @brief Constructs GPU-accessible varshape image-batch data from the
     * concrete strided buffer directly.
     *
     * @param[in] numImages Number of images currently in the batch.
     * @param[in] buffer    Descriptor table + per-image format arrays. The
     *                      descriptor table and `formatList` must point to GPU
     *                      memory; `hostFormatList` to host memory.
     */
    ImageBatchVarShapeDataStridedHip(int32_t numImages, const Buffer& buffer);

    static bool IsCompatibleKind(ImageBatchBufferType bufferType);
};

/**
 * @brief Host-accessible variable-shape image-batch data.
 *
 * The host-resident counterpart to ImageBatchVarShapeDataStridedHip. The
 * descriptor table, `formatList`, and `hostFormatList` all point to host
 * memory; `formatList` and `hostFormatList` MAY alias the same allocation
 * since no D->H sync is required.
 *
 * The lazy host->device descriptor sync that the GPU producer needs is not
 * applicable here — host-only varshape batches can edit the descriptor table
 * in place and hand it straight to host kernels. The matching producer-side
 * design (whether host batches are a separate type, a runtime-tagged variant
 * of ImageBatchVarShape, or skipped entirely in favor of CPU-side per-image
 * loops) is still open.
 */
class ImageBatchVarShapeDataStridedHost : public ImageBatchVarShapeDataStrided {
   public:
    using Buffer = ImageBatchVarShapeBufferStrided;

    ImageBatchVarShapeDataStridedHost(int32_t numImages, const ImageBatchBuffer& buffer);

    /**
     * @brief Constructs host-accessible varshape image-batch data from the
     * concrete strided buffer directly.
     *
     * @param[in] numImages Number of images currently in the batch.
     * @param[in] buffer    Descriptor table + per-image format arrays. All
     *                      pointers must reference host memory; `formatList`
     *                      and `hostFormatList` may alias.
     */
    ImageBatchVarShapeDataStridedHost(int32_t numImages, const Buffer& buffer);

    static bool IsCompatibleKind(ImageBatchBufferType bufferType);
};

}  // namespace roccv
