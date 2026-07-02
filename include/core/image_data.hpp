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

#include "core/image_buffer.hpp"
#include "core/image_format.hpp"
#include "core/util_enums.h"
#include "operator_types.h"

namespace roccv {

/**
 * @brief Discriminator for the kind of buffer an ImageData carries. Used by
 * IsCompatibleKind() / cast<>() to perform safe runtime down-casting through
 * the ImageData hierarchy.
 */
enum class ImageBufferType {
    IMAGE_BUFFER_NONE,         // Default/invalid buffer type. Used when no buffer type is specified.
    IMAGE_BUFFER_STRIDED_HIP,  // GPU-accessible buffer with strided access.
    IMAGE_BUFFER_STRIDED_HOST  // Host-accessible buffer with strided access.
};

/**
 * @brief Holds the underlying image data alongside metadata (format, buffer
 * kind). Non-strided image data is not supported for use right now; use
 * ImageDataStrided to access strided image data instead.
 *
 * ImageData is the interchange type for a single variable-sized image. It
 * does not own the underlying pixel buffer — it is a metadata snapshot, valid
 * only as long as the producing buffer outlives it.
 */
class ImageData {
   public:
    ImageData() = delete;
    virtual ~ImageData() = default;

    /**
     * @brief Returns the pixel format of the image.
     */
    virtual const ImageFormat &format() const;

    /**
     * @brief Returns the device the image data resides on.
     */
    virtual eDeviceType device() const;

    /**
     * @brief Attempts to down-cast this ImageData to a more specific subclass.
     * Returns the casted value if the underlying buffer kind matches what
     * Derived expects, or std::nullopt otherwise.
     *
     * @tparam Derived The target subclass to cast to.
     */
    template <typename Derived>
    std::optional<Derived> cast() const {
        static_assert(std::is_base_of<ImageData, Derived>::value, "Cannot cast ImageData to an unrelated type.");
        static_assert(sizeof(Derived) == sizeof(ImageData), "Derived type must not add any additional data members.");

        if (!Derived::IsCompatibleKind(m_bufferType)) {
            return std::nullopt;
        }

        Derived result(m_format, m_buffer);
        result.m_bufferType = m_bufferType;
        result.m_deviceType = m_deviceType;
        return result;
    }

    static bool IsCompatibleKind(ImageBufferType bufferType);

   protected:
    ImageData(const ImageFormat &format, const ImageBuffer &buffer);

    ImageFormat m_format;
    eDeviceType m_deviceType;
    ImageBufferType m_bufferType;
    ImageBuffer m_buffer;
};

/**
 * @brief Image data backed by one or more pitch-linear planes. Adds typed
 * accessors for plane descriptors on top of the base ImageData. Sub-classed
 * by ImageDataStridedHip and ImageDataStridedHost to discriminate device vs
 * host residency.
 */
class ImageDataStrided : public ImageData {
   public:
    using Buffer = ImageBufferStrided;

    ImageDataStrided(const ImageFormat &format, const ImageBuffer &buffer);

    static bool IsCompatibleKind(ImageBufferType bufferType);

    /**
     * @brief Returns the logical image dimensions, taken from plane 0. For
     * planar formats, individual planes may have smaller dimensions (e.g.
     * chroma sub-sampling); use plane(p) to inspect each plane directly.
     */
    Size2D size() const;

    /**
     * @brief Returns the number of valid planes in the buffer.
     */
    int32_t numPlanes() const;

    /**
     * @brief Returns the descriptor for the requested plane.
     *
     * @param[in] p The plane index. Must satisfy `0 <= p < numPlanes()`.
     */
    const ImagePlaneStrided &plane(int32_t p) const;
};

/**
 * @brief GPU-accessible strided image data.
 */
class ImageDataStridedHip : public ImageDataStrided {
   public:
    using Buffer = ImageBufferStrided;

    ImageDataStridedHip(const ImageFormat &format, const ImageBuffer &buffer);

    /**
     * @brief Constructs GPU-accessible strided image data from a strided
     * image buffer directly.
     *
     * @param[in] format The pixel format.
     * @param[in] buffer A strided image buffer with planes allocated on the GPU.
     */
    ImageDataStridedHip(const ImageFormat &format, const Buffer &buffer);

    static bool IsCompatibleKind(ImageBufferType bufferType);
};

/**
 * @brief Host-accessible strided image data.
 */
class ImageDataStridedHost : public ImageDataStrided {
   public:
    using Buffer = ImageBufferStrided;

    ImageDataStridedHost(const ImageFormat &format, const ImageBuffer &buffer);

    /**
     * @brief Constructs host-accessible strided image data from a strided
     * image buffer directly.
     *
     * @param[in] format The pixel format.
     * @param[in] buffer A strided image buffer with planes allocated on the host.
     */
    ImageDataStridedHost(const ImageFormat &format, const Buffer &buffer);

    static bool IsCompatibleKind(ImageBufferType bufferType);
};

}  // namespace roccv
