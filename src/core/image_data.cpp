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

#include "core/image_data.hpp"

#include "core/image_buffer.hpp"
#include "core/image_format.hpp"
#include "core/util_enums.h"

namespace roccv {

const ImageFormat& ImageData::format() const { return m_format; }

eDeviceType ImageData::device() const { return m_deviceType; }

ImageData::ImageData(const ImageFormat& format, const ImageBuffer& buffer)
    : m_format(format),
      m_deviceType(eDeviceType::GPU),
      m_bufferType(ImageBufferType::IMAGE_BUFFER_NONE),
      m_buffer(buffer) {}

bool ImageData::IsCompatibleKind(ImageBufferType bufferType) {
    return bufferType != ImageBufferType::IMAGE_BUFFER_NONE;
}

ImageDataStrided::ImageDataStrided(const ImageFormat& format, const ImageBuffer& buffer)
    : ImageData(format, buffer) {}

bool ImageDataStrided::IsCompatibleKind(ImageBufferType bufferType) {
    return bufferType == ImageBufferType::IMAGE_BUFFER_STRIDED_HIP ||
           bufferType == ImageBufferType::IMAGE_BUFFER_STRIDED_HOST;
}

Size2D ImageDataStrided::size() const {
    const ImagePlaneStrided& p0 = m_buffer.strided.planes[0];
    return Size2D{p0.width, p0.height};
}

int32_t ImageDataStrided::numPlanes() const { return m_buffer.strided.numPlanes; }

const ImagePlaneStrided& ImageDataStrided::plane(int32_t p) const { return m_buffer.strided.planes[p]; }

ImageDataStridedHip::ImageDataStridedHip(const ImageFormat& format, const ImageBuffer& buffer)
    : ImageDataStrided(format, buffer) {
    m_bufferType = ImageBufferType::IMAGE_BUFFER_STRIDED_HIP;
    m_deviceType = eDeviceType::GPU;
}

ImageDataStridedHip::ImageDataStridedHip(const ImageFormat& format, const ImageDataStridedHip::Buffer& buffer)
    : ImageDataStridedHip(format, ImageBuffer{.strided = buffer}) {}

bool ImageDataStridedHip::IsCompatibleKind(ImageBufferType bufferType) {
    return bufferType == ImageBufferType::IMAGE_BUFFER_STRIDED_HIP;
}

ImageDataStridedHost::ImageDataStridedHost(const ImageFormat& format, const ImageBuffer& buffer)
    : ImageDataStrided(format, buffer) {
    m_bufferType = ImageBufferType::IMAGE_BUFFER_STRIDED_HOST;
    m_deviceType = eDeviceType::CPU;
}

ImageDataStridedHost::ImageDataStridedHost(const ImageFormat& format, const ImageDataStridedHost::Buffer& buffer)
    : ImageDataStridedHost(format, ImageBuffer{.strided = buffer}) {}

bool ImageDataStridedHost::IsCompatibleKind(ImageBufferType bufferType) {
    return bufferType == ImageBufferType::IMAGE_BUFFER_STRIDED_HOST;
}

}  // namespace roccv
