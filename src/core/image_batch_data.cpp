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

#include "core/image_batch_data.hpp"

#include "core/image_batch_buffer.hpp"
#include "core/image_format.hpp"
#include "core/util_enums.h"

namespace roccv {

int32_t ImageBatchData::numImages() const { return m_numImages; }

eDeviceType ImageBatchData::device() const { return m_deviceType; }

ImageBatchData::ImageBatchData(int32_t numImages, const ImageBatchBuffer& buffer)
    : m_numImages(numImages),
      m_deviceType(eDeviceType::GPU),
      m_bufferType(ImageBatchBufferType::IMAGE_BATCH_BUFFER_NONE),
      m_buffer(buffer) {}

bool ImageBatchData::IsCompatibleKind(ImageBatchBufferType bufferType) {
    return bufferType != ImageBatchBufferType::IMAGE_BATCH_BUFFER_NONE;
}

ImageBatchVarShapeData::ImageBatchVarShapeData(int32_t numImages, const ImageBatchBuffer& buffer)
    : ImageBatchData(numImages, buffer) {}

bool ImageBatchVarShapeData::IsCompatibleKind(ImageBatchBufferType bufferType) {
    return bufferType == ImageBatchBufferType::IMAGE_BATCH_VARSHAPE_BUFFER_STRIDED_HIP ||
           bufferType == ImageBatchBufferType::IMAGE_BATCH_VARSHAPE_BUFFER_STRIDED_HOST;
}

Size2D ImageBatchVarShapeData::maxSize() const {
    return Size2D{m_buffer.varShapeStrided.maxWidth, m_buffer.varShapeStrided.maxHeight};
}

ImageFormat ImageBatchVarShapeData::uniqueFormat() const { return m_buffer.varShapeStrided.uniqueFormat; }

const ImageFormat* ImageBatchVarShapeData::formatList() const { return m_buffer.varShapeStrided.formatList; }

const ImageFormat* ImageBatchVarShapeData::hostFormatList() const { return m_buffer.varShapeStrided.hostFormatList; }

ImageBatchVarShapeDataStrided::ImageBatchVarShapeDataStrided(int32_t numImages, const ImageBatchBuffer& buffer)
    : ImageBatchVarShapeData(numImages, buffer) {}

bool ImageBatchVarShapeDataStrided::IsCompatibleKind(ImageBatchBufferType bufferType) {
    return bufferType == ImageBatchBufferType::IMAGE_BATCH_VARSHAPE_BUFFER_STRIDED_HIP ||
           bufferType == ImageBatchBufferType::IMAGE_BATCH_VARSHAPE_BUFFER_STRIDED_HOST;
}

const ImageBufferStrided* ImageBatchVarShapeDataStrided::imageList() const {
    return m_buffer.varShapeStrided.imageList;
}

ImageBatchVarShapeDataStridedHip::ImageBatchVarShapeDataStridedHip(int32_t numImages, const ImageBatchBuffer& buffer)
    : ImageBatchVarShapeDataStrided(numImages, buffer) {
    m_bufferType = ImageBatchBufferType::IMAGE_BATCH_VARSHAPE_BUFFER_STRIDED_HIP;
    m_deviceType = eDeviceType::GPU;
}

ImageBatchVarShapeDataStridedHip::ImageBatchVarShapeDataStridedHip(
    int32_t numImages, const ImageBatchVarShapeDataStridedHip::Buffer& buffer)
    : ImageBatchVarShapeDataStridedHip(numImages, ImageBatchBuffer{.varShapeStrided = buffer}) {}

bool ImageBatchVarShapeDataStridedHip::IsCompatibleKind(ImageBatchBufferType bufferType) {
    return bufferType == ImageBatchBufferType::IMAGE_BATCH_VARSHAPE_BUFFER_STRIDED_HIP;
}

ImageBatchVarShapeDataStridedHost::ImageBatchVarShapeDataStridedHost(int32_t numImages, const ImageBatchBuffer& buffer)
    : ImageBatchVarShapeDataStrided(numImages, buffer) {
    m_bufferType = ImageBatchBufferType::IMAGE_BATCH_VARSHAPE_BUFFER_STRIDED_HOST;
    m_deviceType = eDeviceType::CPU;
}

ImageBatchVarShapeDataStridedHost::ImageBatchVarShapeDataStridedHost(
    int32_t numImages, const ImageBatchVarShapeDataStridedHost::Buffer& buffer)
    : ImageBatchVarShapeDataStridedHost(numImages, ImageBatchBuffer{.varShapeStrided = buffer}) {}

bool ImageBatchVarShapeDataStridedHost::IsCompatibleKind(ImageBatchBufferType bufferType) {
    return bufferType == ImageBatchBufferType::IMAGE_BATCH_VARSHAPE_BUFFER_STRIDED_HOST;
}

}  // namespace roccv
