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

#include "core/image.hpp"

#include "core/data_type.hpp"
#include "core/detail/context.hpp"
#include "core/exception.hpp"
#include "core/image_storage.hpp"

namespace roccv {

namespace {

// Allocates a buffer through `alloc` for the requested device and wraps it
// in an ImageStorage whose shared_ptr deleter frees through the same allocator.
// The allocator reference is captured by reference; callers must ensure it
// outlives every Image (and any handle copied from it) it creates.
std::shared_ptr<ImageStorage> makeStorage(const ImageRequirements& reqs, const IAllocator& alloc, eDeviceType device) {
    const size_t bytes = static_cast<size_t>(reqs.planeRowStride[0]) * reqs.size.h;

    void* buf = nullptr;
    switch (device) {
        case eDeviceType::GPU:
            buf = alloc.allocHipMem(bytes);
            break;
        case eDeviceType::CPU:
            buf = alloc.allocHostMem(bytes);
            break;
    }

    return std::shared_ptr<ImageStorage>(new ImageStorage(buf), [&alloc, device](ImageStorage* s) {
        switch (device) {
            case eDeviceType::GPU:
                alloc.freeHipMem(s->data());
                break;
            case eDeviceType::CPU:
                alloc.freeHostMem(s->data());
                break;
        }
        delete s;
    });
}

// Builds the canonical ImageData stored on Image from a freshly-allocated
// (or wrapped) buffer plus its layout description. Single-plane today —
// ImageFormat is interleaved-only, so only planes[0] is populated.
ImageData makeImageData(const ImageRequirements& reqs, void* buf, eDeviceType device) {
    ImageBufferStrided strided{};
    strided.numPlanes = 1;
    strided.planes[0].width = reqs.size.w;
    strided.planes[0].height = reqs.size.h;
    strided.planes[0].rowStride = reqs.planeRowStride[0];
    strided.planes[0].basePtr = buf;

    switch (device) {
        case eDeviceType::GPU:
            return ImageDataStridedHip(reqs.format, strided);
        case eDeviceType::CPU:
            return ImageDataStridedHost(reqs.format, strided);
    }

    throw Exception("Unsupported device type in Image::makeImageData.", eStatusType::INVALID_VALUE);
}

}  // namespace

// -----------------------------------------------------------------------------
// CalcRequirements
// -----------------------------------------------------------------------------

Image::Requirements Image::CalcRequirements(Size2D size, ImageFormat format) {
    if (size.w < 1 || size.h < 1) {
        throw Exception("Image dimensions must be >= 1.", eStatusType::INVALID_VALUE);
    }

    ImageRequirements reqs;
    reqs.size = size;
    reqs.format = format;

    const int64_t bytesPerPixel = static_cast<int64_t>(DataType(format.dtype()).size()) * format.channels();
    reqs.planeRowStride[0] = bytesPerPixel * size.w;  // packed; no row padding while alignBytes is unused.

    // TODO: derive a sensible default base/row alignment from device attributes.
    reqs.alignBytes = 0;

    return reqs;
}

// -----------------------------------------------------------------------------
// Constructors
// -----------------------------------------------------------------------------

Image::Image(Size2D size, ImageFormat format, eDeviceType device)
    : Image(size, format, GlobalContext().getDefaultAllocator(), device) {}

Image::Image(Size2D size, ImageFormat format, const IAllocator& alloc, eDeviceType device)
    : Image(CalcRequirements(size, format), alloc, device) {}

Image::Image(const Requirements& reqs, eDeviceType device)
    : Image(reqs, GlobalContext().getDefaultAllocator(), device) {}

Image::Image(const Requirements& reqs, const IAllocator& alloc, eDeviceType device)
    : m_data(makeStorage(reqs, alloc, device)), m_metadata(makeImageData(reqs, m_data->data(), device)) {}

Image::Image(ImageData metadata, std::shared_ptr<ImageStorage> storage)
    : m_data(std::move(storage)), m_metadata(std::move(metadata)) {}

// -----------------------------------------------------------------------------
// Accessors
// -----------------------------------------------------------------------------

Size2D Image::size() const noexcept { return m_metadata.cast<ImageDataStrided>()->size(); }

ImageFormat Image::format() const noexcept { return m_metadata.format(); }

eDeviceType Image::device() const noexcept { return m_metadata.device(); }

ImageData Image::exportData() const { return m_metadata; }

// -----------------------------------------------------------------------------
// ImageWrapData
// -----------------------------------------------------------------------------

Image ImageWrapData(const ImageData& data, ImageDataCleanupFunc cleanup) {
    auto strided = data.cast<ImageDataStrided>();
    if (!strided.has_value()) {
        throw Exception("ImageWrapData requires strided image data.", eStatusType::INVALID_VALUE);
    }

    // Storage tracks plane(0)'s base pointer. Single-plane today; multi-plane
    // wraps would need a richer storage shape (or to abandon storing the
    // pointer here at all).
    void* basePtr = strided->plane(0).basePtr;

    // Deleter captures both the original ImageData snapshot and the user's
    // cleanup callback. View-only (cleanup == nullptr) means the deleter
    // touches nothing but the storage object itself.
    auto storage = std::shared_ptr<ImageStorage>(new ImageStorage(basePtr), [data, cleanup](ImageStorage* s) {
        if (cleanup) {
            cleanup(data);
        }
        delete s;
    });

    return Image(data, std::move(storage));
}

}  // namespace roccv
