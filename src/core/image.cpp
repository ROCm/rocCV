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

#include <algorithm>
#include <iterator>

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

}  // namespace

// -----------------------------------------------------------------------------
// CalcRequirements
// -----------------------------------------------------------------------------

Image::Requirements Image::CalcRequirements(Size2D size, ImageFormat format) {
    if (size.w < 1 || size.h < 1) {
        throw Exception("Image dimensions must be >= 1.", eStatusType::INVALID_VALUE);
    }

    const int64_t bytesPerPixel = static_cast<int64_t>(DataType(format.dtype()).size()) * format.channels();

    // TODO: derive a sensible default base/row alignment from device attributes.
    return ImageRequirements{
        .size = size,
        .format = format,
        .planeRowStride = {bytesPerPixel * size.w},
        .alignBytes = 0,
    };
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
    : Image(reqs, device, makeStorage(reqs, alloc, device)) {}

Image::Image(const Requirements& reqs, eDeviceType device, std::shared_ptr<ImageStorage> storage)
    : m_data(std::move(storage)),
      m_size(reqs.size),
      m_format(reqs.format),
      m_device(device),
      m_planeRowStride{} {
    std::copy(std::begin(reqs.planeRowStride), std::end(reqs.planeRowStride), m_planeRowStride.begin());
}

// -----------------------------------------------------------------------------
// exportData
// -----------------------------------------------------------------------------

ImageData Image::exportData() const {
    // TODO: derive numPlanes from m_format when planar formats land. Today's
    // ImageFormat is interleaved-only, so plane 0 covers the whole image and
    // its dimensions match m_size verbatim.
    ImageBufferStrided strided{};
    strided.numPlanes = 1;
    strided.planes[0].width = m_size.w;
    strided.planes[0].height = m_size.h;
    strided.planes[0].rowStride = m_planeRowStride[0];
    strided.planes[0].basePtr = m_data->data();

    switch (m_device) {
        case eDeviceType::GPU:
            return ImageDataStridedHip(m_format, strided);
        case eDeviceType::CPU:
            return ImageDataStridedHost(m_format, strided);
    }

    throw Exception("Unsupported device type in Image::exportData.", eStatusType::INVALID_VALUE);
}

// -----------------------------------------------------------------------------
// ImageWrapData
// -----------------------------------------------------------------------------

Image ImageWrapData(const ImageData& data, ImageDataCleanupFunc cleanup) {
    auto strided = data.cast<ImageDataStrided>();
    if (!strided.has_value()) {
        throw Exception("ImageWrapData requires strided image data.", eStatusType::INVALID_VALUE);
    }

    // Single-plane assumption: storage tracks plane(0) and Requirements only
    // populates planeRowStride[0]. Multi-plane wraps will need to copy each
    // plane's stride and either store per-plane base pointers or derive them
    // from a single owning allocation.
    const ImagePlaneStrided& plane0 = strided->plane(0);

    // Designated initializers to avoid value-initializing ImageFormat through
    // its explicit default ctor (which copy-list-init refuses).
    Image::Requirements reqs{
        .size = Size2D{plane0.width, plane0.height},
        .format = data.format(),
        .planeRowStride = {plane0.rowStride},
        .alignBytes = 0,
    };

    // The deleter captures `data` by value so the original snapshot survives
    // long enough to be passed to the cleanup callback on last-handle drop.
    auto storage = std::shared_ptr<ImageStorage>(new ImageStorage(plane0.basePtr),
                                                 [data, cleanup](ImageStorage* s) {
                                                     if (cleanup) {
                                                         cleanup(data);
                                                     }
                                                     delete s;
                                                 });

    return Image(reqs, data.device(), std::move(storage));
}

}  // namespace roccv
