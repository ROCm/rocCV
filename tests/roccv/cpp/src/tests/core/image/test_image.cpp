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

#include <stdint.h>

#include <core/image.hpp>
#include <core/image_data.hpp>
#include <core/image_format.hpp>
#include <typeinfo>
#include <utility>

#include "image_test_helpers.hpp"
#include "test_helpers.hpp"

using namespace roccv;
using namespace roccv::tests;

namespace {

// =============================================================================
// CalcRequirements
// =============================================================================

/**
 * @brief Packed-row stride for a typical 3-channel uint8 image. Other fields
 * propagate unchanged; remaining plane slots stay zeroed.
 */
void TestCalcRequirementsRgb8() {
    auto reqs = Image::CalcRequirements({320, 240}, FMT_RGB8);

    EXPECT_EQ(reqs.size.w, 320);
    EXPECT_EQ(reqs.size.h, 240);
    EXPECT_EQ(reqs.format.channels(), 3);
    EXPECT_EQ(reqs.planeRowStride[0], static_cast<int64_t>(320 * 3));
    EXPECT_EQ(reqs.planeRowStride[1], 0);
    EXPECT_EQ(reqs.planeRowStride[5], 0);
    EXPECT_EQ(reqs.alignBytes, 0);
}

/**
 * @brief Multi-byte dtype is reflected in the per-pixel byte count.
 */
void TestCalcRequirementsF32() {
    auto reqs = Image::CalcRequirements({64, 64}, FMT_F32);
    EXPECT_EQ(reqs.planeRowStride[0], static_cast<int64_t>(64 * 4));
}

/**
 * @brief Single-channel U8 → row stride equals width.
 */
void TestCalcRequirementsU8() {
    auto reqs = Image::CalcRequirements({100, 50}, FMT_U8);
    EXPECT_EQ(reqs.planeRowStride[0], 100);
}

/**
 * @brief Width or height < 1 must throw INVALID_VALUE.
 */
void TestCalcRequirementsRejectsInvalidDims() {
    EXPECT_EXCEPTION(Image::CalcRequirements({0, 100}, FMT_RGB8), eStatusType::INVALID_VALUE);
    EXPECT_EXCEPTION(Image::CalcRequirements({100, 0}, FMT_RGB8), eStatusType::INVALID_VALUE);
    EXPECT_EXCEPTION(Image::CalcRequirements({-5, 100}, FMT_RGB8), eStatusType::INVALID_VALUE);
    EXPECT_EXCEPTION(Image::CalcRequirements({100, -5}, FMT_RGB8), eStatusType::INVALID_VALUE);
}

/**
 * @brief Large widths must not overflow during stride math; row stride must
 * fit in int64.
 */
void TestCalcRequirementsLargeDims() {
    // 8K image, RGBA8 (4 channels * 1 byte = 4 B/pixel) → 8192 * 4 = 32768 B/row.
    auto reqs = Image::CalcRequirements({8192, 4320}, FMT_RGBA8);
    EXPECT_EQ(reqs.planeRowStride[0], static_cast<int64_t>(8192 * 4));
}

// =============================================================================
// Allocating constructors
// =============================================================================

/**
 * @brief GPU-device ctor routes allocation through allocHipMem with the
 * computed byte count.
 */
void TestImageHipAllocation() {
    CountingAllocator alloc;
    {
        Image img({320, 240}, FMT_RGB8, alloc, eDeviceType::GPU);

        EXPECT_EQ(alloc.hipAllocs, 1);
        EXPECT_EQ(alloc.hostAllocs, 0);
        EXPECT_EQ(AsSize(alloc.lastAllocBytes), AsSize(320 * 3 * 240));

        EXPECT_EQ(img.size().w, 320);
        EXPECT_EQ(img.size().h, 240);
        EXPECT_EQ(AsInt(img.device()), AsInt(eDeviceType::GPU));
        EXPECT_EQ(img.format().channels(), 3);

        // Image is still alive — buffer not yet freed.
        EXPECT_EQ(alloc.hipFrees, 0);
    }
    // Image dropped — buffer freed exactly once via the matching allocator.
    EXPECT_EQ(alloc.hipFrees, 1);
}

/**
 * @brief Same shape as the Hip test but for CPU residency.
 */
void TestImageHostAllocation() {
    CountingAllocator alloc;
    {
        Image img({100, 50}, FMT_U8, alloc, eDeviceType::CPU);

        EXPECT_EQ(alloc.hostAllocs, 1);
        EXPECT_EQ(alloc.hipAllocs, 0);
        EXPECT_EQ(AsSize(alloc.lastAllocBytes), AsSize(100 * 50));
        EXPECT_EQ(AsInt(img.device()), AsInt(eDeviceType::CPU));
    }
    EXPECT_EQ(alloc.hostFrees, 1);
}

/**
 * @brief Constructing from precomputed Requirements yields observably
 * identical state to the (Size2D, ImageFormat) sugar form.
 */
void TestImageRequirementsCtor() {
    CountingAllocator alloc;
    auto reqs = Image::CalcRequirements({64, 32}, FMT_RGBA8);

    Image img(reqs, alloc, eDeviceType::GPU);

    EXPECT_EQ(img.size().w, 64);
    EXPECT_EQ(img.size().h, 32);
    EXPECT_EQ(img.format().channels(), 4);
    EXPECT_EQ(AsSize(alloc.lastAllocBytes), AsSize(64 * 4 * 32));
}

// =============================================================================
// Refcount / lifecycle
// =============================================================================

/**
 * @brief Copying an Image bumps the refcount: both handles see the same
 * underlying buffer, and free is deferred until the LAST handle drops.
 */
void TestImageCopySharesBuffer() {
    CountingAllocator alloc;
    void* buf = nullptr;
    {
        Image first({16, 16}, FMT_U8, alloc, eDeviceType::GPU);
        buf = first.exportData().cast<ImageDataStrided>()->plane(0).basePtr;

        Image second = first;           // refcount bump
        EXPECT_EQ(alloc.hipAllocs, 1);  // No new allocation.
        EXPECT_EQ(AsAddr(second.exportData().cast<ImageDataStrided>()->plane(0).basePtr), AsAddr(buf));

        // Drop `first`; buffer must NOT be freed yet — `second` still holds it.
        {
            Image sink = std::move(first);
        }
        EXPECT_EQ(alloc.hipFrees, 0);
    }
    // All handles dropped — exactly one free.
    EXPECT_EQ(alloc.hipFrees, 1);
}

/**
 * @brief Move-construction transfers the buffer; the source is left empty.
 * The buffer must still free exactly once (when the destination drops).
 */
void TestImageMoveSemantics() {
    CountingAllocator alloc;
    {
        Image src({8, 8}, FMT_U8, alloc, eDeviceType::CPU);
        void* srcBuf = src.exportData().cast<ImageDataStrided>()->plane(0).basePtr;

        Image dst = std::move(src);
        EXPECT_EQ(AsAddr(dst.exportData().cast<ImageDataStrided>()->plane(0).basePtr), AsAddr(srcBuf));
        EXPECT_EQ(alloc.hostFrees, 0);
    }
    EXPECT_EQ(alloc.hostFrees, 1);
}

// =============================================================================
// exportData / exportData<DATA>()
// =============================================================================

/**
 * @brief exportData() returns an ImageData snapshot that mirrors the Image's
 * size, format, device, and base pointer.
 */
void TestImageExportData() {
    CountingAllocator alloc;
    Image img({80, 60}, FMT_RGBA8, alloc, eDeviceType::GPU);
    ImageData data = img.exportData();

    EXPECT_EQ(AsInt(data.device()), AsInt(eDeviceType::GPU));
    EXPECT_EQ(data.format().channels(), 4);

    auto strided = data.cast<ImageDataStrided>();
    EXPECT_EQ(AsInt(strided.has_value()), 1);
    EXPECT_EQ(strided->plane(0).width, 80);
    EXPECT_EQ(strided->plane(0).height, 60);
    EXPECT_EQ(strided->plane(0).rowStride, static_cast<int64_t>(80 * 4));
}

/**
 * @brief Templated exportData<T>() returns the matching subclass directly.
 */
void TestImageExportDataTypedSuccess() {
    CountingAllocator alloc;
    Image img({4, 4}, FMT_U8, alloc, eDeviceType::GPU);

    auto hip = img.exportData<ImageDataStridedHip>();
    EXPECT_EQ(AsInt(hip.device()), AsInt(eDeviceType::GPU));
    EXPECT_EQ(hip.plane(0).width, 4);
}

/**
 * @brief Templated exportData<T>() throws std::bad_cast when the requested
 * subclass does not match the underlying buffer kind.
 */
void TestImageExportDataTypedMismatch() {
    CountingAllocator alloc;
    Image img({4, 4}, FMT_U8, alloc, eDeviceType::GPU);

    bool threw = false;
    try {
        (void)img.exportData<ImageDataStridedHost>();
    } catch (const std::bad_cast&) {
        threw = true;
    }
    EXPECT_EQ(AsInt(threw), 1);
}

// =============================================================================
// ImageWrapData
// =============================================================================

/**
 * @brief View-only wrap (no cleanup callback) round-trips metadata and must
 * not crash when the Image is destroyed (no free attempt on the sentinel ptr).
 */
void TestImageWrapDataViewOnly() {
    Image wrapped = ImageWrapData(MakeFakeHipData(640, 480, FAKE_PTR_A));
    EXPECT_EQ(wrapped.size().w, 640);
    EXPECT_EQ(wrapped.size().h, 480);
    EXPECT_EQ(AsInt(wrapped.device()), AsInt(eDeviceType::GPU));
    EXPECT_EQ(AsAddr(wrapped.exportData().cast<ImageDataStrided>()->plane(0).basePtr), AsAddr(FAKE_PTR_A));
}

/**
 * @brief Wrap with a cleanup callback: the callback fires exactly once when
 * the last Image handle goes out of scope.
 */
void TestImageWrapDataCleanupFires() {
    int callbackInvocations = 0;
    {
        Image wrapped =
            ImageWrapData(MakeFakeHipData(100, 100, FAKE_PTR_A), [&](const ImageData&) { ++callbackInvocations; });
        EXPECT_EQ(callbackInvocations, 0);  // Not fired during normal use.
    }
    EXPECT_EQ(callbackInvocations, 1);
}

/**
 * @brief Cleanup callback receives the original wrapped ImageData snapshot —
 * the captured base pointer must match what was passed to ImageWrapData.
 */
void TestImageWrapDataCleanupReceivesData() {
    void* receivedBasePtr = nullptr;
    {
        Image wrapped = ImageWrapData(MakeFakeHipData(50, 50, FAKE_PTR_A), [&](const ImageData& d) {
            receivedBasePtr = d.cast<ImageDataStrided>()->plane(0).basePtr;
        });
    }
    EXPECT_EQ(AsAddr(receivedBasePtr), AsAddr(FAKE_PTR_A));
}

/**
 * @brief Cleanup must fire only on LAST handle drop — copies bump the
 * refcount, intermediate drops do nothing.
 */
void TestImageWrapDataCleanupFiresOnce() {
    int callbackInvocations = 0;
    {
        Image first =
            ImageWrapData(MakeFakeHipData(10, 10, FAKE_PTR_A), [&](const ImageData&) { ++callbackInvocations; });
        Image second = first;  // refcount = 2
        Image third = first;   // refcount = 3
        {
            Image fourth = third;
            (void)fourth;
        }  // dropped → refcount = 3
        EXPECT_EQ(callbackInvocations, 0);
        // first, second, third still alive at scope exit
    }
    EXPECT_EQ(callbackInvocations, 1);
}

/**
 * @brief Wrapped Image's accessors mirror the wrapped ImageData verbatim —
 * size, format, device, and base pointer all round-trip unchanged.
 */
void TestImageWrapDataAccessors() {
    auto fake = MakeFakeHipData(123, 45, FAKE_PTR_A, FMT_RGBA8);
    Image wrapped = ImageWrapData(fake);

    EXPECT_EQ(wrapped.size().w, 123);
    EXPECT_EQ(wrapped.size().h, 45);
    EXPECT_EQ(wrapped.format().channels(), 4);
    EXPECT_EQ(AsInt(wrapped.device()), AsInt(eDeviceType::GPU));

    auto strided = wrapped.exportData().cast<ImageDataStrided>();
    EXPECT_EQ(AsInt(strided.has_value()), 1);
    EXPECT_EQ(strided->plane(0).width, 123);
    EXPECT_EQ(strided->plane(0).height, 45);
    EXPECT_EQ(AsAddr(strided->plane(0).basePtr), AsAddr(FAKE_PTR_A));
}

}  // namespace

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;
    TEST_CASES_BEGIN();

    // CalcRequirements
    TEST_CASE(TestCalcRequirementsRgb8());
    TEST_CASE(TestCalcRequirementsF32());
    TEST_CASE(TestCalcRequirementsU8());
    TEST_CASE(TestCalcRequirementsRejectsInvalidDims());
    TEST_CASE(TestCalcRequirementsLargeDims());

    // Allocating constructors
    TEST_CASE(TestImageHipAllocation());
    TEST_CASE(TestImageHostAllocation());
    TEST_CASE(TestImageRequirementsCtor());

    // Refcount / lifecycle
    TEST_CASE(TestImageCopySharesBuffer());
    TEST_CASE(TestImageMoveSemantics());

    // exportData
    TEST_CASE(TestImageExportData());
    TEST_CASE(TestImageExportDataTypedSuccess());
    TEST_CASE(TestImageExportDataTypedMismatch());

    // ImageWrapData
    TEST_CASE(TestImageWrapDataViewOnly());
    TEST_CASE(TestImageWrapDataCleanupFires());
    TEST_CASE(TestImageWrapDataCleanupReceivesData());
    TEST_CASE(TestImageWrapDataCleanupFiresOnce());
    TEST_CASE(TestImageWrapDataAccessors());

    TEST_CASES_END();
}
