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

#include <core/image_buffer.hpp>
#include <core/image_data.hpp>
#include <core/image_format.hpp>

#include "image_test_helpers.hpp"
#include "test_helpers.hpp"

using namespace roccv;
using namespace roccv::tests;

namespace {

ImageBufferStrided MakeThreePlaneBuffer() {
    // Mimics a planar layout (e.g. YUV420-style) with sub-sampled chroma — three
    // planes of differing dimensions and strides backed by distinct buffers.
    ImageBufferStrided buf{};
    buf.numPlanes = 3;
    buf.planes[0] = {1920, 1080, 1920, FAKE_PTR_A};  // Y full-resolution
    buf.planes[1] = {960, 540, 960, FAKE_PTR_B};     // U sub-sampled
    buf.planes[2] = {960, 540, 960, FAKE_PTR_C};     // V sub-sampled
    return buf;
}

/**
 * @brief Verifies HIP-strided construction populates all observable state and
 * tags itself as GPU-resident.
 */
void TestImageDataStridedHipConstruction() {
    auto buf = MakeSinglePlaneBuffer(640, 480, 640 * 3, FAKE_PTR_A);
    ImageDataStridedHip data(FMT_RGB8, buf);

    EXPECT_EQ(AsInt(data.device()), AsInt(eDeviceType::GPU));
    EXPECT_EQ(data.numPlanes(), 1);
    EXPECT_EQ(data.size().w, 640);
    EXPECT_EQ(data.size().h, 480);
    EXPECT_EQ(data.plane(0).width, 640);
    EXPECT_EQ(data.plane(0).height, 480);
    EXPECT_EQ(data.plane(0).rowStride, static_cast<int64_t>(640 * 3));
    EXPECT_EQ(AsAddr(data.plane(0).basePtr), AsAddr(FAKE_PTR_A));
    EXPECT_EQ(data.format().channels(), 3);
}

/**
 * @brief Same shape as the Hip test but for Host-resident strided data.
 */
void TestImageDataStridedHostConstruction() {
    auto buf = MakeSinglePlaneBuffer(320, 240, 320, FAKE_PTR_B);
    ImageDataStridedHost data(FMT_U8, buf);

    EXPECT_EQ(AsInt(data.device()), AsInt(eDeviceType::CPU));
    EXPECT_EQ(data.numPlanes(), 1);
    EXPECT_EQ(data.size().w, 320);
    EXPECT_EQ(data.size().h, 240);
    EXPECT_EQ(AsAddr(data.plane(0).basePtr), AsAddr(FAKE_PTR_B));
    EXPECT_EQ(data.format().channels(), 1);
}

/**
 * @brief Multi-plane buffers must round-trip per-plane dimensions and pointers
 * unchanged. size() reports plane 0 by convention; planes 1..N may be smaller.
 */
void TestImageDataStridedMultiPlane() {
    auto buf = MakeThreePlaneBuffer();
    ImageDataStridedHip data(FMT_U8, buf);

    EXPECT_EQ(data.numPlanes(), 3);
    EXPECT_EQ(data.size().w, 1920);
    EXPECT_EQ(data.size().h, 1080);

    EXPECT_EQ(data.plane(0).width, 1920);
    EXPECT_EQ(data.plane(0).height, 1080);
    EXPECT_EQ(AsAddr(data.plane(0).basePtr), AsAddr(FAKE_PTR_A));

    EXPECT_EQ(data.plane(1).width, 960);
    EXPECT_EQ(data.plane(1).height, 540);
    EXPECT_EQ(AsAddr(data.plane(1).basePtr), AsAddr(FAKE_PTR_B));

    EXPECT_EQ(data.plane(2).width, 960);
    EXPECT_EQ(data.plane(2).height, 540);
    EXPECT_EQ(AsAddr(data.plane(2).basePtr), AsAddr(FAKE_PTR_C));
}

/**
 * @brief The two leaf ctors (taking ImageBuffer vs ImageBufferStrided directly)
 * must produce observably identical state.
 */
void TestImageDataStridedSugarCtor() {
    auto buf = MakeSinglePlaneBuffer(100, 200, 400, FAKE_PTR_A);

    ImageDataStridedHip wide(FMT_RGBA8, ImageBuffer{.strided = buf});
    ImageDataStridedHip sugar(FMT_RGBA8, buf);

    EXPECT_EQ(AsInt(wide.device()), AsInt(sugar.device()));
    EXPECT_EQ(wide.numPlanes(), sugar.numPlanes());
    EXPECT_EQ(AsAddr(wide.plane(0).basePtr), AsAddr(sugar.plane(0).basePtr));
    EXPECT_EQ(wide.plane(0).rowStride, sugar.plane(0).rowStride);

    ImageDataStridedHost wideHost(FMT_U8, ImageBuffer{.strided = buf});
    ImageDataStridedHost sugarHost(FMT_U8, buf);
    EXPECT_EQ(AsInt(wideHost.device()), AsInt(sugarHost.device()));
    EXPECT_EQ(AsAddr(wideHost.plane(0).basePtr), AsAddr(sugarHost.plane(0).basePtr));
}

/**
 * @brief IsCompatibleKind on each level discriminates the buffer kinds it
 * accepts. Base accepts anything-but-NONE; Strided accepts both Hip and Host;
 * leaves accept only their own.
 */
void TestImageDataIsCompatibleKind() {
    EXPECT_EQ(AsInt(ImageData::IsCompatibleKind(ImageBufferType::IMAGE_BUFFER_NONE)), 0);
    EXPECT_EQ(AsInt(ImageData::IsCompatibleKind(ImageBufferType::IMAGE_BUFFER_STRIDED_HIP)), 1);
    EXPECT_EQ(AsInt(ImageData::IsCompatibleKind(ImageBufferType::IMAGE_BUFFER_STRIDED_HOST)), 1);

    EXPECT_EQ(AsInt(ImageDataStrided::IsCompatibleKind(ImageBufferType::IMAGE_BUFFER_NONE)), 0);
    EXPECT_EQ(AsInt(ImageDataStrided::IsCompatibleKind(ImageBufferType::IMAGE_BUFFER_STRIDED_HIP)), 1);
    EXPECT_EQ(AsInt(ImageDataStrided::IsCompatibleKind(ImageBufferType::IMAGE_BUFFER_STRIDED_HOST)), 1);

    EXPECT_EQ(AsInt(ImageDataStridedHip::IsCompatibleKind(ImageBufferType::IMAGE_BUFFER_NONE)), 0);
    EXPECT_EQ(AsInt(ImageDataStridedHip::IsCompatibleKind(ImageBufferType::IMAGE_BUFFER_STRIDED_HIP)), 1);
    EXPECT_EQ(AsInt(ImageDataStridedHip::IsCompatibleKind(ImageBufferType::IMAGE_BUFFER_STRIDED_HOST)), 0);

    EXPECT_EQ(AsInt(ImageDataStridedHost::IsCompatibleKind(ImageBufferType::IMAGE_BUFFER_NONE)), 0);
    EXPECT_EQ(AsInt(ImageDataStridedHost::IsCompatibleKind(ImageBufferType::IMAGE_BUFFER_STRIDED_HIP)), 0);
    EXPECT_EQ(AsInt(ImageDataStridedHost::IsCompatibleKind(ImageBufferType::IMAGE_BUFFER_STRIDED_HOST)), 1);
}

/**
 * @brief Round-trip a derived ImageData through the base reference and back
 * via cast<>(). Successful casts must preserve every observable field; casts
 * to incompatible kinds must return std::nullopt.
 */
void TestImageDataCast() {
    auto buf = MakeSinglePlaneBuffer(800, 600, 800 * 4, FAKE_PTR_A);

    // Hip → base → Hip should round-trip, Hip → Host should fail.
    {
        ImageDataStridedHip hip(FMT_RGBA8, buf);
        const ImageData& base = hip;

        auto asHip = base.cast<ImageDataStridedHip>();
        EXPECT_EQ(AsInt(asHip.has_value()), 1);
        EXPECT_EQ(AsInt(asHip->device()), AsInt(eDeviceType::GPU));
        EXPECT_EQ(AsAddr(asHip->plane(0).basePtr), AsAddr(FAKE_PTR_A));
        EXPECT_EQ(asHip->plane(0).width, 800);

        auto asStrided = base.cast<ImageDataStrided>();
        EXPECT_EQ(AsInt(asStrided.has_value()), 1);
        EXPECT_EQ(AsInt(asStrided->device()), AsInt(eDeviceType::GPU));

        auto asHost = base.cast<ImageDataStridedHost>();
        EXPECT_EQ(AsInt(asHost.has_value()), 0);
    }

    // Symmetrically: Host → base → Host succeeds, Host → Hip fails.
    {
        ImageDataStridedHost host(FMT_RGBA8, buf);
        const ImageData& base = host;

        auto asHost = base.cast<ImageDataStridedHost>();
        EXPECT_EQ(AsInt(asHost.has_value()), 1);
        EXPECT_EQ(AsInt(asHost->device()), AsInt(eDeviceType::CPU));

        auto asHip = base.cast<ImageDataStridedHip>();
        EXPECT_EQ(AsInt(asHip.has_value()), 0);
    }
}

}  // namespace

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;
    TEST_CASES_BEGIN();

    TEST_CASE(TestImageDataStridedHipConstruction());
    TEST_CASE(TestImageDataStridedHostConstruction());
    TEST_CASE(TestImageDataStridedMultiPlane());
    TEST_CASE(TestImageDataStridedSugarCtor());
    TEST_CASE(TestImageDataIsCompatibleKind());
    TEST_CASE(TestImageDataCast());

    TEST_CASES_END();
}
