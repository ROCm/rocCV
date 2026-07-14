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

#include <core/image_batch_buffer.hpp>
#include <core/image_batch_data.hpp>
#include <core/image_buffer.hpp>
#include <core/image_format.hpp>

#include "image_test_helpers.hpp"
#include "test_helpers.hpp"

using namespace roccv;
using namespace roccv::tests;

namespace {

// Static descriptor/format storage for the batch buffer. These are real host
// allocations (so the pointers are valid) but the batch tests only read
// metadata back out of them; nothing dereferences the per-image basePtr fields.
ImageBufferStrided g_imageList[2];
ImageFormat g_formatList[2] = {FMT_RGB8, FMT_RGB8};
ImageFormat g_hostFormatList[2] = {FMT_RGB8, FMT_RGB8};

// Builds a homogeneous two-image varshape descriptor with a known bounding box
// and uniqueFormat. The returned struct's pointers reference module-static
// arrays so addresses remain stable across calls within a test.
ImageBatchVarShapeBufferStrided MakeHomogeneousBuffer() {
    g_imageList[0] = MakeSinglePlaneBuffer(640, 480, 640 * 3, FAKE_PTR_A);
    g_imageList[1] = MakeSinglePlaneBuffer(320, 240, 320 * 3, FAKE_PTR_B);
    g_formatList[0] = FMT_RGB8;
    g_formatList[1] = FMT_RGB8;
    g_hostFormatList[0] = FMT_RGB8;
    g_hostFormatList[1] = FMT_RGB8;

    ImageBatchVarShapeBufferStrided buf{};
    buf.uniqueFormat = FMT_RGB8;
    buf.maxWidth = 640;
    buf.maxHeight = 480;
    buf.formatList = g_formatList;
    buf.hostFormatList = g_hostFormatList;
    buf.imageList = g_imageList;
    return buf;
}

/**
 * @brief Verifies HIP-strided varshape construction populates all observable
 * state and tags itself as GPU-resident.
 */
void TestImageBatchVarShapeDataStridedHipConstruction() {
    auto buf = MakeHomogeneousBuffer();
    ImageBatchVarShapeDataStridedHip data(2, buf);

    EXPECT_EQ(AsInt(data.device()), AsInt(eDeviceType::GPU));
    EXPECT_EQ(data.numImages(), 2);
    EXPECT_EQ(data.maxSize().w, 640);
    EXPECT_EQ(data.maxSize().h, 480);
    EXPECT_EQ(data.uniqueFormat().channels(), 3);
    EXPECT_EQ(AsAddr(data.formatList()), AsAddr(g_formatList));
    EXPECT_EQ(AsAddr(data.hostFormatList()), AsAddr(g_hostFormatList));
    EXPECT_EQ(AsAddr(data.imageList()), AsAddr(g_imageList));
    EXPECT_EQ(data.imageList()[0].planes[0].width, 640);
    EXPECT_EQ(data.imageList()[1].planes[0].width, 320);
}

/**
 * @brief Same shape as the Hip test but for Host-resident varshape data.
 */
void TestImageBatchVarShapeDataStridedHostConstruction() {
    auto buf = MakeHomogeneousBuffer();
    ImageBatchVarShapeDataStridedHost data(2, buf);

    EXPECT_EQ(AsInt(data.device()), AsInt(eDeviceType::CPU));
    EXPECT_EQ(data.numImages(), 2);
    EXPECT_EQ(data.maxSize().w, 640);
    EXPECT_EQ(data.maxSize().h, 480);
    EXPECT_EQ(data.uniqueFormat().channels(), 3);
    EXPECT_EQ(AsAddr(data.imageList()), AsAddr(g_imageList));
}

/**
 * @brief Empty batch: maxSize collapses to 0x0 and uniqueFormat is FMT_NONE.
 * Producers signal "no images" via numImages == 0; the buffer fields stay
 * valid pointers but get ignored.
 */
void TestImageBatchVarShapeDataEmpty() {
    ImageBatchVarShapeBufferStrided buf{};
    buf.uniqueFormat = FMT_NONE;
    buf.maxWidth = 0;
    buf.maxHeight = 0;
    buf.formatList = g_formatList;
    buf.hostFormatList = g_hostFormatList;
    buf.imageList = g_imageList;

    ImageBatchVarShapeDataStridedHip data(0, buf);

    EXPECT_EQ(data.numImages(), 0);
    EXPECT_EQ(data.maxSize().w, 0);
    EXPECT_EQ(data.maxSize().h, 0);
    EXPECT_EQ(AsInt(data.uniqueFormat() == FMT_NONE), 1);
}

/**
 * @brief Heterogeneous formats: per-image formatList carries each entry
 * verbatim; uniqueFormat is FMT_NONE since no single format spans the batch.
 */
void TestImageBatchVarShapeDataHeterogeneousFormats() {
    g_imageList[0] = MakeSinglePlaneBuffer(640, 480, 640 * 3, FAKE_PTR_A);
    g_imageList[1] = MakeSinglePlaneBuffer(320, 240, 320 * 4, FAKE_PTR_B);
    g_formatList[0] = FMT_RGB8;
    g_formatList[1] = FMT_RGBA8;
    g_hostFormatList[0] = FMT_RGB8;
    g_hostFormatList[1] = FMT_RGBA8;

    ImageBatchVarShapeBufferStrided buf{};
    buf.uniqueFormat = FMT_NONE;
    buf.maxWidth = 640;
    buf.maxHeight = 480;
    buf.formatList = g_formatList;
    buf.hostFormatList = g_hostFormatList;
    buf.imageList = g_imageList;

    ImageBatchVarShapeDataStridedHip data(2, buf);

    EXPECT_EQ(AsInt(data.uniqueFormat() == FMT_NONE), 1);
    EXPECT_EQ(AsInt(data.hostFormatList()[0] == FMT_RGB8), 1);
    EXPECT_EQ(AsInt(data.hostFormatList()[1] == FMT_RGBA8), 1);
}

/**
 * @brief The two leaf ctors (taking ImageBatchBuffer vs the concrete strided
 * buffer directly) must produce observably identical state.
 */
void TestImageBatchVarShapeDataSugarCtor() {
    auto buf = MakeHomogeneousBuffer();

    ImageBatchVarShapeDataStridedHip wide(2, ImageBatchBuffer{.varShapeStrided = buf});
    ImageBatchVarShapeDataStridedHip sugar(2, buf);

    EXPECT_EQ(AsInt(wide.device()), AsInt(sugar.device()));
    EXPECT_EQ(wide.numImages(), sugar.numImages());
    EXPECT_EQ(wide.maxSize().w, sugar.maxSize().w);
    EXPECT_EQ(wide.maxSize().h, sugar.maxSize().h);
    EXPECT_EQ(AsAddr(wide.imageList()), AsAddr(sugar.imageList()));

    ImageBatchVarShapeDataStridedHost wideHost(2, ImageBatchBuffer{.varShapeStrided = buf});
    ImageBatchVarShapeDataStridedHost sugarHost(2, buf);
    EXPECT_EQ(AsInt(wideHost.device()), AsInt(sugarHost.device()));
    EXPECT_EQ(AsAddr(wideHost.imageList()), AsAddr(sugarHost.imageList()));
}

/**
 * @brief IsCompatibleKind on each level discriminates the buffer kinds it
 * accepts. Base accepts anything-but-NONE; VarShape and VarShapeStrided accept
 * both Hip and Host varshape; leaves accept only their own.
 */
void TestImageBatchDataIsCompatibleKind() {
    EXPECT_EQ(AsInt(ImageBatchData::IsCompatibleKind(ImageBatchBufferType::IMAGE_BATCH_BUFFER_NONE)), 0);
    EXPECT_EQ(AsInt(ImageBatchData::IsCompatibleKind(ImageBatchBufferType::IMAGE_BATCH_VARSHAPE_BUFFER_STRIDED_HIP)),
              1);
    EXPECT_EQ(AsInt(ImageBatchData::IsCompatibleKind(ImageBatchBufferType::IMAGE_BATCH_VARSHAPE_BUFFER_STRIDED_HOST)),
              1);

    EXPECT_EQ(AsInt(ImageBatchVarShapeData::IsCompatibleKind(ImageBatchBufferType::IMAGE_BATCH_BUFFER_NONE)), 0);
    EXPECT_EQ(
        AsInt(ImageBatchVarShapeData::IsCompatibleKind(ImageBatchBufferType::IMAGE_BATCH_VARSHAPE_BUFFER_STRIDED_HIP)),
        1);
    EXPECT_EQ(
        AsInt(ImageBatchVarShapeData::IsCompatibleKind(ImageBatchBufferType::IMAGE_BATCH_VARSHAPE_BUFFER_STRIDED_HOST)),
        1);

    EXPECT_EQ(AsInt(ImageBatchVarShapeDataStrided::IsCompatibleKind(ImageBatchBufferType::IMAGE_BATCH_BUFFER_NONE)), 0);
    EXPECT_EQ(AsInt(ImageBatchVarShapeDataStrided::IsCompatibleKind(
                  ImageBatchBufferType::IMAGE_BATCH_VARSHAPE_BUFFER_STRIDED_HIP)),
              1);
    EXPECT_EQ(AsInt(ImageBatchVarShapeDataStrided::IsCompatibleKind(
                  ImageBatchBufferType::IMAGE_BATCH_VARSHAPE_BUFFER_STRIDED_HOST)),
              1);

    EXPECT_EQ(AsInt(ImageBatchVarShapeDataStridedHip::IsCompatibleKind(ImageBatchBufferType::IMAGE_BATCH_BUFFER_NONE)),
              0);
    EXPECT_EQ(AsInt(ImageBatchVarShapeDataStridedHip::IsCompatibleKind(
                  ImageBatchBufferType::IMAGE_BATCH_VARSHAPE_BUFFER_STRIDED_HIP)),
              1);
    EXPECT_EQ(AsInt(ImageBatchVarShapeDataStridedHip::IsCompatibleKind(
                  ImageBatchBufferType::IMAGE_BATCH_VARSHAPE_BUFFER_STRIDED_HOST)),
              0);

    EXPECT_EQ(AsInt(ImageBatchVarShapeDataStridedHost::IsCompatibleKind(ImageBatchBufferType::IMAGE_BATCH_BUFFER_NONE)),
              0);
    EXPECT_EQ(AsInt(ImageBatchVarShapeDataStridedHost::IsCompatibleKind(
                  ImageBatchBufferType::IMAGE_BATCH_VARSHAPE_BUFFER_STRIDED_HIP)),
              0);
    EXPECT_EQ(AsInt(ImageBatchVarShapeDataStridedHost::IsCompatibleKind(
                  ImageBatchBufferType::IMAGE_BATCH_VARSHAPE_BUFFER_STRIDED_HOST)),
              1);
}

/**
 * @brief Round-trip a derived ImageBatchData through the base reference and
 * back via cast<>(). Successful casts must preserve every observable field;
 * casts to incompatible kinds must return std::nullopt.
 */
void TestImageBatchDataCast() {
    auto buf = MakeHomogeneousBuffer();

    // Hip → base → Hip should round-trip; intermediate VarShape/Strided also
    // succeed; Hip → Host fails.
    {
        ImageBatchVarShapeDataStridedHip hip(2, buf);
        const ImageBatchData& base = hip;

        auto asHip = base.cast<ImageBatchVarShapeDataStridedHip>();
        EXPECT_EQ(AsInt(asHip.has_value()), 1);
        EXPECT_EQ(AsInt(asHip->device()), AsInt(eDeviceType::GPU));
        EXPECT_EQ(asHip->numImages(), 2);
        EXPECT_EQ(asHip->maxSize().w, 640);
        EXPECT_EQ(AsAddr(asHip->imageList()), AsAddr(g_imageList));

        auto asStrided = base.cast<ImageBatchVarShapeDataStrided>();
        EXPECT_EQ(AsInt(asStrided.has_value()), 1);
        EXPECT_EQ(AsInt(asStrided->device()), AsInt(eDeviceType::GPU));

        auto asVar = base.cast<ImageBatchVarShapeData>();
        EXPECT_EQ(AsInt(asVar.has_value()), 1);
        EXPECT_EQ(asVar->maxSize().h, 480);

        auto asHost = base.cast<ImageBatchVarShapeDataStridedHost>();
        EXPECT_EQ(AsInt(asHost.has_value()), 0);
    }

    // Symmetrically: Host → base → Host succeeds, Host → Hip fails.
    {
        ImageBatchVarShapeDataStridedHost host(2, buf);
        const ImageBatchData& base = host;

        auto asHost = base.cast<ImageBatchVarShapeDataStridedHost>();
        EXPECT_EQ(AsInt(asHost.has_value()), 1);
        EXPECT_EQ(AsInt(asHost->device()), AsInt(eDeviceType::CPU));
        EXPECT_EQ(asHost->numImages(), 2);

        auto asHip = base.cast<ImageBatchVarShapeDataStridedHip>();
        EXPECT_EQ(AsInt(asHip.has_value()), 0);
    }
}

/**
 * @brief Regression: casting up to an intermediate type
 * (ImageBatchVarShapeData / ...Strided) must preserve the source's residency
 * and buffer kind. The intermediate constructors leave both at their base
 * defaults (GPU / IMAGE_BATCH_BUFFER_NONE), so cast<>() must carry them over
 * from the source. Otherwise a host batch would silently report
 * device()==GPU and lose its buffer kind, which then breaks a subsequent
 * re-cast back down to the leaf type.
 */
void TestImageBatchDataCastPreservesResidencyOnUpcast() {
    auto buf = MakeHomogeneousBuffer();

    // Upcasting a host leaf to either intermediate type must keep CPU
    // residency.
    ImageBatchVarShapeDataStridedHost host(2, buf);
    const ImageBatchData& base = host;

    auto strided = base.cast<ImageBatchVarShapeDataStrided>();
    EXPECT_EQ(AsInt(strided.has_value()), 1);
    EXPECT_EQ(AsInt(strided->device()), AsInt(eDeviceType::CPU));

    auto asVar = base.cast<ImageBatchVarShapeData>();
    EXPECT_EQ(AsInt(asVar.has_value()), 1);
    EXPECT_EQ(AsInt(asVar->device()), AsInt(eDeviceType::CPU));

    // The buffer kind must survive the upcast too: re-casting the upcasted
    // value back down to the host leaf must still succeed (it would fail if the
    // kind had been reset to IMAGE_BATCH_BUFFER_NONE)...
    const ImageBatchData& stridedBase = strided.value();
    auto backToHost = stridedBase.cast<ImageBatchVarShapeDataStridedHost>();
    EXPECT_EQ(AsInt(backToHost.has_value()), 1);
    EXPECT_EQ(AsInt(backToHost->device()), AsInt(eDeviceType::CPU));

    // ...and must not spuriously match the wrong leaf kind.
    auto backToHip = stridedBase.cast<ImageBatchVarShapeDataStridedHip>();
    EXPECT_EQ(AsInt(backToHip.has_value()), 0);
}

}  // namespace

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;
    TEST_CASES_BEGIN();

    TEST_CASE(TestImageBatchVarShapeDataStridedHipConstruction());
    TEST_CASE(TestImageBatchVarShapeDataStridedHostConstruction());
    TEST_CASE(TestImageBatchVarShapeDataEmpty());
    TEST_CASE(TestImageBatchVarShapeDataHeterogeneousFormats());
    TEST_CASE(TestImageBatchVarShapeDataSugarCtor());
    TEST_CASE(TestImageBatchDataIsCompatibleKind());
    TEST_CASE(TestImageBatchDataCast());
    TEST_CASE(TestImageBatchDataCastPreservesResidencyOnUpcast());

    TEST_CASES_END();
}
