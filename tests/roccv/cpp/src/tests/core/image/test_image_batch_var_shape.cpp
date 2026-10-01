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

#include <hip/hip_runtime.h>
#include <stdint.h>

#include <core/image.hpp>
#include <core/image_batch_data.hpp>
#include <core/image_batch_var_shape.hpp>
#include <core/image_format.hpp>
#include <utility>
#include <vector>

#include "image_test_helpers.hpp"
#include "test_helpers.hpp"

using namespace roccv;
using namespace roccv::tests;

namespace {

// =============================================================================
// Construction
// =============================================================================

void TestConstruction() {
    CountingAllocator alloc;
    {
        ImageBatchVarShape batch(8, alloc);
        EXPECT_EQ(batch.capacity(), 8);
        EXPECT_EQ(batch.numImages(), 0);
        EXPECT_EQ(AsInt(batch.begin() == batch.end()), 1);
    }
    EXPECT_EQ(alloc.hipAllocs, 2);
    EXPECT_EQ(alloc.pinnedAllocs, 2);
    EXPECT_EQ(alloc.hipFrees, 2);
    EXPECT_EQ(alloc.pinnedFrees, 2);
}

void TestConstructionCpu() {
    CountingAllocator alloc;
    {
        ImageBatchVarShape batch(8, alloc, eDeviceType::CPU);
        EXPECT_EQ(batch.capacity(), 8);
        EXPECT_EQ(batch.numImages(), 0);
        EXPECT_EQ(AsInt(batch.device()), AsInt(eDeviceType::CPU));
    }
    // A CPU batch allocates only its two host descriptor buffers — no device
    // memory, no pinned memory, no fence.
    EXPECT_EQ(alloc.hostAllocs, 2);
    EXPECT_EQ(alloc.hostFrees, 2);
    EXPECT_EQ(alloc.hipAllocs, 0);
    EXPECT_EQ(alloc.pinnedAllocs, 0);
}

void TestConstructionRejectsBadCapacity() {
    CountingAllocator alloc;
    EXPECT_EXCEPTION(ImageBatchVarShape(0, alloc), eStatusType::INVALID_VALUE);
    EXPECT_EXCEPTION(ImageBatchVarShape(-3, alloc), eStatusType::INVALID_VALUE);
}

// =============================================================================
// pushBack — basic
// =============================================================================

void TestPushBackSingle() {
    CountingAllocator alloc;
    ImageBatchVarShape batch(4, alloc);

    Image img = MakeFakeGpuImage(640, 480, FAKE_PTR_A);
    batch.pushBack(img);

    EXPECT_EQ(batch.numImages(), 1);
    EXPECT_EQ(batch[0].size().w, 640);
    EXPECT_EQ(batch[0].size().h, 480);
    EXPECT_EQ(AsInt(batch[0].format() == FMT_RGB8), 1);
}

void TestPushBackMultipleHeterogeneousSizes() {
    CountingAllocator alloc;
    ImageBatchVarShape batch(4, alloc);

    batch.pushBack(MakeFakeGpuImage(640, 480, FAKE_PTR_A));
    batch.pushBack(MakeFakeGpuImage(320, 240, FAKE_PTR_B));
    batch.pushBack(MakeFakeGpuImage(800, 200, FAKE_PTR_C));

    EXPECT_EQ(batch.numImages(), 3);
    EXPECT_EQ(batch.maxSize().w, 800);
    EXPECT_EQ(batch.maxSize().h, 480);
    EXPECT_EQ(AsInt(batch.uniqueFormat() == FMT_RGB8), 1);
}

void TestPushBackIteratorRange() {
    CountingAllocator alloc;
    ImageBatchVarShape batch(8, alloc);

    std::vector<Image> imgs;
    imgs.push_back(MakeFakeGpuImage(100, 100, FAKE_PTR_A));
    imgs.push_back(MakeFakeGpuImage(200, 200, FAKE_PTR_B));
    imgs.push_back(MakeFakeGpuImage(300, 300, FAKE_PTR_C));

    batch.pushBack(imgs.begin(), imgs.end());

    EXPECT_EQ(batch.numImages(), 3);
    EXPECT_EQ(batch.maxSize().w, 300);
}

// =============================================================================
// pushBack — validation
// =============================================================================

void TestPushBackCapacityOverflow() {
    CountingAllocator alloc;
    ImageBatchVarShape batch(2, alloc);

    batch.pushBack(MakeFakeGpuImage(64, 64, FAKE_PTR_A));
    batch.pushBack(MakeFakeGpuImage(64, 64, FAKE_PTR_B));

    EXPECT_EXCEPTION(batch.pushBack(MakeFakeGpuImage(64, 64, FAKE_PTR_C)), eStatusType::OUT_OF_BOUNDS);
}

void TestPushBackHostImageRejected() {
    CountingAllocator alloc;
    ImageBatchVarShape batch(4, alloc);

    Image cpuImg = MakeFakeHostImage(64, 64, FAKE_PTR_A, FMT_U8);
    EXPECT_EXCEPTION(batch.pushBack(cpuImg), eStatusType::INVALID_VALUE);
}

void TestPushBackGpuImageRejectedOnCpuBatch() {
    CountingAllocator alloc;
    ImageBatchVarShape batch(4, alloc, eDeviceType::CPU);

    Image gpuImg = MakeFakeGpuImage(64, 64, FAKE_PTR_A);
    EXPECT_EXCEPTION(batch.pushBack(gpuImg), eStatusType::INVALID_VALUE);
}

// Note: pushBack's single-plane validation is defense-in-depth — Image's own
// exportData() (image.cpp:118) currently hardcodes numPlanes=1 regardless of
// the underlying buffer, so the public API can't construct a multi-plane Image
// for this guard to fire on. The test would need to be revisited when planar
// formats land in Image itself.

void TestPushBackRangeRollbackOnFailure() {
    CountingAllocator alloc;
    ImageBatchVarShape batch(8, alloc);

    // Pre-populate so we can confirm the rollback restores exactly the
    // pre-call state, not just back to zero.
    batch.pushBack(MakeFakeGpuImage(100, 100, FAKE_PTR_A));
    EXPECT_EQ(batch.numImages(), 1);

    // Mid-range CPU image — should rollback the partially-pushed entries.
    std::vector<Image> imgs;
    imgs.push_back(MakeFakeGpuImage(200, 200, FAKE_PTR_B));
    imgs.push_back(MakeFakeHostImage(300, 300, FAKE_PTR_C));  // Will throw.

    EXPECT_EXCEPTION(batch.pushBack(imgs.begin(), imgs.end()), eStatusType::INVALID_VALUE);

    // Pre-call state is intact: 1 image, original maxSize.
    EXPECT_EQ(batch.numImages(), 1);
    EXPECT_EQ(batch.maxSize().w, 100);
}

void TestPushBackRangeOverflowPrechecked() {
    CountingAllocator alloc;
    ImageBatchVarShape batch(2, alloc);

    std::vector<Image> imgs;
    imgs.push_back(MakeFakeGpuImage(10, 10, FAKE_PTR_A));
    imgs.push_back(MakeFakeGpuImage(20, 20, FAKE_PTR_B));
    imgs.push_back(MakeFakeGpuImage(30, 30, FAKE_PTR_C));  // 3rd overflows capacity 2.

    EXPECT_EXCEPTION(batch.pushBack(imgs.begin(), imgs.end()), eStatusType::OUT_OF_BOUNDS);
    // Pre-checked: nothing was pushed.
    EXPECT_EQ(batch.numImages(), 0);
}

// =============================================================================
// popBack / clear
// =============================================================================

void TestPopBack() {
    CountingAllocator alloc;
    ImageBatchVarShape batch(4, alloc);

    batch.pushBack(MakeFakeGpuImage(100, 100, FAKE_PTR_A));
    batch.pushBack(MakeFakeGpuImage(200, 200, FAKE_PTR_B));
    batch.popBack();

    EXPECT_EQ(batch.numImages(), 1);
    // maxSize was reset on pop; the rescan should drop back to 100.
    EXPECT_EQ(batch.maxSize().w, 100);
}

void TestPopBackMultiple() {
    CountingAllocator alloc;
    ImageBatchVarShape batch(4, alloc);

    batch.pushBack(MakeFakeGpuImage(100, 100, FAKE_PTR_A));
    batch.pushBack(MakeFakeGpuImage(200, 200, FAKE_PTR_B));
    batch.pushBack(MakeFakeGpuImage(300, 300, FAKE_PTR_C));
    batch.popBack(2);

    EXPECT_EQ(batch.numImages(), 1);
    EXPECT_EQ(batch.maxSize().w, 100);
}

void TestPopBackUnderflow() {
    CountingAllocator alloc;
    ImageBatchVarShape batch(4, alloc);
    batch.pushBack(MakeFakeGpuImage(100, 100, FAKE_PTR_A));

    EXPECT_EXCEPTION(batch.popBack(2), eStatusType::OUT_OF_BOUNDS);
    // State preserved.
    EXPECT_EQ(batch.numImages(), 1);
}

void TestClearAndReuse() {
    CountingAllocator alloc;
    ImageBatchVarShape batch(4, alloc);

    batch.pushBack(MakeFakeGpuImage(100, 100, FAKE_PTR_A));
    batch.pushBack(MakeFakeGpuImage(200, 200, FAKE_PTR_B));
    batch.clear();

    EXPECT_EQ(batch.numImages(), 0);
    EXPECT_EQ(batch.maxSize().w, 0);
    EXPECT_EQ(AsInt(batch.uniqueFormat() == FMT_NONE), 1);

    // Reuse after clear.
    batch.pushBack(MakeFakeGpuImage(50, 50, FAKE_PTR_C, FMT_U8));
    EXPECT_EQ(batch.numImages(), 1);
    EXPECT_EQ(AsInt(batch.uniqueFormat() == FMT_U8), 1);
}

// =============================================================================
// uniqueFormat / maxSize cache
// =============================================================================

void TestUniqueFormatHomogeneous() {
    CountingAllocator alloc;
    ImageBatchVarShape batch(4, alloc);
    batch.pushBack(MakeFakeGpuImage(64, 64, FAKE_PTR_A));
    batch.pushBack(MakeFakeGpuImage(128, 128, FAKE_PTR_B));
    EXPECT_EQ(AsInt(batch.uniqueFormat() == FMT_RGB8), 1);
}

void TestUniqueFormatHeterogeneous() {
    CountingAllocator alloc;
    ImageBatchVarShape batch(4, alloc);
    batch.pushBack(MakeFakeGpuImage(64, 64, FAKE_PTR_A));
    batch.pushBack(MakeFakeGpuImage(64, 64, FAKE_PTR_B, FMT_RGBA8));
    EXPECT_EQ(AsInt(batch.uniqueFormat() == FMT_NONE), 1);
}

void TestUniqueFormatEmptyBatch() {
    CountingAllocator alloc;
    ImageBatchVarShape batch(4, alloc);
    EXPECT_EQ(AsInt(batch.uniqueFormat() == FMT_NONE), 1);
    EXPECT_EQ(batch.maxSize().w, 0);
    EXPECT_EQ(batch.maxSize().h, 0);
}

// =============================================================================
// exportData
// =============================================================================

// exportData tests use the default allocator instead of CountingAllocator
// because they exercise the real H2D hipMemcpyAsync, which requires the
// device-side buffer to be a real hipMalloc'd pointer.

void TestExportDataEmpty() {
    ImageBatchVarShape batch(4);

    auto data = batch.exportData(0);
    EXPECT_EQ(data.numImages(), 0);
    EXPECT_EQ(data.maxSize().w, 0);
    EXPECT_EQ(data.maxSize().h, 0);
    EXPECT_EQ(AsInt(data.uniqueFormat() == FMT_NONE), 1);
    EXPECT_EQ(AsInt(data.device()), AsInt(eDeviceType::GPU));
}

void TestExportDataMetadata() {
    ImageBatchVarShape batch(4);
    batch.pushBack(MakeFakeGpuImage(640, 480, FAKE_PTR_A));
    batch.pushBack(MakeFakeGpuImage(320, 240, FAKE_PTR_B));

    auto data = batch.exportData(0);
    EXPECT_EQ(data.numImages(), 2);
    EXPECT_EQ(data.maxSize().w, 640);
    EXPECT_EQ(data.maxSize().h, 480);
    EXPECT_EQ(AsInt(data.uniqueFormat() == FMT_RGB8), 1);
    EXPECT_EQ(AsInt(data.imageList() != nullptr), 1);
    EXPECT_EQ(AsInt(data.formatList() != nullptr), 1);
    EXPECT_EQ(AsInt(data.hostFormatList() != nullptr), 1);
    // Pinned host mirror format entries are immediately host-readable.
    EXPECT_EQ(AsInt(data.hostFormatList()[0] == FMT_RGB8), 1);
    EXPECT_EQ(AsInt(data.hostFormatList()[1] == FMT_RGB8), 1);
}

void TestExportDataCastRoundTrip() {
    ImageBatchVarShape batch(4);
    batch.pushBack(MakeFakeGpuImage(64, 64, FAKE_PTR_A));

    auto hipData = batch.exportData<ImageBatchVarShapeDataStridedHip>(0);
    EXPECT_EQ(hipData.numImages(), 1);
    EXPECT_EQ(AsInt(hipData.device()), AsInt(eDeviceType::GPU));

    // Cast through the base reference: succeeds for compatible kinds, nullopt
    // for the host-resident leaf.
    const ImageBatchData& base = hipData;
    EXPECT_EQ(AsInt(base.cast<ImageBatchVarShapeDataStridedHip>().has_value()), 1);
    EXPECT_EQ(AsInt(base.cast<ImageBatchVarShapeDataStridedHost>().has_value()), 0);
}

// =============================================================================
// CPU path
// =============================================================================

void TestCpuPushBackAndCaches() {
    CountingAllocator alloc;
    ImageBatchVarShape batch(4, alloc, eDeviceType::CPU);

    batch.pushBack(MakeFakeHostImage(640, 480, FAKE_PTR_A));
    batch.pushBack(MakeFakeHostImage(320, 240, FAKE_PTR_B));
    batch.pushBack(MakeFakeHostImage(800, 200, FAKE_PTR_C));

    EXPECT_EQ(batch.numImages(), 3);
    EXPECT_EQ(batch.maxSize().w, 800);
    EXPECT_EQ(batch.maxSize().h, 480);
    EXPECT_EQ(AsInt(batch.uniqueFormat() == FMT_RGB8), 1);

    batch.popBack(2);
    EXPECT_EQ(batch.numImages(), 1);
    EXPECT_EQ(batch.maxSize().w, 640);

    batch.clear();
    EXPECT_EQ(batch.numImages(), 0);
    EXPECT_EQ(batch.maxSize().w, 0);
    EXPECT_EQ(AsInt(batch.uniqueFormat() == FMT_NONE), 1);
}

// CPU exportData performs no H2D copy, so it runs against the CountingAllocator's
// malloc-backed host buffers without needing a real device.
void TestExportDataHost() {
    CountingAllocator alloc;
    ImageBatchVarShape batch(4, alloc, eDeviceType::CPU);
    batch.pushBack(MakeFakeHostImage(640, 480, FAKE_PTR_A));
    batch.pushBack(MakeFakeHostImage(320, 240, FAKE_PTR_B));

    auto data = batch.exportData<ImageBatchVarShapeDataStridedHost>(0);
    EXPECT_EQ(data.numImages(), 2);
    EXPECT_EQ(AsInt(data.device()), AsInt(eDeviceType::CPU));
    EXPECT_EQ(data.maxSize().w, 640);
    EXPECT_EQ(data.maxSize().h, 480);
    EXPECT_EQ(AsInt(data.uniqueFormat() == FMT_RGB8), 1);
    EXPECT_EQ(AsInt(data.imageList() != nullptr), 1);
    EXPECT_EQ(AsInt(data.formatList() != nullptr), 1);
    // formatList and hostFormatList alias the same host allocation for CPU batches.
    EXPECT_EQ(AsInt(data.formatList() == data.hostFormatList()), 1);
    EXPECT_EQ(AsInt(data.hostFormatList()[0] == FMT_RGB8), 1);
    // Host descriptor table is directly readable: per-image dimensions match.
    EXPECT_EQ(static_cast<int32_t>(data.imageList()[0].planes[0].width), 640);
    EXPECT_EQ(static_cast<int32_t>(data.imageList()[1].planes[0].width), 320);
    EXPECT_EQ(AsInt(data.imageList()[0].planes[0].basePtr == FAKE_PTR_A), 1);
}

void TestExportDataHostCastRoundTrip() {
    CountingAllocator alloc;
    ImageBatchVarShape batch(4, alloc, eDeviceType::CPU);
    batch.pushBack(MakeFakeHostImage(64, 64, FAKE_PTR_A));

    ImageBatchVarShapeDataStrided data = batch.exportData(0);
    // A CPU snapshot casts to the host leaf but not the device leaf.
    EXPECT_EQ(AsInt(data.cast<ImageBatchVarShapeDataStridedHost>().has_value()), 1);
    EXPECT_EQ(AsInt(data.cast<ImageBatchVarShapeDataStridedHip>().has_value()), 0);
}

// =============================================================================
// Move semantics
// =============================================================================

void TestMoveConstruction() {
    CountingAllocator alloc;
    {
        ImageBatchVarShape src(4, alloc);
        src.pushBack(MakeFakeGpuImage(100, 100, FAKE_PTR_A));
        src.pushBack(MakeFakeGpuImage(200, 200, FAKE_PTR_B));

        ImageBatchVarShape dst(std::move(src));
        EXPECT_EQ(dst.numImages(), 2);
        EXPECT_EQ(dst.maxSize().w, 200);

        // Source is valid-but-empty; destructor must not double-free.
        EXPECT_EQ(src.numImages(), 0);
        EXPECT_EQ(src.capacity(), 0);
    }
    // Exactly one set of allocations should have been freed.
    EXPECT_EQ(alloc.hipAllocs, alloc.hipFrees);
    EXPECT_EQ(alloc.pinnedAllocs, alloc.pinnedFrees);
}

// =============================================================================
// Iterator
// =============================================================================

void TestIteratorRangeFor() {
    CountingAllocator alloc;
    ImageBatchVarShape batch(4, alloc);
    batch.pushBack(MakeFakeGpuImage(100, 100, FAKE_PTR_A));
    batch.pushBack(MakeFakeGpuImage(200, 200, FAKE_PTR_B));
    batch.pushBack(MakeFakeGpuImage(300, 300, FAKE_PTR_C));

    int32_t expectedW = 100;
    int32_t count = 0;
    for (const Image& img : batch) {
        EXPECT_EQ(img.size().w, expectedW);
        expectedW += 100;
        ++count;
    }
    EXPECT_EQ(count, 3);
}

}  // namespace

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;
    TEST_CASES_BEGIN();

    TEST_CASE(TestConstruction());
    TEST_CASE(TestConstructionCpu());
    TEST_CASE(TestConstructionRejectsBadCapacity());

    TEST_CASE(TestPushBackSingle());
    TEST_CASE(TestPushBackMultipleHeterogeneousSizes());
    TEST_CASE(TestPushBackIteratorRange());

    TEST_CASE(TestPushBackCapacityOverflow());
    TEST_CASE(TestPushBackHostImageRejected());
    TEST_CASE(TestPushBackGpuImageRejectedOnCpuBatch());
    TEST_CASE(TestPushBackRangeRollbackOnFailure());
    TEST_CASE(TestPushBackRangeOverflowPrechecked());

    TEST_CASE(TestPopBack());
    TEST_CASE(TestPopBackMultiple());
    TEST_CASE(TestPopBackUnderflow());
    TEST_CASE(TestClearAndReuse());

    TEST_CASE(TestUniqueFormatHomogeneous());
    TEST_CASE(TestUniqueFormatHeterogeneous());
    TEST_CASE(TestUniqueFormatEmptyBatch());

    TEST_CASE(TestExportDataEmpty());
    TEST_CASE(TestExportDataMetadata());
    TEST_CASE(TestExportDataCastRoundTrip());

    TEST_CASE(TestCpuPushBackAndCaches());
    TEST_CASE(TestExportDataHost());
    TEST_CASE(TestExportDataHostCastRoundTrip());

    TEST_CASE(TestMoveConstruction());

    TEST_CASE(TestIteratorRangeFor());

    TEST_CASES_END();
}
