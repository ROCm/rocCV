/*
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include <core/hip_assert.h>
#include <hip/hip_runtime.h>

#include <core/detail/type_traits.hpp>
#include <core/image.hpp>
#include <core/image_batch_var_shape.hpp>
#include <core/image_data.hpp>
#include <core/image_format.hpp>
#include <core/wrappers/border_wrapper.hpp>
#include <core/wrappers/interpolation_wrapper.hpp>
#include <core/wrappers/image_batch_var_shape_wrapper.hpp>
#include <vector>

#include "test_helpers.hpp"

using namespace roccv;
using namespace roccv::tests;

namespace {

// Per-image copy kernel: writes dst[n,y,x,c] = src[n,y,x,c] for the image at batch index n.
// Each launch covers exactly one image — the host loop steps through n and resizes the
// grid to that image's dimensions, which avoids needing a max-bounds check inside the kernel.
template <typename T>
__global__ void VarShapeCopyKernel(ImageBatchVarShapeWrapper<T> src, ImageBatchVarShapeWrapper<T> dst, int32_t n) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dst.width(n) || y >= dst.height(n)) return;
    dst.at(n, y, x, 0) = src.at(n, y, x, 0);
}

// Roundtrip test: build a varshape batch from heterogeneous host pixel data, run the copy kernel from src varshape
// wrapper to dst varshape wrapper, read pixels back, and verify byte-equality with the original input.
template <typename T, typename BT = detail::BaseType<T>>
void TestRoundtripCopy(const std::vector<Size2D>& sizes, ImageFormat fmt) {
    const int channels = detail::NumElements<T>;
    const int32_t numImages = static_cast<int32_t>(sizes.size());

    // Generate per-image host pixel data.
    std::vector<std::vector<BT>> hostPixels(numImages);
    for (int32_t i = 0; i < numImages; ++i) {
        hostPixels[i].resize(static_cast<size_t>(sizes[i].w) * sizes[i].h * channels);
        FillVector(hostPixels[i], /*seed=*/static_cast<uint32_t>(0x1000 + i));
    }

    hipStream_t stream = nullptr;

    // Build source batch and copy host pixels in.
    ImageBatchVarShape srcBatch(numImages);
    std::vector<Image> srcImages;
    srcImages.reserve(numImages);
    for (int32_t i = 0; i < numImages; ++i) {
        srcImages.emplace_back(sizes[i], fmt);
        auto sd = srcImages[i].exportData<ImageDataStridedHip>();
        const ImagePlaneStrided& sp = sd.plane(0);
        const size_t rowBytes = static_cast<size_t>(sizes[i].w) * channels * sizeof(BT);
        HIP_VALIDATE_NO_ERRORS(hipMemcpy2DAsync(sp.basePtr, sp.rowStride, hostPixels[i].data(), rowBytes, rowBytes,
                                                sizes[i].h, hipMemcpyHostToDevice, stream));
        srcBatch.pushBack(srcImages[i]);
    }

    // Build destination batch with matching shapes.
    ImageBatchVarShape dstBatch(numImages);
    std::vector<Image> dstImages;
    dstImages.reserve(numImages);
    for (int32_t i = 0; i < numImages; ++i) {
        dstImages.emplace_back(sizes[i], fmt);
        dstBatch.pushBack(dstImages[i]);
    }

    auto srcData = srcBatch.exportData(stream);
    auto dstData = dstBatch.exportData(stream);
    ImageBatchVarShapeWrapper<T> srcWrap(srcData);
    ImageBatchVarShapeWrapper<T> dstWrap(dstData);

    // Launch one kernel per image (sizes vary so a single 3D launch can't bound y to per-image height cleanly).
    for (int32_t i = 0; i < numImages; ++i) {
        dim3 block(16, 16);
        dim3 grid((sizes[i].w + block.x - 1) / block.x, (sizes[i].h + block.y - 1) / block.y);
        VarShapeCopyKernel<T><<<grid, block, 0, stream>>>(srcWrap, dstWrap, i);
    }

    HIP_VALIDATE_NO_ERRORS(hipStreamSynchronize(stream));

    // Read back dst pixels and verify byte-for-byte against the original host input.
    for (int32_t i = 0; i < numImages; ++i) {
        std::vector<BT> dstHost(static_cast<size_t>(sizes[i].w) * sizes[i].h * channels);
        auto dd = dstImages[i].exportData<ImageDataStridedHip>();
        const ImagePlaneStrided& dp = dd.plane(0);
        const size_t rowBytes = static_cast<size_t>(sizes[i].w) * channels * sizeof(BT);
        HIP_VALIDATE_NO_ERRORS(hipMemcpy2D(dstHost.data(), rowBytes, dp.basePtr, dp.rowStride, rowBytes, sizes[i].h,
                                           hipMemcpyDeviceToHost));
        CompareVectors(dstHost, hostPixels[i]);
    }
}

// Border-composition test: write the BORDER_TYPE_CONSTANT fallback for every output pixel by reading from a coordinate
// that is guaranteed to be out of bounds for every image (-1, -1). Confirms BorderWrapper<B, ImageBatchVarShapeWrapper<T>>
// correctly delegates to width(n) / height(n) for per-image bounds; otherwise it would dereference invalid memory.
template <typename T>
__global__ void VarShapeBorderConstantKernel(
    BorderWrapper<eBorderType::BORDER_TYPE_CONSTANT, ImageBatchVarShapeWrapper<T>> src, ImageBatchVarShapeWrapper<T> dst,
    int32_t n) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dst.width(n) || y >= dst.height(n)) return;
    dst.at(n, y, x, 0) = src.at(n, -1, -1, 0);
}

template <typename T, typename BT = detail::BaseType<T>>
void TestBorderConstantComposition(const std::vector<Size2D>& sizes, ImageFormat fmt, T borderValue) {
    const int channels = detail::NumElements<T>;
    const int32_t numImages = static_cast<int32_t>(sizes.size());

    // Source pixels content doesn't matter — every read is forced OOB.
    ImageBatchVarShape srcBatch(numImages);
    ImageBatchVarShape dstBatch(numImages);
    std::vector<Image> srcImages, dstImages;
    srcImages.reserve(numImages);
    dstImages.reserve(numImages);
    for (int32_t i = 0; i < numImages; ++i) {
        srcImages.emplace_back(sizes[i], fmt);
        dstImages.emplace_back(sizes[i], fmt);
        srcBatch.pushBack(srcImages[i]);
        dstBatch.pushBack(dstImages[i]);
    }

    hipStream_t stream = nullptr;
    auto srcData = srcBatch.exportData(stream);
    auto dstData = dstBatch.exportData(stream);

    auto srcWrap = MakeBorderWrapper<eBorderType::BORDER_TYPE_CONSTANT>(ImageBatchVarShapeWrapper<T>(srcData), borderValue);
    ImageBatchVarShapeWrapper<T> dstWrap(dstData);

    for (int32_t i = 0; i < numImages; ++i) {
        dim3 block(16, 16);
        dim3 grid((sizes[i].w + block.x - 1) / block.x, (sizes[i].h + block.y - 1) / block.y);
        VarShapeBorderConstantKernel<T><<<grid, block, 0, stream>>>(srcWrap, dstWrap, i);
    }
    HIP_VALIDATE_NO_ERRORS(hipStreamSynchronize(stream));

    // Expect every output pixel of every image to equal borderValue.
    std::vector<BT> borderBytes(channels);
    for (int c = 0; c < channels; ++c) borderBytes[c] = detail::GetElement(borderValue, c);

    for (int32_t i = 0; i < numImages; ++i) {
        const size_t pixels = static_cast<size_t>(sizes[i].w) * sizes[i].h;
        std::vector<BT> dstHost(pixels * channels);
        auto dd = dstImages[i].exportData<ImageDataStridedHip>();
        const ImagePlaneStrided& dp = dd.plane(0);
        const size_t rowBytes = static_cast<size_t>(sizes[i].w) * channels * sizeof(BT);
        HIP_VALIDATE_NO_ERRORS(hipMemcpy2D(dstHost.data(), rowBytes, dp.basePtr, dp.rowStride, rowBytes, sizes[i].h,
                                           hipMemcpyDeviceToHost));
        std::vector<BT> expected(pixels * channels);
        for (size_t p = 0; p < pixels; ++p) {
            for (int c = 0; c < channels; ++c) expected[p * channels + c] = borderBytes[c];
        }
        CompareVectors(dstHost, expected);
    }
}

// Interpolation-composition test: NEAREST interpolation at integer coordinates is the identity, so a roundtrip copy
// via InterpolationWrapper<NEAREST, REPLICATE, VarShape> must equal the source. Confirms the full wrapper chain
// composes correctly over a VarShape backing.
template <typename T>
__global__ void VarShapeInterpNearestKernel(
    InterpolationWrapper<eInterpolationType::INTERP_TYPE_NEAREST,
                         BorderWrapper<eBorderType::BORDER_TYPE_REPLICATE, ImageBatchVarShapeWrapper<T>>>
        src,
    ImageBatchVarShapeWrapper<T> dst, int32_t n) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dst.width(n) || y >= dst.height(n)) return;
    dst.at(n, y, x, 0) = src.at(n, static_cast<float>(y), static_cast<float>(x), 0);
}

template <typename T, typename BT = detail::BaseType<T>>
void TestInterpolationNearestComposition(const std::vector<Size2D>& sizes, ImageFormat fmt) {
    const int channels = detail::NumElements<T>;
    const int32_t numImages = static_cast<int32_t>(sizes.size());

    std::vector<std::vector<BT>> hostPixels(numImages);
    for (int32_t i = 0; i < numImages; ++i) {
        hostPixels[i].resize(static_cast<size_t>(sizes[i].w) * sizes[i].h * channels);
        FillVector(hostPixels[i], static_cast<uint32_t>(0x2000 + i));
    }

    hipStream_t stream = nullptr;
    ImageBatchVarShape srcBatch(numImages);
    ImageBatchVarShape dstBatch(numImages);
    std::vector<Image> srcImages, dstImages;
    srcImages.reserve(numImages);
    dstImages.reserve(numImages);
    for (int32_t i = 0; i < numImages; ++i) {
        srcImages.emplace_back(sizes[i], fmt);
        auto sd = srcImages[i].exportData<ImageDataStridedHip>();
        const ImagePlaneStrided& sp = sd.plane(0);
        const size_t rowBytes = static_cast<size_t>(sizes[i].w) * channels * sizeof(BT);
        HIP_VALIDATE_NO_ERRORS(hipMemcpy2DAsync(sp.basePtr, sp.rowStride, hostPixels[i].data(), rowBytes, rowBytes,
                                                sizes[i].h, hipMemcpyHostToDevice, stream));
        srcBatch.pushBack(srcImages[i]);

        dstImages.emplace_back(sizes[i], fmt);
        dstBatch.pushBack(dstImages[i]);
    }

    auto srcData = srcBatch.exportData(stream);
    auto dstData = dstBatch.exportData(stream);

    auto srcWrap = MakeInterpolationWrapper<eInterpolationType::INTERP_TYPE_NEAREST>(
        MakeBorderWrapper<eBorderType::BORDER_TYPE_REPLICATE>(ImageBatchVarShapeWrapper<T>(srcData), T{}));
    ImageBatchVarShapeWrapper<T> dstWrap(dstData);

    for (int32_t i = 0; i < numImages; ++i) {
        dim3 block(16, 16);
        dim3 grid((sizes[i].w + block.x - 1) / block.x, (sizes[i].h + block.y - 1) / block.y);
        VarShapeInterpNearestKernel<T><<<grid, block, 0, stream>>>(srcWrap, dstWrap, i);
    }
    HIP_VALIDATE_NO_ERRORS(hipStreamSynchronize(stream));

    for (int32_t i = 0; i < numImages; ++i) {
        std::vector<BT> dstHost(static_cast<size_t>(sizes[i].w) * sizes[i].h * channels);
        auto dd = dstImages[i].exportData<ImageDataStridedHip>();
        const ImagePlaneStrided& dp = dd.plane(0);
        const size_t rowBytes = static_cast<size_t>(sizes[i].w) * channels * sizeof(BT);
        HIP_VALIDATE_NO_ERRORS(hipMemcpy2D(dstHost.data(), rowBytes, dp.basePtr, dp.rowStride, rowBytes, sizes[i].h,
                                           hipMemcpyDeviceToHost));
        CompareVectors(dstHost, hostPixels[i]);
    }
}

// Verify accessor surface: width(n), height(n), batches(), channels().
template <typename T>
void TestAccessors(const std::vector<Size2D>& sizes, ImageFormat fmt) {
    const int32_t numImages = static_cast<int32_t>(sizes.size());
    ImageBatchVarShape batch(numImages);
    std::vector<Image> handles;
    handles.reserve(numImages);
    for (int32_t i = 0; i < numImages; ++i) {
        handles.emplace_back(sizes[i], fmt);
        batch.pushBack(handles[i]);
    }
    auto data = batch.exportData(0);
    ImageBatchVarShapeWrapper<T> wrap(data);

    EXPECT_EQ(wrap.batches(), static_cast<int64_t>(numImages));
    EXPECT_EQ(wrap.channels(), static_cast<int64_t>(detail::NumElements<T>));
    // width/height are device pointers under the hood; reading them on host post-sync is safe because exportData
    // recorded a hipEvent that hipStreamSynchronize on the null stream above (implicit) drains. The descriptor table
    // lives in device memory though, so we round-trip the lookups through a tiny D->H pull via the wrapper's host
    // mirror path — here we just check that the construction succeeded; per-image width/height behavior is exercised
    // end-to-end by TestRoundtripCopy.
    HIP_VALIDATE_NO_ERRORS(hipStreamSynchronize(0));
}

}  // namespace

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;
    TEST_CASES_BEGIN();

    // Single-channel, heterogeneous sizes.
    TEST_CASE(TestRoundtripCopy<uchar1>({{16, 12}, {32, 24}, {7, 5}, {48, 9}}, FMT_U8));
    TEST_CASE(TestRoundtripCopy<float1>({{16, 12}, {32, 24}, {7, 5}, {48, 9}}, FMT_F32));

    // Multi-channel interleaved.
    TEST_CASE(TestRoundtripCopy<uchar3>({{16, 12}, {32, 24}, {7, 5}, {48, 9}}, FMT_RGB8));
    TEST_CASE(TestRoundtripCopy<uchar4>({{16, 12}, {32, 24}, {7, 5}, {48, 9}}, FMT_RGBA8));

    // Homogeneous batch — the wrapper should still work when all images share the same shape.
    TEST_CASE(TestRoundtripCopy<uchar4>({{64, 64}, {64, 64}, {64, 64}}, FMT_RGBA8));

    // Single image, large.
    TEST_CASE(TestRoundtripCopy<uchar3>({{128, 96}}, FMT_RGB8));

    TEST_CASE(TestAccessors<uchar3>({{16, 12}, {32, 24}, {7, 5}}, FMT_RGB8));

    // BorderWrapper composed over ImageBatchVarShapeWrapper: constant-fill via guaranteed-OOB read.
    TEST_CASE(
        TestBorderConstantComposition<uchar3>({{16, 12}, {32, 24}, {7, 5}}, FMT_RGB8, make_uchar3(0xAB, 0xCD, 0xEF)));
    TEST_CASE(TestBorderConstantComposition<uchar4>({{16, 12}, {32, 24}, {7, 5}}, FMT_RGBA8,
                                                    make_uchar4(0x12, 0x34, 0x56, 0x78)));

    // InterpolationWrapper<NEAREST> composed over ImageBatchVarShapeWrapper: integer-coord roundtrip is identity.
    TEST_CASE(TestInterpolationNearestComposition<uchar1>({{16, 12}, {32, 24}, {7, 5}}, FMT_U8));
    TEST_CASE(TestInterpolationNearestComposition<uchar3>({{16, 12}, {32, 24}, {7, 5}}, FMT_RGB8));
    TEST_CASE(TestInterpolationNearestComposition<uchar4>({{16, 12}, {32, 24}, {7, 5}}, FMT_RGBA8));

    TEST_CASES_END();
}
