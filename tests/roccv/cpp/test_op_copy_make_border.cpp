/*
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include <core/detail/type_traits.hpp>
#include <core/wrappers/border_wrapper.hpp>
#include <op_copy_make_border.hpp>

#include "test_helpers.hpp"

using namespace roccv;
using namespace roccv::tests;

namespace {
template <typename T, eBorderType BorderType, typename BT = detail::BaseType<T>>
std::vector<BT> GoldenCopyMakeBorder(std::vector<BT> input, int batchSize, Size2D inputSize, Size2D outputSize, int top,
                                     int left, float4 borderValue) {
    int channels = detail::NumElements<T>;

    // Convert border value into the type of the image
    T borderVal = detail::RangeCast<T>(borderValue);

    // Wrap the input images in a BorderWrapper to handle out of bounds image behavior. The BorderWrapper has already
    // been tested in another test so it can be used reliably.
    BorderWrapper<T, BorderType> inputWrap(ImageWrapper<T>(input, batchSize, inputSize.w, inputSize.h), borderVal);

    std::vector<BT> output(batchSize * outputSize.h * outputSize.w * channels);
    ImageWrapper<T> outputWrap(output, batchSize, outputSize.w, outputSize.h);

    for (int b = 0; b < batchSize; b++) {
        for (int y = 0; y < outputSize.h; y++) {
            for (int x = 0; x < outputSize.w; x++) {
                // CopyMakeBorder essentially copies the input image into the output image with a specified left and top
                // shift on the x and y coordinates respectively. Out of bounds behavior will be handled by the
                // BorderWrapper wrapping the input image.
                outputWrap.at(b, y, x, 0) = inputWrap.at(b, y - top, x - left, 0);
            }
        }
    }

    return output;
}

template <typename T, eBorderType BorderType, typename BT = detail::BaseType<T>>
void TestCorrectness(int batchSize, Size2D inputSize, Size2D outputSize, ImageFormat format, int top, int left,
                     float4 borderValue, eDeviceType device) {
    Tensor inputTensor(batchSize, inputSize, format, device);
    Tensor outputTensor(batchSize, outputSize, format, device);

    // Generate random input data
    std::vector<BT> input(batchSize * inputSize.h * inputSize.w * format.channels());
    FillVector(input);

    CopyVectorIntoTensor(inputTensor, input);

    // Run roccv::CopyMakeBorder to get actual operator results
    hipStream_t stream;
    HIP_VALIDATE_NO_ERRORS(hipStreamCreate(&stream));
    CopyMakeBorder op;
    op(stream, inputTensor, outputTensor, top, left, BorderType, borderValue, device);
    HIP_VALIDATE_NO_ERRORS(hipStreamSynchronize(stream));
    HIP_VALIDATE_NO_ERRORS(hipStreamDestroy(stream));

    // Copy results into host output vector
    std::vector<BT> actualOutput(batchSize * outputSize.h * outputSize.w * format.channels());
    CopyTensorIntoVector(actualOutput, outputTensor);

    // Run golden model to obtain golden vector
    std::vector<BT> goldenOutput =
        GoldenCopyMakeBorder<T, BorderType>(input, batchSize, inputSize, outputSize, top, left, borderValue);

    // Compare actual results with golden results
    CompareVectors(actualOutput, goldenOutput);
}
}  // namespace

eTestStatusType test_op_copy_make_border(int argc, char** argv) {
    TEST_CASES_BEGIN();

    // clang-format off

    // GPU Tests

    // U8
    TEST_CASE((TestCorrectness<uchar1, eBorderType::BORDER_TYPE_CONSTANT>(1, {40, 10}, {60, 20}, FMT_U8, 5, 10, make_float4(1.0f, 0.5f, 0.5f, 1.0f), eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<uchar3, eBorderType::BORDER_TYPE_REFLECT>(3, {40, 10}, {80, 25}, FMT_RGB8, 0, 5, make_float4(1.0f, 0.5f, 0.5f, 1.0f), eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<uchar3, eBorderType::BORDER_TYPE_REPLICATE>(5, {10, 5}, {10, 25}, FMT_RGB8, 10, 0, make_float4(1.0f, 0.5f, 0.5f, 1.0f), eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<uchar4, eBorderType::BORDER_TYPE_WRAP>(5, {10, 5}, {10, 25}, FMT_RGBA8, 10, 0, make_float4(1.0f, 0.5f, 0.5f, 1.0f), eDeviceType::GPU)));

    // S8
    TEST_CASE((TestCorrectness<char1, eBorderType::BORDER_TYPE_CONSTANT>(1, {40, 10}, {60, 20}, FMT_S8, 5, 10, make_float4(1.0f, 0.5f, 0.5f, 1.0f), eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<char3, eBorderType::BORDER_TYPE_REFLECT>(3, {40, 10}, {80, 25}, FMT_RGBs8, 0, 5, make_float4(1.0f, 0.5f, 0.5f, 1.0f), eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<char3, eBorderType::BORDER_TYPE_REPLICATE>(5, {10, 5}, {10, 25}, FMT_RGBs8, 10, 0, make_float4(1.0f, 0.5f, 0.5f, 1.0f), eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<char4, eBorderType::BORDER_TYPE_WRAP>(5, {10, 5}, {10, 25}, FMT_RGBAs8, 10, 0, make_float4(1.0f, 0.5f, 0.5f, 1.0f), eDeviceType::GPU)));

    // U16
    TEST_CASE((TestCorrectness<ushort1, eBorderType::BORDER_TYPE_CONSTANT>(1, {40, 10}, {60, 20}, FMT_U16, 5, 10, make_float4(1.0f, 0.5f, 0.5f, 1.0f), eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<ushort3, eBorderType::BORDER_TYPE_REFLECT>(3, {40, 10}, {80, 25}, FMT_RGB16, 0, 5, make_float4(1.0f, 0.5f, 0.5f, 1.0f), eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<ushort3, eBorderType::BORDER_TYPE_REPLICATE>(5, {10, 5}, {10, 25}, FMT_RGB16, 10, 0, make_float4(1.0f, 0.5f, 0.5f, 1.0f), eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<ushort4, eBorderType::BORDER_TYPE_WRAP>(5, {10, 5}, {10, 25}, FMT_RGBA16, 10, 0, make_float4(1.0f, 0.5f, 0.5f, 1.0f), eDeviceType::GPU)));

    // S16
    TEST_CASE((TestCorrectness<short1, eBorderType::BORDER_TYPE_CONSTANT>(1, {40, 10}, {60, 20}, FMT_S16, 5, 10, make_float4(1.0f, 0.5f, 0.5f, 1.0f), eDeviceType::GPU)));

    // U32
    TEST_CASE((TestCorrectness<uint1, eBorderType::BORDER_TYPE_CONSTANT>(1, {40, 10}, {60, 20}, FMT_U32, 5, 10, make_float4(1.0f, 0.5f, 0.5f, 1.0f), eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<uint3, eBorderType::BORDER_TYPE_REFLECT>(3, {40, 10}, {80, 25}, FMT_RGB32, 0, 5, make_float4(1.0f, 0.5f, 0.5f, 1.0f), eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<uint3, eBorderType::BORDER_TYPE_REPLICATE>(5, {10, 5}, {10, 25}, FMT_RGB32, 10, 0, make_float4(1.0f, 0.5f, 0.5f, 1.0f), eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<uint4, eBorderType::BORDER_TYPE_WRAP>(5, {10, 5}, {10, 25}, FMT_RGBA32, 10, 0, make_float4(1.0f, 0.5f, 0.5f, 1.0f), eDeviceType::GPU)));

    // S32
    TEST_CASE((TestCorrectness<int1, eBorderType::BORDER_TYPE_CONSTANT>(1, {40, 10}, {60, 20}, FMT_S32, 5, 10, make_float4(1.0f, 0.5f, 0.5f, 1.0f), eDeviceType::GPU)));

    // F32
    TEST_CASE((TestCorrectness<float1, eBorderType::BORDER_TYPE_CONSTANT>(1, {40, 10}, {60, 20}, FMT_F32, 5, 10, make_float4(1.0f, 0.5f, 0.5f, 1.0f), eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<float3, eBorderType::BORDER_TYPE_REFLECT>(3, {40, 10}, {80, 25}, FMT_RGBf32, 0, 5, make_float4(1.0f, 0.5f, 0.5f, 1.0f), eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<float3, eBorderType::BORDER_TYPE_REPLICATE>(5, {10, 5}, {10, 25}, FMT_RGBf32, 10, 0, make_float4(1.0f, 0.5f, 0.5f, 1.0f), eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<float4, eBorderType::BORDER_TYPE_WRAP>(5, {10, 5}, {10, 25}, FMT_RGBAf32, 10, 0, make_float4(1.0f, 0.5f, 0.5f, 1.0f), eDeviceType::GPU)));

    // F64
    TEST_CASE((TestCorrectness<double1, eBorderType::BORDER_TYPE_CONSTANT>(1, {40, 10}, {60, 20}, FMT_F64, 5, 10, make_float4(1.0f, 0.5f, 0.5f, 1.0f), eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<double3, eBorderType::BORDER_TYPE_REFLECT>(3, {40, 10}, {80, 25}, FMT_RGBf64, 0, 5, make_float4(1.0f, 0.5f, 0.5f, 1.0f), eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<double3, eBorderType::BORDER_TYPE_REPLICATE>(5, {10, 5}, {10, 25}, FMT_RGBf64, 10, 0, make_float4(1.0f, 0.5f, 0.5f, 1.0f), eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<double4, eBorderType::BORDER_TYPE_WRAP>(5, {10, 5}, {10, 25}, FMT_RGBAf64, 10, 0, make_float4(1.0f, 0.5f, 0.5f, 1.0f), eDeviceType::GPU)));


    // CPU Tests

    // U8
    TEST_CASE((TestCorrectness<uchar1, eBorderType::BORDER_TYPE_CONSTANT>(1, {40, 10}, {60, 20}, FMT_U8, 5, 10, make_float4(1.0f, 0.5f, 0.5f, 1.0f), eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<uchar3, eBorderType::BORDER_TYPE_REFLECT>(3, {40, 10}, {80, 25}, FMT_RGB8, 0, 5, make_float4(1.0f, 0.5f, 0.5f, 1.0f), eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<uchar3, eBorderType::BORDER_TYPE_REPLICATE>(5, {10, 5}, {10, 25}, FMT_RGB8, 10, 0, make_float4(1.0f, 0.5f, 0.5f, 1.0f), eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<uchar4, eBorderType::BORDER_TYPE_WRAP>(5, {10, 5}, {10, 25}, FMT_RGBA8, 10, 0, make_float4(1.0f, 0.5f, 0.5f, 1.0f), eDeviceType::CPU)));

    // S8
    TEST_CASE((TestCorrectness<char1, eBorderType::BORDER_TYPE_CONSTANT>(1, {40, 10}, {60, 20}, FMT_S8, 5, 10, make_float4(1.0f, 0.5f, 0.5f, 1.0f), eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<char3, eBorderType::BORDER_TYPE_REFLECT>(3, {40, 10}, {80, 25}, FMT_RGBs8, 0, 5, make_float4(1.0f, 0.5f, 0.5f, 1.0f), eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<char3, eBorderType::BORDER_TYPE_REPLICATE>(5, {10, 5}, {10, 25}, FMT_RGBs8, 10, 0, make_float4(1.0f, 0.5f, 0.5f, 1.0f), eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<char4, eBorderType::BORDER_TYPE_WRAP>(5, {10, 5}, {10, 25}, FMT_RGBAs8, 10, 0, make_float4(1.0f, 0.5f, 0.5f, 1.0f), eDeviceType::CPU)));

    // U16
    TEST_CASE((TestCorrectness<ushort1, eBorderType::BORDER_TYPE_CONSTANT>(1, {40, 10}, {60, 20}, FMT_U16, 5, 10, make_float4(1.0f, 0.5f, 0.5f, 1.0f), eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<ushort3, eBorderType::BORDER_TYPE_REFLECT>(3, {40, 10}, {80, 25}, FMT_RGB16, 0, 5, make_float4(1.0f, 0.5f, 0.5f, 1.0f), eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<ushort3, eBorderType::BORDER_TYPE_REPLICATE>(5, {10, 5}, {10, 25}, FMT_RGB16, 10, 0, make_float4(1.0f, 0.5f, 0.5f, 1.0f), eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<ushort4, eBorderType::BORDER_TYPE_WRAP>(5, {10, 5}, {10, 25}, FMT_RGBA16, 10, 0, make_float4(1.0f, 0.5f, 0.5f, 1.0f), eDeviceType::CPU)));

    // S16
    TEST_CASE((TestCorrectness<short1, eBorderType::BORDER_TYPE_CONSTANT>(1, {40, 10}, {60, 20}, FMT_S16, 5, 10, make_float4(1.0f, 0.5f, 0.5f, 1.0f), eDeviceType::CPU)));

    // U32
    TEST_CASE((TestCorrectness<uint1, eBorderType::BORDER_TYPE_CONSTANT>(1, {40, 10}, {60, 20}, FMT_U32, 5, 10, make_float4(1.0f, 0.5f, 0.5f, 1.0f), eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<uint3, eBorderType::BORDER_TYPE_REFLECT>(3, {40, 10}, {80, 25}, FMT_RGB32, 0, 5, make_float4(1.0f, 0.5f, 0.5f, 1.0f), eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<uint3, eBorderType::BORDER_TYPE_REPLICATE>(5, {10, 5}, {10, 25}, FMT_RGB32, 10, 0, make_float4(1.0f, 0.5f, 0.5f, 1.0f), eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<uint4, eBorderType::BORDER_TYPE_WRAP>(5, {10, 5}, {10, 25}, FMT_RGBA32, 10, 0, make_float4(1.0f, 0.5f, 0.5f, 1.0f), eDeviceType::CPU)));

    // S32
    TEST_CASE((TestCorrectness<int1, eBorderType::BORDER_TYPE_CONSTANT>(1, {40, 10}, {60, 20}, FMT_S32, 5, 10, make_float4(1.0f, 0.5f, 0.5f, 1.0f), eDeviceType::CPU)));

    // F32
    TEST_CASE((TestCorrectness<float1, eBorderType::BORDER_TYPE_CONSTANT>(1, {40, 10}, {60, 20}, FMT_F32, 5, 10, make_float4(1.0f, 0.5f, 0.5f, 1.0f), eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<float3, eBorderType::BORDER_TYPE_REFLECT>(3, {40, 10}, {80, 25}, FMT_RGBf32, 0, 5, make_float4(1.0f, 0.5f, 0.5f, 1.0f), eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<float3, eBorderType::BORDER_TYPE_REPLICATE>(5, {10, 5}, {10, 25}, FMT_RGBf32, 10, 0, make_float4(1.0f, 0.5f, 0.5f, 1.0f), eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<float4, eBorderType::BORDER_TYPE_WRAP>(5, {10, 5}, {10, 25}, FMT_RGBAf32, 10, 0, make_float4(1.0f, 0.5f, 0.5f, 1.0f), eDeviceType::CPU)));

    // F64
    TEST_CASE((TestCorrectness<double1, eBorderType::BORDER_TYPE_CONSTANT>(1, {40, 10}, {60, 20}, FMT_F64, 5, 10, make_float4(1.0f, 0.5f, 0.5f, 1.0f), eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<double3, eBorderType::BORDER_TYPE_REFLECT>(3, {40, 10}, {80, 25}, FMT_RGBf64, 0, 5, make_float4(1.0f, 0.5f, 0.5f, 1.0f), eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<double3, eBorderType::BORDER_TYPE_REPLICATE>(5, {10, 5}, {10, 25}, FMT_RGBf64, 10, 0, make_float4(1.0f, 0.5f, 0.5f, 1.0f), eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<double4, eBorderType::BORDER_TYPE_WRAP>(5, {10, 5}, {10, 25}, FMT_RGBAf64, 10, 0, make_float4(1.0f, 0.5f, 0.5f, 1.0f), eDeviceType::CPU)));

    // clang-format on

    TEST_CASES_END();
}