/**
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <common/conversion_helpers.hpp>
#include <core/detail/swizzling.hpp>
#include <core/detail/type_traits.hpp>
#include <core/wrappers/image_wrapper.hpp>
#include <op_cvt_color.hpp>

#include "test_helpers.hpp"

using namespace roccv;
using namespace roccv::tests;

namespace {

template <typename T, eSwizzle S, typename BT = detail::BaseType<T>>
std::vector<BT> GoldenReorder(std::vector<BT>& input, int samples, int width, int height) {
    ImageWrapper<T> inputWrap(input, samples, width, height);
    std::vector<BT> output(samples * width * height * detail::NumElements<T>);
    ImageWrapper<T> outputWrap(output, samples, width, height);

    for (int b = 0; b < samples; b++) {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                outputWrap.at(b, y, x, 0) = detail::Swizzle<S>(inputWrap.at(b, y, x, 0));
            }
        }
    }

    return output;
}

template <typename T, eSwizzle S, typename BT = detail::BaseType<T>>
std::vector<BT> GoldenYUVToRGB(std::vector<BT>& input, int samples, int width, int height, float delta) {
    ImageWrapper<T> inputWrap(input, samples, width, height);
    std::vector<BT> output(samples * width * height * detail::NumElements<T>);
    ImageWrapper<T> outputWrap(output, samples, width, height);

    for (int b = 0; b < samples; b++) {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                T val = inputWrap.at(b, y, x, 0);
                float3 valF = detail::StaticCast<float3>(val);

                // Convert from YUV to RGB
                float3 rgb = make_float3(valF.x + (valF.z - delta) * 1.140f,                                // R
                                         valF.x + (valF.y - delta) * -0.395f + (valF.z - delta) * -0.581f,  // G
                                         valF.x + (valF.y - delta) * 2.032f);                               // B

                // Saturate cast to type T (this clamps to proper ranges) and swizzle to either RGB/BGR
                outputWrap.at(b, y, x, 0) = detail::Swizzle<S>(detail::SaturateCast<T>(rgb));
            }
        }
    }

    return output;
}

template <typename T, eSwizzle S, typename BT = detail::BaseType<T>>
std::vector<BT> GoldenRGBToYUV(std::vector<BT>& input, int samples, int width, int height, float delta) {
    ImageWrapper<T> inputWrap(input, samples, width, height);
    std::vector<BT> output(samples * width * height * detail::NumElements<T>);
    ImageWrapper<T> outputWrap(output, samples, width, height);

    for (int b = 0; b < samples; b++) {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                // Ensure input order is grabbed in RGB layout
                T val = detail::Swizzle<S>(inputWrap.at(b, y, x, 0));
                float3 valF = detail::StaticCast<float3>(val);

                float luminance = valF.x * 0.299f + valF.y * 0.587f + valF.z * 0.114f;
                float cr = (valF.x - luminance) * 0.877f + delta;
                float cb = (valF.z - luminance) * 0.492f + delta;

                float3 yuv = make_float3(luminance, cb, cr);

                outputWrap.at(b, y, x, 0) = detail::SaturateCast<T>(yuv);
            }
        }
    }

    return output;
}

template <typename T, eSwizzle S, typename BT = detail::BaseType<T>>
std::vector<BT> GoldenRGBToGrayscale(std::vector<BT>& input, int samples, int width, int height) {
    ImageWrapper<T> inputWrap(input, samples, width, height);
    std::vector<BT> output(samples * width * height);

    // Output must always be uchar1 for grayscale
    ImageWrapper<uchar1> outputWrap(output, samples, width, height);

    for (int b = 0; b < samples; b++) {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                // Grab input and Swizzle to ensure it is in RGB order
                T inVal = detail::Swizzle<S>(inputWrap.at(b, y, x, 0));
                float3 inValF = detail::StaticCast<float3>(inVal);

                // Calculate luminance
                float luminance = inValF.x * 0.299f + inValF.y * 0.587f + inValF.z * 0.114f;

                outputWrap.at(b, y, x, 0) = detail::SaturateCast<uchar1>(luminance);
            }
        }
    }

    return output;
}

/**
 * @brief Golden model for the Cvt Color operator.
 *
 * @tparam T The image's pixel datatype.
 * @tparam BT The image's base datatype.
 * @param input Input image data.
 * @param samples Number of samples in the batch.
 * @param width Width of each image in the batch.
 * @param height Height of each image in the batch.
 * @param code Color conversion code to use.
 * @return A vector containing the results of the convert color operator.
 */
template <typename T, typename BT = detail::BaseType<T>>
std::vector<BT> GoldenCvtColor(std::vector<BT>& input, int samples, int width, int height, eColorConversionCode code) {
    // clang-format off
    switch (code) {
        case eColorConversionCode::COLOR_RGB2YUV:  return GoldenRGBToYUV<T, eSwizzle::XYZW>(input, samples, width, height, 128.0f);
        case eColorConversionCode::COLOR_BGR2YUV:  return GoldenRGBToYUV<T, eSwizzle::ZYXW>(input, samples, width, height, 128.0f);
        case eColorConversionCode::COLOR_YUV2RGB:  return GoldenYUVToRGB<T, eSwizzle::XYZW>(input, samples, width, height, 128.0f);
        case eColorConversionCode::COLOR_YUV2BGR:  return GoldenYUVToRGB<T, eSwizzle::ZYXW>(input, samples, width, height, 128.0f);
        case eColorConversionCode::COLOR_RGB2GRAY: return GoldenRGBToGrayscale<T, eSwizzle::XYZW>(input, samples, width, height);
        case eColorConversionCode::COLOR_BGR2GRAY: return GoldenRGBToGrayscale<T, eSwizzle::ZYXW>(input, samples, width, height);
        case eColorConversionCode::COLOR_RGB2BGR:
        case eColorConversionCode::COLOR_BGR2RGB:  return GoldenReorder<T, eSwizzle::ZYXW>(input, samples, width, height);
        default: throw std::runtime_error("Unsupported color conversion code");
    }
    // clang-format on
}

/**
 * @brief Compares the Golden model against the CPU/GPU implementations of roccv::CvtColor.
 *
 * @tparam T Image pixel dtype.
 * @tparam BT Image base dtype.
 * @param samples Number of images in the batch.
 * @param width Image width.
 * @param height Image height.
 * @param inFmt Image input format.
 * @param outFmt Image output format.
 * @param code Color conversion code.
 * @param device The device to run conformance tests on.
 */
template <typename T, typename BT = detail::BaseType<T>>
void TestCorrectness(int samples, int width, int height, ImageFormat inFmt, ImageFormat outFmt,
                     eColorConversionCode code, eDeviceType device) {
    Tensor inputTensor(samples, {width, height}, inFmt, device);
    Tensor outputTensor(samples, {width, height}, outFmt, device);

    std::vector<BT> inputData(samples * width * height * inFmt.channels());
    FillVector(inputData);
    CopyVectorIntoTensor(inputTensor, inputData);

    hipStream_t stream;
    HIP_VALIDATE_NO_ERRORS(hipStreamCreate(&stream));
    CvtColor op;
    op(stream, inputTensor, outputTensor, code, device);
    HIP_VALIDATE_NO_ERRORS(hipStreamSynchronize(stream));
    HIP_VALIDATE_NO_ERRORS(hipStreamDestroy(stream));
    std::vector<BT> outputActual(outputTensor.shape().size());
    CopyTensorIntoVector(outputActual, outputTensor);

    std::vector<BT> outputGolden = GoldenCvtColor<T>(inputData, samples, width, height, code);

    CompareVectorsNear(outputActual, outputGolden);
}
}  // namespace

eTestStatusType test_op_cvt_color(int argc, char** argv) {
    TEST_CASES_BEGIN();

    // // GPU correctness tests
    TEST_CASE(TestCorrectness<uchar3>(1, 480, 360, FMT_RGB8, FMT_RGB8, COLOR_RGB2YUV, eDeviceType::GPU));
    TEST_CASE(TestCorrectness<uchar3>(2, 480, 120, FMT_RGB8, FMT_RGB8, COLOR_BGR2YUV, eDeviceType::GPU));
    TEST_CASE(TestCorrectness<uchar3>(5, 360, 360, FMT_RGB8, FMT_RGB8, COLOR_YUV2RGB, eDeviceType::GPU));
    TEST_CASE(TestCorrectness<uchar3>(3, 134, 360, FMT_RGB8, FMT_RGB8, COLOR_YUV2BGR, eDeviceType::GPU));
    TEST_CASE(TestCorrectness<uchar3>(7, 134, 360, FMT_RGB8, FMT_RGB8, COLOR_RGB2BGR, eDeviceType::GPU));
    TEST_CASE(TestCorrectness<uchar3>(6, 134, 360, FMT_RGB8, FMT_RGB8, COLOR_BGR2RGB, eDeviceType::GPU));
    TEST_CASE(TestCorrectness<uchar3>(9, 480, 120, FMT_RGB8, FMT_U8, COLOR_RGB2GRAY, eDeviceType::GPU));
    TEST_CASE(TestCorrectness<uchar3>(8, 134, 360, FMT_RGB8, FMT_U8, COLOR_BGR2GRAY, eDeviceType::GPU));

    // // CPU correctness tests
    TEST_CASE(TestCorrectness<uchar3>(8, 480, 360, FMT_RGB8, FMT_RGB8, COLOR_RGB2YUV, eDeviceType::CPU));
    TEST_CASE(TestCorrectness<uchar3>(2, 480, 120, FMT_RGB8, FMT_RGB8, COLOR_BGR2YUV, eDeviceType::CPU));
    TEST_CASE(TestCorrectness<uchar3>(5, 360, 360, FMT_RGB8, FMT_RGB8, COLOR_YUV2RGB, eDeviceType::CPU));
    TEST_CASE(TestCorrectness<uchar3>(3, 134, 360, FMT_RGB8, FMT_RGB8, COLOR_YUV2BGR, eDeviceType::CPU));
    TEST_CASE(TestCorrectness<uchar3>(7, 134, 360, FMT_RGB8, FMT_RGB8, COLOR_RGB2BGR, eDeviceType::CPU));
    TEST_CASE(TestCorrectness<uchar3>(6, 134, 360, FMT_RGB8, FMT_RGB8, COLOR_BGR2RGB, eDeviceType::CPU));
    TEST_CASE(TestCorrectness<uchar3>(9, 480, 120, FMT_RGB8, FMT_U8, COLOR_RGB2GRAY, eDeviceType::CPU));
    TEST_CASE(TestCorrectness<uchar3>(3, 134, 360, FMT_RGB8, FMT_U8, COLOR_BGR2GRAY, eDeviceType::CPU));

    TEST_CASES_END();
}
