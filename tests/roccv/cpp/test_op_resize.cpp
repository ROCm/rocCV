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

#include <core/detail/casting.hpp>
#include <core/detail/type_traits.hpp>
#include <core/wrappers/interpolation_wrapper.hpp>
#include <op_resize.hpp>

#include "test_helpers.hpp"

using namespace roccv;
using namespace roccv::tests;

namespace {

/**
 * @brief Golden model for the resize operator.
 *
 * @tparam T The image pixel's datatype (e.g. uchar3).
 * @tparam InterpType The interpolation type to use during resizing.
 * @tparam BT The base type of the image datatype (e.g. unsigned char).
 * @param input An input vector containing a batch of images.
 * @param batchSize The number of images in the batch.
 * @param inputSize The size of the input images.
 * @param outputSize The requested size of the output images.
 * @return A vector containing the data for the images resized to outputSize.
 */
template <typename T, eInterpolationType InterpType, typename BT = detail::BaseType<T>>
std::vector<BT> GoldenResize(std::vector<detail::BaseType<T>> &input, int batchSize, Size2D inputSize,
                             Size2D outputSize) {
    size_t numOutputElements = batchSize * outputSize.w * outputSize.h * detail::NumElements<T>;

    std::vector<detail::BaseType<T>> output(numOutputElements);
    ImageWrapper<T> outputWrap(output, batchSize, outputSize.w, outputSize.h);

    // Use the replicate (or clamping) border mode by default to handle out of bounds conditions with certain
    // interpolation modes.
    InterpolationWrapper<T, eBorderType::BORDER_TYPE_REPLICATE, InterpType> inputWrap(
        BorderWrapper<T, eBorderType::BORDER_TYPE_REPLICATE>(
            ImageWrapper<T>(input, batchSize, inputSize.w, inputSize.h), T{}));

    // Determine the scaling factor required to map from the output coordinates to the corresponding input coordinates
    // on both the x and y axes.
    float2 scaleRatio =
        make_float2(inputSize.w / static_cast<float>(outputSize.w), inputSize.h / static_cast<float>(outputSize.h));

    for (int b = 0; b < batchSize; b++) {
        for (int y = 0; y < outputSize.h; y++) {
            for (int x = 0; x < outputSize.w; x++) {
                // Map the destination coordinates to their corresponding source coordinates based on the calculated
                // scaling factors. 0.5 is added before the mapping to get the center of the destination pixel
                // coordinates. After scaling, subtract 0.5 to move the scaled input coordinates to their top-left pixel
                // representation.
                float srcX = ((x + 0.5f) * scaleRatio.x) - 0.5f;
                float srcY = ((y + 0.5f) * scaleRatio.y) - 0.5f;

                // Set the output value to the input value at the scaled coordinates. Since inputWrap is an
                // InterpolationWrapper, border conditions and interpolation are handled automatically.
                outputWrap.at(b, y, x, 0) = inputWrap.at(b, srcY, srcX, 0);
            }
        }
    }

    return output;
}

/**
 * @brief Compares the results of roccv::Resize and the golden model implementation.
 *
 * @tparam T The image's pixel datatype.
 * @tparam InterpType The interpolation type to use during resizing.
 * @tparam BT The image's base datatype.
 * @param batchSize The number of images in the batch.
 * @param inputSize The size of the input images.
 * @param outputSize The size of the output images.
 * @param format The image format to use (must match with the given type T).
 * @param device The device to run this correctness test on.
 * @throws std::runtime_error on test failure.
 */
template <typename T, eInterpolationType InterpType, typename BT = detail::BaseType<T>>
void TestCorrectness(int batchSize, Size2D inputSize, Size2D outputSize, ImageFormat format, eDeviceType device) {
    // Prepare tensors for roccv::Resize
    Tensor inputTensor(batchSize, inputSize, format, device);
    Tensor outputTensor(batchSize, outputSize, format, device);

    // Generate random input data and move into input tensor
    std::vector<BT> input(inputTensor.shape().size());
    FillVector(input);

    CopyVectorIntoTensor(inputTensor, input);

    // Run roccv::Resize
    hipStream_t stream;
    HIP_VALIDATE_NO_ERRORS(hipStreamCreate(&stream));
    Resize op;
    op(stream, inputTensor, outputTensor, InterpType, device);
    HIP_VALIDATE_NO_ERRORS(hipStreamSynchronize(stream));
    HIP_VALIDATE_NO_ERRORS(hipStreamDestroy(stream));

    // Move output tensor into host allocated output vector containing actual results
    std::vector<BT> actualOutput(outputTensor.shape().size());
    CopyTensorIntoVector(actualOutput, outputTensor);

    // Get golden results
    std::vector<BT> goldenOutput = GoldenResize<T, InterpType>(input, batchSize, inputSize, outputSize);

    CompareVectorsNear(actualOutput, goldenOutput);
}

}  // namespace

eTestStatusType test_op_resize(int argc, char **argv) {
    TEST_CASES_BEGIN();

    // clang-format off

    // GPU Tests

    // U8 - Linear interpolation
    TEST_CASE((TestCorrectness<uchar1, eInterpolationType::INTERP_TYPE_LINEAR>(1, {100, 50}, {200, 50}, FMT_U8, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<uchar3, eInterpolationType::INTERP_TYPE_LINEAR>(3, {100, 50}, {100, 50}, FMT_RGB8, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<uchar4, eInterpolationType::INTERP_TYPE_LINEAR>(5, {100, 50}, {50, 25}, FMT_RGBA8, eDeviceType::GPU)));

    // U8 - Nearest interpolation
    TEST_CASE((TestCorrectness<uchar1, eInterpolationType::INTERP_TYPE_NEAREST>(1, {100, 50}, {200, 50}, FMT_U8, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<uchar3, eInterpolationType::INTERP_TYPE_NEAREST>(3, {100, 50}, {100, 50}, FMT_RGB8, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<uchar4, eInterpolationType::INTERP_TYPE_NEAREST>(5, {100, 50}, {50, 25}, FMT_RGBA8, eDeviceType::GPU)));

    // F32 - Linear interpolation
    TEST_CASE((TestCorrectness<float1, eInterpolationType::INTERP_TYPE_LINEAR>(1, {100, 50}, {200, 50}, FMT_F32, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<float3, eInterpolationType::INTERP_TYPE_LINEAR>(3, {100, 50}, {100, 50}, FMT_RGBf32, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<float4, eInterpolationType::INTERP_TYPE_LINEAR>(5, {100, 50}, {50, 25}, FMT_RGBAf32, eDeviceType::GPU)));

    // F32 - Nearest interpolation
    TEST_CASE((TestCorrectness<float1, eInterpolationType::INTERP_TYPE_NEAREST>(1, {100, 50}, {200, 50}, FMT_F32, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<float3, eInterpolationType::INTERP_TYPE_NEAREST>(3, {100, 50}, {100, 50}, FMT_RGBf32, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<float4, eInterpolationType::INTERP_TYPE_NEAREST>(5, {100, 50}, {50, 25}, FMT_RGBAf32, eDeviceType::GPU)));

    // CPU Tests

    // U8 - Linear interpolation
    TEST_CASE((TestCorrectness<uchar1, eInterpolationType::INTERP_TYPE_LINEAR>(1, {100, 50}, {200, 50}, FMT_U8, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<uchar3, eInterpolationType::INTERP_TYPE_LINEAR>(3, {100, 50}, {100, 50}, FMT_RGB8, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<uchar4, eInterpolationType::INTERP_TYPE_LINEAR>(5, {100, 50}, {50, 25}, FMT_RGBA8, eDeviceType::CPU)));

    // U8 - Nearest interpolation
    TEST_CASE((TestCorrectness<uchar1, eInterpolationType::INTERP_TYPE_NEAREST>(1, {100, 50}, {200, 50}, FMT_U8, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<uchar3, eInterpolationType::INTERP_TYPE_NEAREST>(3, {100, 50}, {100, 50}, FMT_RGB8, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<uchar4, eInterpolationType::INTERP_TYPE_NEAREST>(5, {100, 50}, {50, 25}, FMT_RGBA8, eDeviceType::CPU)));

    // F32 - Linear interpolation
    TEST_CASE((TestCorrectness<float1, eInterpolationType::INTERP_TYPE_LINEAR>(1, {100, 50}, {200, 50}, FMT_F32, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<float3, eInterpolationType::INTERP_TYPE_LINEAR>(3, {100, 50}, {100, 50}, FMT_RGBf32, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<float4, eInterpolationType::INTERP_TYPE_LINEAR>(5, {100, 50}, {50, 25}, FMT_RGBAf32, eDeviceType::CPU)));

    // F32 - Nearest interpolation
    TEST_CASE((TestCorrectness<float1, eInterpolationType::INTERP_TYPE_NEAREST>(1, {100, 50}, {200, 50}, FMT_F32, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<float3, eInterpolationType::INTERP_TYPE_NEAREST>(3, {100, 50}, {100, 50}, FMT_RGBf32, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<float4, eInterpolationType::INTERP_TYPE_NEAREST>(5, {100, 50}, {50, 25}, FMT_RGBAf32, eDeviceType::CPU)));
    // clang-format on

    TEST_CASES_END();
}
