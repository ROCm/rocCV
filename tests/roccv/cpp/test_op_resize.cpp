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

#include <algorithm>
#include <core/detail/casting.hpp>
#include <core/detail/type_traits.hpp>
#include <core/wrappers/interpolation_wrapper.hpp>
#include <filesystem>
#include <iostream>
#include <op_resize.hpp>
#include <opencv2/opencv.hpp>

#include "test_helpers.hpp"

using namespace roccv;
using namespace roccv::tests;

namespace {
template <typename T, eInterpolationType InterpType, typename BT = detail::BaseType<T>>
std::vector<BT> GoldenResize(std::vector<detail::BaseType<T>> &input, int batchSize, Size2D inputSize,
                             Size2D outputSize) {
    size_t numOutputElements = batchSize * outputSize.w * outputSize.h * detail::NumElements<T>;

    std::vector<detail::BaseType<T>> output(numOutputElements);
    ImageWrapper<T> outputWrap(output, batchSize, outputSize.w, outputSize.h);

    InterpolationWrapper<T, eBorderType::BORDER_TYPE_CONSTANT, InterpType> inputWrap(
        BorderWrapper<T, eBorderType::BORDER_TYPE_CONSTANT>(ImageWrapper<T>(input, batchSize, inputSize.w, inputSize.h),
                                                            T{}));

    float2 scaleRatio =
        make_float2(inputSize.w / static_cast<float>(outputSize.w), inputSize.h / static_cast<float>(outputSize.h));

    for (int b = 0; b < batchSize; b++) {
        for (int y = 0; y < outputSize.h; y++) {
            for (int x = 0; x < outputSize.w; x++) {
                float srcX = std::min<float>(((x + 0.5f) * scaleRatio.x) - 0.5f, inputSize.w - 1);
                float srcY = std::min<float>(((y + 0.5f) * scaleRatio.y) - 0.5f, inputSize.h - 1);

                outputWrap.at(b, y, x, 0) = inputWrap.at(b, srcY, srcX, 0);
            }
        }
    }

    return output;
}

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
    TEST_CASE((TestCorrectness<uchar1, eInterpolationType::INTERP_TYPE_LINEAR>(1, {100, 50}, {200, 50}, FMT_U8, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<uchar3, eInterpolationType::INTERP_TYPE_LINEAR>(1, {100, 50}, {100, 50}, FMT_RGB8, eDeviceType::GPU)));
    // clang-format on

    TEST_CASES_END();
}
