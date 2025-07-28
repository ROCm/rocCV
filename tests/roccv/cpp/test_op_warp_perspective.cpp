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

#include <core/detail/casting.hpp>
#include <core/detail/type_traits.hpp>
#include <core/wrappers/interpolation_wrapper.hpp>
#include <op_warp_perspective.hpp>

#include "math_utils.hpp"
#include "test_helpers.hpp"

using namespace roccv;
using namespace roccv::tests;

namespace {
template <typename T, eBorderType BorderType, eInterpolationType InterpType>
std::vector<detail::BaseType<T>> GoldenWarpPerspective(std::vector<detail::BaseType<T>>& input,
                                                       const std::array<float, 9>& mat, bool isInverted, int batchSize,
                                                       Size2D inputSize, Size2D outputSize, float4 borderValue) {
    // Create interpolation wrapper for input vector
    InterpolationWrapper<T, BorderType, InterpType> inputWrap((BorderWrapper<T, BorderType>(
        ImageWrapper<T>(input, batchSize, inputSize.w, inputSize.h), detail::RangeCast<T>(borderValue))));

    // Create ImageWrapper for output vector. We also need to create said output vector.
    std::vector<detail::BaseType<T>> output(batchSize * outputSize.w * outputSize.h * detail::NumElements<T>);
    ImageWrapper<T> outputWrap(output, batchSize, outputSize.w, outputSize.h);

    // If given matrix is not the inverted representation of the transformation, we have to invert it first (since we
    // transform from output -> input).
    std::optional<std::array<float, 9>> invMat = std::make_optional(mat);
    if (!isInverted) {
        invMat = MatInv(mat);
    }

    if (!invMat.has_value()) {
        throw std::runtime_error("The given matrix could not be inverted.");
    }

    // Iterate through the output wrapper
    for (int b = 0; b < outputWrap.batches(); b++) {
        for (int y = 0; y < outputWrap.height(); y++) {
            for (int x = 0; x < outputWrap.width(); x++) {
                // Get transformed input point by multiplying by the given perspective transformation matrix
                Point2D inputCoord =
                    MatTransform((Point2D){static_cast<float>(x), static_cast<float>(y)}, invMat.value());
                outputWrap.at(b, y, x, 0) = inputWrap.at(b, inputCoord.y, inputCoord.x, 0);
            }
        }
    }

    return output;
}

template <typename T, eBorderType BorderType, eInterpolationType InterpType>
void TestCorrectness(int batchSize, Size2D inputSize, Size2D outputSize, ImageFormat format, bool isInverted,
                     std::array<float, 9> mat, float4 borderValue, eDeviceType device) {
    using BT = detail::BaseType<T>;

    // Generate input data
    std::vector<BT> input(batchSize * inputSize.w * inputSize.h * format.channels());
    FillVector(input);

    // Create input and output tensors for rocCV result
    Tensor inputTensor(batchSize, inputSize, format, device);
    Tensor outputTensor(batchSize, outputSize, format, device);

    // Copy input data into input tensor
    CopyVectorIntoTensor(inputTensor, input);

    // Copy mat into PerspectiveMatrix for rocCV op
    PerspectiveTransform transMat;
    for (int i = 0; i < 9; i++) {
        transMat[i] = mat[i];
    }

    hipStream_t stream;
    HIP_VALIDATE_NO_ERRORS(hipStreamCreate(&stream));
    WarpPerspective op;
    op(stream, inputTensor, outputTensor, transMat, isInverted, InterpType, BorderType, borderValue, device);
    HIP_VALIDATE_NO_ERRORS(hipStreamSynchronize(stream));
    HIP_VALIDATE_NO_ERRORS(hipStreamDestroy(stream));

    // Copy output tensor into host-allocated output vector
    std::vector<BT> actualOutput(outputTensor.shape().size());
    CopyTensorIntoVector(actualOutput, outputTensor);

    // Determine golden results
    std::vector<BT> goldenOutput = GoldenWarpPerspective<T, BorderType, InterpType>(input, mat, isInverted, batchSize,
                                                                                    inputSize, outputSize, borderValue);

    // Compare output results
    CompareVectorsNear(actualOutput, goldenOutput);
}

// Some pre-defined transformation matrices to use for our test cases. Should not randomly generate these since we want
// to ensure we're doing transformations which make sense.
// clang-format off
static const std::array<float, 9> MAT_IDENTITY =            {1, 0,    0,  0,    1,    0,  0,      0,      1};
static const std::array<float, 9> MAT_TRANSLATE =           {1, 0,    30, 0,    1,    50, 0,      0,      1};
static const std::array<float, 9> MAT_SCALE =               {2, 0,    0,  0,    0.5f, 0,  0,      0,      1};
static const std::array<float, 9> MAT_HORIZONTAL_SHEAR =    {1, 1.5f, 0,  0,    1,    0,  0,      0,      1};
static const std::array<float, 9> MAT_VERTICAL_SHEAR =      {1, 0,    0,  0.7f, 1,    0,  0,      0,      1};
static const std::array<float, 9> MAT_PERSPECTIVE_SKEW =    {1, 0,    0,  0,    1,    0,  0.001f, 0.001f, 1};
// clang-format on

}  // namespace

eTestStatusType test_op_warp_perspective(int argc, char** argv) {
    TEST_CASES_BEGIN();

    // clang-format off
    
    // U8
    TEST_CASE((TestCorrectness<uchar1, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_LINEAR>(1, {20, 30}, {20, 30}, FMT_U8,        false, MAT_IDENTITY,            make_float4(0.0f, 0.0f, 0.0f, 1.0f), eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<uchar3, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_LINEAR>(3, {20, 30}, {56, 85}, FMT_RGB8,      true,  MAT_TRANSLATE,           make_float4(0.0f, 0.0f, 0.0f, 1.0f), eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<uchar4, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_LINEAR>(5, {40, 10}, {34, 86}, FMT_RGBA8,     false, MAT_SCALE,               make_float4(0.0f, 0.0f, 0.0f, 1.0f), eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<uchar1, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_LINEAR>(1, {20, 30}, {20, 30}, FMT_U8,        true,  MAT_HORIZONTAL_SHEAR,    make_float4(0.0f, 0.0f, 0.0f, 1.0f), eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<uchar3, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_LINEAR>(3, {20, 30}, {56, 85}, FMT_RGB8,      false, MAT_VERTICAL_SHEAR,      make_float4(0.0f, 0.0f, 0.0f, 1.0f), eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<uchar4, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_LINEAR>(5, {40, 10}, {34, 86}, FMT_RGBA8,     true,  MAT_PERSPECTIVE_SKEW,    make_float4(0.0f, 0.0f, 0.0f, 1.0f), eDeviceType::GPU)));

    // F32
    TEST_CASE((TestCorrectness<float1, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_LINEAR>(1, {20, 30}, {20, 30}, FMT_F32,       false, MAT_IDENTITY,            make_float4(0.0f, 0.0f, 0.0f, 1.0f), eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<float3, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_LINEAR>(3, {20, 30}, {56, 85}, FMT_RGBf32,    true,  MAT_TRANSLATE,           make_float4(0.0f, 0.0f, 0.0f, 1.0f), eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<float4, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_LINEAR>(5, {40, 10}, {34, 86}, FMT_RGBAf32,   false, MAT_SCALE,               make_float4(0.0f, 0.0f, 0.0f, 1.0f), eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<float1, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_LINEAR>(1, {20, 30}, {20, 30}, FMT_F32,       true,  MAT_HORIZONTAL_SHEAR,    make_float4(0.0f, 0.0f, 0.0f, 1.0f), eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<float3, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_LINEAR>(3, {20, 30}, {56, 85}, FMT_RGBf32,    false, MAT_VERTICAL_SHEAR,      make_float4(0.0f, 0.0f, 0.0f, 1.0f), eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<float4, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_LINEAR>(5, {40, 10}, {34, 86}, FMT_RGBAf32,   true,  MAT_PERSPECTIVE_SKEW,    make_float4(0.0f, 0.0f, 0.0f, 1.0f), eDeviceType::GPU)));

    // clang-format on

    TEST_CASES_END();
}