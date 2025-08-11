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

#include <core/detail/type_traits.hpp>
#include <core/wrappers/interpolation_wrapper.hpp>
#include <op_warp_affine.hpp>

#include "math_utils.hpp"
#include "test_helpers.hpp"

using namespace roccv;
using namespace roccv::tests;

namespace {

/**
 * @brief Golden model for applying an affine transformation to an image. This is based on the golden model for a
 * perspective transformation, as an affine transformation is essentially a perspective transform with the last row
 * always set to [0, 0, 1].
 *
 * @tparam T Image datatype.
 * @tparam BorderType Border mode to use.
 * @tparam InterpType Interpolation mode to use.
 * @param input Vector containing input images.
 * @param mat The affine 2x3 transformation matrix in row-major order.
 * @param isInverted Whether the given matrix is inverted or not.
 * @param batchSize Number of images within the batch.
 * @param inputSize Width and height of the input images.
 * @param outputSize Width and height of the requested output images.
 * @param borderValue Border value to use as a fallback when going out of bounds.
 * @return An output vector containing the results.
 */
template <typename T, eBorderType BorderType, eInterpolationType InterpType>
std::vector<detail::BaseType<T>> GoldenWarpAffine(std::vector<detail::BaseType<T>>& input,
                                                  const std::array<float, 6>& mat, bool isInverted, int batchSize,
                                                  Size2D inputSize, Size2D outputSize, float4 borderValue) {
    // Create interpolation wrapper for input vector
    InterpolationWrapper<T, BorderType, InterpType> inputWrap((BorderWrapper<T, BorderType>(
        ImageWrapper<T>(input, batchSize, inputSize.w, inputSize.h), detail::RangeCast<T>(borderValue))));

    // Create ImageWrapper for output vector. We also need to create said output vector.
    std::vector<detail::BaseType<T>> output(batchSize * outputSize.w * outputSize.h * detail::NumElements<T>);
    ImageWrapper<T> outputWrap(output, batchSize, outputSize.w, outputSize.h);

    // Prepare the transformation matrix. An affine transform is effectively a 3x3 perspective transform with its last
    // row set to [0, 0, 1].
    std::array<float, 9> matPerspective = {mat[0], mat[1], mat[2], mat[3], mat[4], mat[5], 0, 0, 1};

    // If given matrix is not the inverted representation of the transformation, we have to invert it first (since we
    // transform from output -> input).
    std::optional<std::array<float, 9>> invMat = std::make_optional(matPerspective);
    if (!isInverted) {
        invMat = MatInv(matPerspective);
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

/**
 * @brief Tests correctness for the warp affine operator by comparing roccv::WarpAffine results with the defined golden
 * model.
 *
 * @tparam T Image datatype.
 * @tparam BorderType Border mode to use.
 * @tparam InterpType Interpolation mode to use.
 * @param batchSize Number of images within the batch.
 * @param inputSize Width and height of the input images.
 * @param outputSize Width and height of the requested output images.
 * @param format Format of the images (must match with T).
 * @param isInverted Whether the given matrix is inverted or not.
 * @param mat The affine 2x3 transformation matrix in row-major order.
 * @param borderValue Border value to use as a fallback when going out of bounds.
 * @param device The device to run the roccv::WarpAffine operator on.
 */
template <typename T, eBorderType BorderType, eInterpolationType InterpType>
void TestCorrectness(int batchSize, Size2D inputSize, Size2D outputSize, ImageFormat format, bool isInverted,
                     std::array<float, 6> mat, float4 borderValue, eDeviceType device) {
    using BT = detail::BaseType<T>;

    // Generate input data
    std::vector<BT> input(batchSize * inputSize.w * inputSize.h * format.channels());
    FillVector(input);

    // Create input and output tensors for rocCV result
    Tensor inputTensor(batchSize, inputSize, format, device);
    Tensor outputTensor(batchSize, outputSize, format, device);

    // Copy input data into input tensor
    CopyVectorIntoTensor(inputTensor, input);

    // Copy mat into AffineTransform matrix for rocCV op
    AffineTransform transMat;
    for (int i = 0; i < 6; i++) {
        transMat[i] = mat[i];
    }

    hipStream_t stream;
    HIP_VALIDATE_NO_ERRORS(hipStreamCreate(&stream));
    WarpAffine op;
    op(stream, inputTensor, outputTensor, transMat, isInverted, InterpType, BorderType, borderValue, device);
    HIP_VALIDATE_NO_ERRORS(hipStreamSynchronize(stream));
    HIP_VALIDATE_NO_ERRORS(hipStreamDestroy(stream));

    // Copy output tensor into host-allocated output vector
    std::vector<BT> actualOutput(outputTensor.shape().size());
    CopyTensorIntoVector(actualOutput, outputTensor);

    // Determine golden results
    std::vector<BT> goldenOutput = GoldenWarpAffine<T, BorderType, InterpType>(input, mat, isInverted, batchSize,
                                                                               inputSize, outputSize, borderValue);

    // Compare output results. Results are only accurate up to 1E-5 with affine warp.
    CompareVectorsNear(actualOutput, goldenOutput, 1E-5);
}

// Some common pre-defined affine transforms to test various functionality
// clang-format off
static const std::array<float, 6> MAT_IDENTITY =    {1,     0, 0,     0, 1, 0};
static const std::array<float, 6> MAT_REFLECT =     {-1.0f, 0, 0,     0, 1, 0};
static const std::array<float, 6> MAT_TRANSLATE =   {1,     0, 10.0f, 0, 1, -30.0f};
static const std::array<float, 6> MAT_SCALE =       {2.0f,  0, 0,     0, 1, 0};
// clang-format on

}  // namespace

eTestStatusType test_op_warp_affine(int argc, char** argv) {
    TEST_CASES_BEGIN();

    // clang-format off
    // GPU Tests
    // U8
    TEST_CASE((TestCorrectness<uchar1, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_LINEAR>(1, {20, 30}, {20, 30}, FMT_U8,        false, MAT_IDENTITY,            make_float4(0.0f, 0.0f, 0.0f, 1.0f), eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<uchar3, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_LINEAR>(3, {20, 30}, {56, 85}, FMT_RGB8,      true,  MAT_TRANSLATE,           make_float4(0.0f, 0.0f, 0.0f, 1.0f), eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<uchar4, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_LINEAR>(5, {40, 10}, {34, 86}, FMT_RGBA8,     false, MAT_SCALE,               make_float4(0.0f, 0.0f, 0.0f, 1.0f), eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<uchar1, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_LINEAR>(1, {20, 30}, {20, 30}, FMT_U8,        true,  MAT_REFLECT,             make_float4(0.0f, 0.0f, 0.0f, 1.0f), eDeviceType::GPU)));

    // S8
    TEST_CASE((TestCorrectness<char1, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_LINEAR>(1, {20, 30}, {20, 30}, FMT_S8,        false, MAT_IDENTITY,            make_float4(0.0f, 0.0f, 0.0f, 1.0f), eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<char1, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_LINEAR>(1, {20, 30}, {20, 30}, FMT_S8,        true,  MAT_REFLECT,             make_float4(0.0f, 0.0f, 0.0f, 1.0f), eDeviceType::GPU)));

    // U16
    TEST_CASE((TestCorrectness<ushort1, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_LINEAR>(1, {20, 30}, {20, 30}, FMT_U16,        false, MAT_IDENTITY,            make_float4(0.0f, 0.0f, 0.0f, 1.0f), eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<ushort3, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_LINEAR>(3, {20, 30}, {56, 85}, FMT_RGB16,      true,  MAT_TRANSLATE,           make_float4(0.0f, 0.0f, 0.0f, 1.0f), eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<ushort4, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_LINEAR>(5, {40, 10}, {34, 86}, FMT_RGBA16,     false, MAT_SCALE,               make_float4(0.0f, 0.0f, 0.0f, 1.0f), eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<ushort1, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_LINEAR>(1, {20, 30}, {20, 30}, FMT_U16,        true,  MAT_REFLECT,             make_float4(0.0f, 0.0f, 0.0f, 1.0f), eDeviceType::GPU)));

    // S16
    TEST_CASE((TestCorrectness<short1, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_LINEAR>(1, {20, 30}, {20, 30}, FMT_S16,        false, MAT_IDENTITY,            make_float4(0.0f, 0.0f, 0.0f, 1.0f), eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<short1, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_LINEAR>(1, {20, 30}, {20, 30}, FMT_S16,        true,  MAT_REFLECT,             make_float4(0.0f, 0.0f, 0.0f, 1.0f), eDeviceType::GPU)));

    // U32
    TEST_CASE((TestCorrectness<uint1, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_LINEAR>(1, {20, 30}, {20, 30}, FMT_U32,        false, MAT_IDENTITY,            make_float4(0.0f, 0.0f, 0.0f, 1.0f), eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<uint3, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_LINEAR>(3, {20, 30}, {56, 85}, FMT_RGB32,      true,  MAT_TRANSLATE,           make_float4(0.0f, 0.0f, 0.0f, 1.0f), eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<uint4, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_LINEAR>(5, {40, 10}, {34, 86}, FMT_RGBA32,     false, MAT_SCALE,               make_float4(0.0f, 0.0f, 0.0f, 1.0f), eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<uint1, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_LINEAR>(1, {20, 30}, {20, 30}, FMT_U32,        true,  MAT_REFLECT,             make_float4(0.0f, 0.0f, 0.0f, 1.0f), eDeviceType::GPU)));

    // S32
    TEST_CASE((TestCorrectness<int1, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_LINEAR>(1, {20, 30}, {20, 30}, FMT_S32,        false, MAT_IDENTITY,            make_float4(0.0f, 0.0f, 0.0f, 1.0f), eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<int1, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_LINEAR>(1, {20, 30}, {20, 30}, FMT_S32,        true,  MAT_REFLECT,             make_float4(0.0f, 0.0f, 0.0f, 1.0f), eDeviceType::GPU)));

    // F32
    TEST_CASE((TestCorrectness<float1, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_LINEAR>(1, {20, 30}, {20, 30}, FMT_F32,       false, MAT_IDENTITY,            make_float4(0.0f, 0.0f, 0.0f, 1.0f), eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<float3, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_LINEAR>(3, {20, 30}, {56, 85}, FMT_RGBf32,    true,  MAT_TRANSLATE,           make_float4(0.0f, 0.0f, 0.0f, 1.0f), eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<float4, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_LINEAR>(5, {40, 10}, {34, 86}, FMT_RGBAf32,   false, MAT_SCALE,               make_float4(0.0f, 0.0f, 0.0f, 1.0f), eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<float1, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_LINEAR>(1, {20, 30}, {20, 30}, FMT_F32,       true,  MAT_REFLECT,             make_float4(0.0f, 0.0f, 0.0f, 1.0f), eDeviceType::GPU)));

    // F64
    TEST_CASE((TestCorrectness<double1, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_LINEAR>(1, {20, 30}, {20, 30}, FMT_F64,         false, MAT_IDENTITY,            make_float4(0.0f, 0.0f, 0.0f, 1.0f), eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<double3, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_LINEAR>(3, {20, 30}, {56, 85}, FMT_RGBf64,      true,  MAT_TRANSLATE,           make_float4(0.0f, 0.0f, 0.0f, 1.0f), eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<double4, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_LINEAR>(5, {40, 10}, {34, 86}, FMT_RGBAf64,     false, MAT_SCALE,               make_float4(0.0f, 0.0f, 0.0f, 1.0f), eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<double1, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_LINEAR>(1, {20, 30}, {20, 30}, FMT_F64,         true,  MAT_REFLECT,             make_float4(0.0f, 0.0f, 0.0f, 1.0f), eDeviceType::GPU)));

    // CPU Tests
    // U8
    TEST_CASE((TestCorrectness<uchar1, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_LINEAR>(1, {20, 30}, {20, 30}, FMT_U8,        false, MAT_IDENTITY,            make_float4(0.0f, 0.0f, 0.0f, 1.0f), eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<uchar3, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_LINEAR>(3, {20, 30}, {56, 85}, FMT_RGB8,      true,  MAT_TRANSLATE,           make_float4(0.0f, 0.0f, 0.0f, 1.0f), eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<uchar4, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_LINEAR>(5, {40, 10}, {34, 86}, FMT_RGBA8,     false, MAT_SCALE,               make_float4(0.0f, 0.0f, 0.0f, 1.0f), eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<uchar1, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_LINEAR>(1, {20, 30}, {20, 30}, FMT_U8,        true,  MAT_REFLECT,             make_float4(0.0f, 0.0f, 0.0f, 1.0f), eDeviceType::CPU)));

    // S8
    TEST_CASE((TestCorrectness<char1, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_LINEAR>(1, {20, 30}, {20, 30}, FMT_S8,        false, MAT_IDENTITY,            make_float4(0.0f, 0.0f, 0.0f, 1.0f), eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<char1, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_LINEAR>(1, {20, 30}, {20, 30}, FMT_S8,        true,  MAT_REFLECT,             make_float4(0.0f, 0.0f, 0.0f, 1.0f), eDeviceType::CPU)));

    // U16
    TEST_CASE((TestCorrectness<ushort1, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_LINEAR>(1, {20, 30}, {20, 30}, FMT_U16,        false, MAT_IDENTITY,            make_float4(0.0f, 0.0f, 0.0f, 1.0f), eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<ushort3, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_LINEAR>(3, {20, 30}, {56, 85}, FMT_RGB16,      true,  MAT_TRANSLATE,           make_float4(0.0f, 0.0f, 0.0f, 1.0f), eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<ushort4, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_LINEAR>(5, {40, 10}, {34, 86}, FMT_RGBA16,     false, MAT_SCALE,               make_float4(0.0f, 0.0f, 0.0f, 1.0f), eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<ushort1, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_LINEAR>(1, {20, 30}, {20, 30}, FMT_U16,        true,  MAT_REFLECT,             make_float4(0.0f, 0.0f, 0.0f, 1.0f), eDeviceType::CPU)));

    // S16
    TEST_CASE((TestCorrectness<short1, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_LINEAR>(1, {20, 30}, {20, 30}, FMT_S16,        false, MAT_IDENTITY,            make_float4(0.0f, 0.0f, 0.0f, 1.0f), eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<short1, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_LINEAR>(1, {20, 30}, {20, 30}, FMT_S16,        true,  MAT_REFLECT,             make_float4(0.0f, 0.0f, 0.0f, 1.0f), eDeviceType::CPU)));

    // U32
    TEST_CASE((TestCorrectness<uint1, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_LINEAR>(1, {20, 30}, {20, 30}, FMT_U32,        false, MAT_IDENTITY,            make_float4(0.0f, 0.0f, 0.0f, 1.0f), eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<uint3, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_LINEAR>(3, {20, 30}, {56, 85}, FMT_RGB32,      true,  MAT_TRANSLATE,           make_float4(0.0f, 0.0f, 0.0f, 1.0f), eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<uint4, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_LINEAR>(5, {40, 10}, {34, 86}, FMT_RGBA32,     false, MAT_SCALE,               make_float4(0.0f, 0.0f, 0.0f, 1.0f), eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<uint1, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_LINEAR>(1, {20, 30}, {20, 30}, FMT_U32,        true,  MAT_REFLECT,             make_float4(0.0f, 0.0f, 0.0f, 1.0f), eDeviceType::CPU)));

    // S32
    TEST_CASE((TestCorrectness<int1, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_LINEAR>(1, {20, 30}, {20, 30}, FMT_S32,        false, MAT_IDENTITY,            make_float4(0.0f, 0.0f, 0.0f, 1.0f), eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<int1, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_LINEAR>(1, {20, 30}, {20, 30}, FMT_S32,        true,  MAT_REFLECT,             make_float4(0.0f, 0.0f, 0.0f, 1.0f), eDeviceType::CPU)));

    // F32
    TEST_CASE((TestCorrectness<float1, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_LINEAR>(1, {20, 30}, {20, 30}, FMT_F32,       false, MAT_IDENTITY,            make_float4(0.0f, 0.0f, 0.0f, 1.0f), eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<float3, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_LINEAR>(3, {20, 30}, {56, 85}, FMT_RGBf32,    true,  MAT_TRANSLATE,           make_float4(0.0f, 0.0f, 0.0f, 1.0f), eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<float4, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_LINEAR>(5, {40, 10}, {34, 86}, FMT_RGBAf32,   false, MAT_SCALE,               make_float4(0.0f, 0.0f, 0.0f, 1.0f), eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<float1, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_LINEAR>(1, {20, 30}, {20, 30}, FMT_F32,       true,  MAT_REFLECT,             make_float4(0.0f, 0.0f, 0.0f, 1.0f), eDeviceType::CPU)));

    // F64
    TEST_CASE((TestCorrectness<double1, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_LINEAR>(1, {20, 30}, {20, 30}, FMT_F64,         false, MAT_IDENTITY,            make_float4(0.0f, 0.0f, 0.0f, 1.0f), eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<double3, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_LINEAR>(3, {20, 30}, {56, 85}, FMT_RGBf64,      true,  MAT_TRANSLATE,           make_float4(0.0f, 0.0f, 0.0f, 1.0f), eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<double4, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_LINEAR>(5, {40, 10}, {34, 86}, FMT_RGBAf64,     false, MAT_SCALE,               make_float4(0.0f, 0.0f, 0.0f, 1.0f), eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<double1, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_LINEAR>(1, {20, 30}, {20, 30}, FMT_F64,         true,  MAT_REFLECT,             make_float4(0.0f, 0.0f, 0.0f, 1.0f), eDeviceType::CPU)));
    // clang-format on

    TEST_CASES_END();
}