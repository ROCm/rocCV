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
#include <cstdint>
#include <core/detail/casting.hpp>
#include <core/detail/type_traits.hpp>
#include <core/wrappers/image_wrapper.hpp>
#include <op_center_crop.hpp>
#include "test_helpers.hpp"

using namespace roccv;
using namespace roccv::tests;

namespace {

/**
 * @brief Verified golden C++ model for the center crop operation.
 *
 * @tparam T Vectorized datatype of the image's pixels.
 * @tparam BT Base type of the image's data.
 * @param[in] input Input vector containing image data.
 * @param[out] output Output vector containing cropped image data.
 * @param[in] batchSize The number of images in the batch.
 * @param[in] width Image width.
 * @param[in] height Image height.
 * @param[in] cropSize Cropped region descriptor
 * @return None.
 */
template <typename T, typename BT = detail::BaseType<T>>
void GenerateGoldenCrop(std::vector<BT>& input, std::vector<BT>& output, int32_t batchSize, int32_t width, int32_t height, Size2D cropSize) {
    // Wrap input/output vectors for simplified data access
    ImageWrapper<T> src(input, batchSize, width, height);
    ImageWrapper<T> dst(output, batchSize, cropSize.w, cropSize.h);

    int topLeftX = (width >> 1) - (cropSize.w >> 1);
    int topLeftY = (height >> 1) - (cropSize.h >> 1);
    for (int b = 0; b < batchSize; b++) {
        for (int y = 0; y < cropSize.h; y++) {
            for (int x = 0; x < cropSize.w; x++) {
                dst.at(b, y, x, 0) = detail::SaturateCast<T>(src.at(b, (y + topLeftY), (x + topLeftX), 0));
            }
        }
    }
}

/**
 * @brief Tests correctness of the center crop operator, comparing it against a generated golden result.
 *
 * @tparam T Underlying datatype of the image's pixels.
 * @tparam BT Base type of the image data.
 * @param[in] batchSize Number of images in the batch.
 * @param[in] width Width of each image in the batch.
 * @param[in] height Height of each image in the batch.
 * @param[in] cropSize Cropped region descriptor
 * @param[in] format Image format.
 * @param[in] device Device this correctness test should be run on.
 * @return None.
*/
template <typename T, typename BT = detail::BaseType<T>>
void TestCorrectness(int batchSize, int width, int height, Size2D cropSize, ImageFormat format, eDeviceType device) {
    // Create input and output tensor based on test parameters
    Tensor input(batchSize, {width, height}, format, device);
    Tensor output(batchSize, {cropSize.w, cropSize.h}, format, device);

    // Create a vector and fill it with random data.
    std::vector<BT> inputData(input.shape().size());
    FillVector(inputData);
    // Copy generated input data into input tensor
    CopyVectorIntoTensor(input, inputData);

    // Run roccv::CenterCrop operator
    hipStream_t stream;
    HIP_VALIDATE_NO_ERRORS(hipStreamCreate(&stream));
    CenterCrop op;
    op(stream, input, output, cropSize, device);
    HIP_VALIDATE_NO_ERRORS(hipStreamSynchronize(stream));
    HIP_VALIDATE_NO_ERRORS(hipStreamDestroy(stream));

    // Copy data from output tensor into a host allocated vector
    std::vector<BT> result(output.shape().size());
    CopyTensorIntoVector(result, output);

    // Calculate golden output reference
    std::vector<BT> ref(output.shape().size());
    GenerateGoldenCrop<T>(inputData, ref, batchSize, width, height, cropSize);

    // Compare data in actual output versus the generated golden reference image
    CompareVectors(result, ref);
}

/**
 * @brief Tests operator error handling. Ensures that exceptions are being thrown properly.
 *
 */
void TestNegative() {
    TensorShape validShape(TensorLayout(eTensorLayout::TENSOR_LAYOUT_NHWC), {1, 10, 10, 1});
    Tensor validGPUTensor(validShape, DataType(eDataType::DATA_TYPE_U8), eDeviceType::GPU);
    Tensor validCPUTensor(validShape, DataType(eDataType::DATA_TYPE_U8), eDeviceType::CPU);
    Size2D cropSize = {5, 5};
    CenterCrop op;

    // Test output tensor on CPU for GPU operation
    EXPECT_EXCEPTION(op(nullptr, validGPUTensor, validCPUTensor, cropSize, eDeviceType::GPU), eStatusType::INVALID_OPERATION);

    // Test input tensor on CPU for GPU operation
    EXPECT_EXCEPTION(op(nullptr, validCPUTensor, validGPUTensor, cropSize, eDeviceType::GPU), eStatusType::INVALID_OPERATION);

    // Test unsupported layout
    TensorShape invalidLayoutShape(TensorLayout(eTensorLayout::TENSOR_LAYOUT_NC), {5, 5});
    Tensor invalidTensor(invalidLayoutShape, DataType(eDataType::DATA_TYPE_U8), eDeviceType::GPU);
    EXPECT_EXCEPTION(op(nullptr, validGPUTensor, invalidTensor, cropSize, eDeviceType::GPU), eStatusType::INVALID_COMBINATION);

    // Test invalid crop dimensions
    TensorShape outputShape(TensorLayout(eTensorLayout::TENSOR_LAYOUT_NHWC), {1, 5, 5, 1});
    Tensor output(outputShape, DataType(eDataType::DATA_TYPE_U8), eDeviceType::GPU);
    Size2D invalidCropSize = {6, 6};
    EXPECT_EXCEPTION(op(nullptr, validGPUTensor, output, invalidCropSize, eDeviceType::GPU), eStatusType::INVALID_COMBINATION);

    // Test invalid output shape
    TensorShape invalidOutputShape(TensorLayout(eTensorLayout::TENSOR_LAYOUT_NHWC), {1, 55, 55, 1});
    Tensor invalidOutput(invalidOutputShape, DataType(eDataType::DATA_TYPE_U8), eDeviceType::GPU);
    EXPECT_EXCEPTION(op(nullptr, validGPUTensor, invalidOutput, cropSize, eDeviceType::GPU), eStatusType::INVALID_COMBINATION);
}

}

eTestStatusType test_op_center_crop(int argc, char** argv) {
    TEST_CASES_BEGIN();

    // Test negative CenterCrop operator cases
    TEST_CASE(TestNegative());

    // GPU correctness tests
    TEST_CASE(TestCorrectness<uchar1>(1, 256, 256, (Size2D){100, 100}, FMT_U8, eDeviceType::GPU));
    TEST_CASE(TestCorrectness<uchar3>(2, 256, 256, (Size2D){80, 80}, FMT_RGB8, eDeviceType::GPU));
    TEST_CASE(TestCorrectness<uchar4>(3, 256, 128, (Size2D){40, 40}, FMT_RGBA8, eDeviceType::GPU));

    TEST_CASE(TestCorrectness<char1>(2, 256, 256, (Size2D){100, 100}, FMT_S8, eDeviceType::GPU));

    TEST_CASE(TestCorrectness<ushort1>(1, 256, 256, (Size2D){50, 50}, FMT_U16, eDeviceType::GPU));
    TEST_CASE(TestCorrectness<ushort3>(2, 128, 128, (Size2D){30, 30}, FMT_RGB16, eDeviceType::GPU));
    TEST_CASE(TestCorrectness<ushort4>(3, 128, 128, (Size2D){27, 27}, FMT_RGBA16, eDeviceType::GPU));

    TEST_CASE(TestCorrectness<short1>(2, 256, 256, (Size2D){55, 55}, FMT_S16, eDeviceType::GPU));

    TEST_CASE(TestCorrectness<uint1>(1, 256, 256, (Size2D){50, 50}, FMT_U32, eDeviceType::GPU));
    TEST_CASE(TestCorrectness<uint3>(2, 128, 128, (Size2D){30, 30}, FMT_RGB32, eDeviceType::GPU));
    TEST_CASE(TestCorrectness<uint4>(3, 128, 128, (Size2D){60, 60}, FMT_RGBA32, eDeviceType::GPU));

    TEST_CASE(TestCorrectness<int1>(2, 256, 256, (Size2D){100, 100}, FMT_S32, eDeviceType::GPU));

    TEST_CASE(TestCorrectness<float1>(1, 256, 256, (Size2D){50, 50}, FMT_F32, eDeviceType::GPU));
    TEST_CASE(TestCorrectness<float3>(2, 128, 128, (Size2D){70, 50}, FMT_RGBf32, eDeviceType::GPU));
    TEST_CASE(TestCorrectness<float4>(3, 128, 128, (Size2D){60, 60}, FMT_RGBAf32, eDeviceType::GPU));

    TEST_CASE(TestCorrectness<double1>(1, 256, 256, (Size2D){50, 50}, FMT_F64, eDeviceType::GPU));
    TEST_CASE(TestCorrectness<double3>(2, 128, 128, (Size2D){30, 30}, FMT_RGBf64, eDeviceType::GPU));
    TEST_CASE(TestCorrectness<double4>(3, 128, 128, (Size2D){60, 60}, FMT_RGBAf64, eDeviceType::GPU));

    // CPU correctness tests
    TEST_CASE(TestCorrectness<uchar1>(1, 256, 256, (Size2D){100, 100}, FMT_U8, eDeviceType::CPU));
    TEST_CASE(TestCorrectness<uchar3>(2, 256, 256, (Size2D){80, 80}, FMT_RGB8, eDeviceType::CPU));
    TEST_CASE(TestCorrectness<uchar4>(3, 256, 128, (Size2D){40, 40}, FMT_RGBA8, eDeviceType::CPU));

    TEST_CASE(TestCorrectness<char1>(2, 256, 256, (Size2D){100, 100}, FMT_S8, eDeviceType::CPU));

    TEST_CASE(TestCorrectness<ushort1>(1, 256, 256, (Size2D){50, 50}, FMT_U16, eDeviceType::CPU));
    TEST_CASE(TestCorrectness<ushort3>(2, 128, 128, (Size2D){30, 30}, FMT_RGB16, eDeviceType::CPU));
    TEST_CASE(TestCorrectness<ushort4>(3, 128, 128, (Size2D){27, 27}, FMT_RGBA16, eDeviceType::CPU));

    TEST_CASE(TestCorrectness<short1>(2, 256, 256, (Size2D){55, 55}, FMT_S16, eDeviceType::CPU));

    TEST_CASE(TestCorrectness<uint1>(1, 256, 256, (Size2D){50, 50}, FMT_U32, eDeviceType::CPU));
    TEST_CASE(TestCorrectness<uint3>(2, 128, 128, (Size2D){30, 30}, FMT_RGB32, eDeviceType::CPU));
    TEST_CASE(TestCorrectness<uint4>(3, 128, 128, (Size2D){60, 60}, FMT_RGBA32, eDeviceType::CPU));

    TEST_CASE(TestCorrectness<int1>(2, 256, 256, (Size2D){100, 100}, FMT_S32, eDeviceType::CPU));

    TEST_CASE(TestCorrectness<float1>(1, 256, 256, (Size2D){50, 50}, FMT_F32, eDeviceType::CPU));
    TEST_CASE(TestCorrectness<float3>(2, 128, 128, (Size2D){70, 50}, FMT_RGBf32, eDeviceType::CPU));
    TEST_CASE(TestCorrectness<float4>(3, 128, 128, (Size2D){60, 60}, FMT_RGBAf32, eDeviceType::CPU));

    TEST_CASE(TestCorrectness<double1>(1, 256, 256, (Size2D){50, 50}, FMT_F64, eDeviceType::CPU));
    TEST_CASE(TestCorrectness<double3>(2, 128, 128, (Size2D){30, 30}, FMT_RGBf64, eDeviceType::CPU));
    TEST_CASE(TestCorrectness<double4>(3, 128, 128, (Size2D){60, 60}, FMT_RGBAf64, eDeviceType::CPU));

    TEST_CASES_END();
}
