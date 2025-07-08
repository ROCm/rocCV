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
#include <core/wrappers/image_wrapper.hpp>
#include <filesystem>
#include <iostream>
#include <op_flip.hpp>
#include <opencv2/opencv.hpp>

#include "test_helpers.hpp"

using namespace roccv;
using namespace roccv::tests;

namespace {

/**
 * @brief Verified golden C++ model for the flip operation.
 *
 * @tparam T Vectorized datatype of the image's pixels.
 * @tparam BT Base type of the image's data.
 * @param input An input vector containing image data.
 * @param batchSize The number of images in the batch.
 * @param width Image width.
 * @param height Image height.
 * @param channels Number of channels in the image.
 * @param flipCode
 * @return std::vector<BT>
 */
template <typename T, typename BT = detail::BaseType<T>>
std::vector<BT> GoldenFlip(std::vector<BT>& input, int32_t batchSize, int32_t width, int32_t height, int32_t flipCode) {
    // Create an output vector the same size as the input vector
    std::vector<BT> output(input.size());

    // Wrap input/output vectors for simplified data access
    ImageWrapper<T> src(input, batchSize, width, height);
    ImageWrapper<T> dst(output, batchSize, width, height);

    for (int b = 0; b < batchSize; ++b) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                T srcValue;

                // Flip along y-axis
                if (flipCode > 0) {
                    srcValue = src.at(b, y, (src.width() - 1 - x), 0);
                }
                // Flip along x-axis
                else if (flipCode == 0) {
                    srcValue = src.at(b, (src.height() - 1 - y), x, 0);
                }
                // Flip along both axes
                else {
                    srcValue = src.at(b, (src.height() - 1 - y), (src.width() - 1 - x), 0);
                }

                dst.at(b, y, x, 0) = detail::SaturateCast<T>(srcValue);
            }
        }
    }

    return output;
}

/**
 * @brief Tests correctness of the Flip operator, comparing it against a generated golden result.
 *
 * @tparam T Underlying datatype of the image's pixels.
 * @tparam BT Base type of the image data.
 * @param batchSize The number of images in the batch.
 * @param width The width of each image in the batch.
 * @param height The height of each image in the batch.
 * @param flipCode The flip code for each image in the batch.
 * @param format The image format.
 * @param device The device this correctness test should be run on.
 * @return The result of the test.
 */
template <typename T, typename BT = detail::BaseType<T>>
eTestStatusType TestCorrectness(int batchSize, int width, int height, int flipCode, ImageFormat format,
                                eDeviceType device) {
    // Generate random input data
    std::vector<BT> inputData = GenerateRandVector<BT>({width, height}, batchSize, format);

    // Create input and output tensor based on test parameters
    Tensor input(batchSize, {width, height}, format, device);
    Tensor output(batchSize, {width, height}, format, device);

    // Copy generated input data into input tensor
    CopyVectorIntoTensor(inputData, input);

    // Calculate golden output reference
    std::vector<BT> ref = GoldenFlip<T>(inputData, batchSize, width, height, flipCode);

    // Run roccv::Flip operator to obtain actual results
    hipStream_t stream;
    HIP_VALIDATE_NO_ERRORS(hipStreamCreate(&stream));

    Flip op;
    op(stream, input, output, flipCode, device);
    HIP_VALIDATE_NO_ERRORS(hipStreamSynchronize(stream));
    HIP_VALIDATE_NO_ERRORS(hipStreamDestroy(stream));

    // Copy data from output tensor into a host vector
    std::vector<BT> outputVec = CopyTensorIntoVector<BT>(output);

    // Compare final results
    return CompareVectors(outputVec, ref);
}

}  // namespace

eTestStatusType test_op_flip(int argc, char** argv) {
    try {
        // clang-format off
        
        // GPU correctness tests
        EXPECT_TEST_STATUS(TestCorrectness<uchar3>(1, 480, 360, 0, FMT_RGB8, eDeviceType::GPU), eTestStatusType::TEST_SUCCESS);
        EXPECT_TEST_STATUS(TestCorrectness<uchar4>(2, 480, 120, 1, FMT_RGBA8, eDeviceType::GPU), eTestStatusType::TEST_SUCCESS);
        EXPECT_TEST_STATUS(TestCorrectness<uchar1>(3, 360, 360, -1, FMT_U8, eDeviceType::GPU), eTestStatusType::TEST_SUCCESS);
        EXPECT_TEST_STATUS(TestCorrectness<uchar3>(4, 134, 360, 0, FMT_RGB8, eDeviceType::GPU), eTestStatusType::TEST_SUCCESS);
        EXPECT_TEST_STATUS(TestCorrectness<float1>(5, 134, 360, 1, FMT_F32, eDeviceType::GPU), eTestStatusType::TEST_SUCCESS);
        EXPECT_TEST_STATUS(TestCorrectness<float3>(4, 134, 360, 0, FMT_RGBf32, eDeviceType::GPU), eTestStatusType::TEST_SUCCESS);
        EXPECT_TEST_STATUS(TestCorrectness<float4>(2, 480, 120, 1, FMT_RGBAf32, eDeviceType::GPU), eTestStatusType::TEST_SUCCESS);
        EXPECT_TEST_STATUS(TestCorrectness<int1>(5, 134, 360, -1, FMT_S32, eDeviceType::GPU), eTestStatusType::TEST_SUCCESS);

        // CPU correctness tests
        EXPECT_TEST_STATUS(TestCorrectness<uchar3>(1, 480, 360, 0, FMT_RGB8, eDeviceType::CPU), eTestStatusType::TEST_SUCCESS);
        EXPECT_TEST_STATUS(TestCorrectness<uchar4>(2, 480, 120, 1, FMT_RGBA8, eDeviceType::CPU), eTestStatusType::TEST_SUCCESS);
        EXPECT_TEST_STATUS(TestCorrectness<uchar1>(3, 360, 360, -1, FMT_U8, eDeviceType::CPU), eTestStatusType::TEST_SUCCESS);
        EXPECT_TEST_STATUS(TestCorrectness<uchar3>(4, 134, 360, 0, FMT_RGB8, eDeviceType::CPU), eTestStatusType::TEST_SUCCESS);
        EXPECT_TEST_STATUS(TestCorrectness<float1>(5, 134, 360, 1, FMT_F32, eDeviceType::CPU), eTestStatusType::TEST_SUCCESS);
        EXPECT_TEST_STATUS(TestCorrectness<float3>(4, 134, 360, 0, FMT_RGBf32, eDeviceType::CPU), eTestStatusType::TEST_SUCCESS);
        EXPECT_TEST_STATUS(TestCorrectness<float4>(2, 480, 120, 1, FMT_RGBAf32, eDeviceType::CPU), eTestStatusType::TEST_SUCCESS);
        EXPECT_TEST_STATUS(TestCorrectness<int1>(5, 134, 360, -1, FMT_S32, eDeviceType::CPU), eTestStatusType::TEST_SUCCESS);

        // clang-format on

    } catch (Exception e) {
        std::cout << "Exception: " << e.what() << std::endl;
        return eTestStatusType::TEST_FAILURE;
    }
    return eTestStatusType::TEST_SUCCESS;
}

// eTestStatusType TestErrorHandling() {
//     TensorShape validShape(TensorLayout(eTensorLayout::TENSOR_LAYOUT_NHWC), {1, 1, 1, 1});
//     Tensor validGPUTensor(validShape, DataType(eDataType::DATA_TYPE_U8), eDeviceType::GPU);
//     Tensor validCPUTensor(validShape, DataType(eDataType::DATA_TYPE_U8), eDeviceType::CPU);
//     Flip op;

//     {
//         // Test output tensor on CPU for GPU operation
//         EXPECT_EXCEPTION(op(nullptr, validGPUTensor, validCPUTensor, 0, eDeviceType::GPU),
//                          eStatusType::INVALID_OPERATION);
//     }

//     {
//         // Test input tensor on CPU for GPU operation
//         EXPECT_EXCEPTION(op(nullptr, validCPUTensor, validGPUTensor, 0, eDeviceType::GPU),
//                          eStatusType::INVALID_OPERATION);
//     }

//     {
//         // Test unsupported layout
//         TensorShape invalidLayoutShape(TensorLayout(eTensorLayout::TENSOR_LAYOUT_NC), {1, 1});
//         Tensor invalidTensor(invalidLayoutShape, DataType(eDataType::DATA_TYPE_U8), eDeviceType::GPU);
//         EXPECT_EXCEPTION(op(nullptr, invalidTensor, validGPUTensor, 0, eDeviceType::GPU),
//                          eStatusType::INVALID_COMBINATION);
//     }

//     {
//         // Test unsupported data type
//         Tensor invalidTensor(validGPUTensor.shape(), DataType(eDataType::DATA_TYPE_U32), eDeviceType::GPU);
//         EXPECT_EXCEPTION(op(nullptr, invalidTensor, validGPUTensor, 0, eDeviceType::GPU),
//         eStatusType::NOT_IMPLEMENTED);
//     }

//     {
//         // Test input/output shape mismatch
//         Tensor invalidTensor(TensorShape(validGPUTensor.layout(), {2, 2, 2, 2}), DataType(eDataType::DATA_TYPE_U8),
//                              eDeviceType::GPU);
//         EXPECT_EXCEPTION(op(nullptr, invalidTensor, validGPUTensor, 0, eDeviceType::GPU),
//                          eStatusType::INVALID_COMBINATION);
//     }

//     return eTestStatusType::TEST_SUCCESS;
// }
