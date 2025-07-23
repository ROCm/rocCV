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
#include "core/detail/casting.hpp"
#include "core/detail/type_traits.hpp"
#include "core/detail/math/vectorized_type_math.hpp"
#include <core/wrappers/image_wrapper.hpp>
#include <filesystem>
#include <iostream>
#include <op_gamma_contrast.hpp>
#include <opencv2/opencv.hpp>
#include "operator_types.h"

#include "test_helpers.hpp"

using namespace roccv;
using namespace roccv::tests;

// Keep all non-entrypoint functions in an anonymous namespace to prevent redefinition errors across translation units.
namespace {

/**
 * @brief Verified golden C++ model for the Gamma Contrast operation.
 *
 * @tparam T Vectorized datatype of the image's pixels.
 * @tparam BT Base type of the image's data.
 * @param[in] input An input vector containing image data.
 * @param[in] batchSize The number of images in the batch.
 * @param[in] width Image width.
 * @param[in] height Image height.
 * @param[in] channels Number of channels in the image.
 * @param[in] gamma_value Vector of gamma_values. One for each image in the tensor.
 * @return Vector containing the results of the operation.
 */
template <typename T, typename BT = detail::BaseType<T>>
std::vector<BT> GoldenGammaContrast(std::vector<BT>& input, int32_t batchSize, int32_t width, int32_t height, float gamma) {
    // Create an output vector the same size as the input vector
    std::vector<BT> output(input.size());

    // Wrap input/output vectors for simplified data access
    ImageWrapper<T> src(input, batchSize, width, height);
    ImageWrapper<T> dst(output, batchSize, width, height);
    using work_type = detail::MakeType<float, detail::NumElements<T>>;

    for (int b = 0; b < batchSize; ++b) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                auto inVal = (detail::RangeCast<work_type>(src.at(b, y, x, 0)));
                work_type result = detail::math::vpowf(inVal, gamma);
                if constexpr (detail::NumElements<T> == 4) {
                    dst.at(b, y, x, 0) = detail::RangeCast<T>((detail::MakeType<float, 4>){result.x, result.y, result.z, inVal.w});
                } else {
                    dst.at(b, y, x, 0) = detail::RangeCast<T>(result);
                }
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
 * @param[in] batchSize The number of images in the batch.
 * @param[in] width The width of each image in the batch.
 * @param[in] height The height of each image in the batch.
 * @param[in] gamma_value The vector of gamma values, one value per image.
 * @param[in] format The image format.
 * @param[in] device The device this correctness test should be run on.
 */
template <typename T, typename BT = detail::BaseType<T>>
void TestCorrectness(int batchSize, int width, int height, float gamma, ImageFormat format, eDeviceType device) {
    // Create input and output tensor based on test parameters
    Tensor input(batchSize, {width, height}, format, device);
    Tensor output(batchSize, {width, height}, format, device);
    
    // Create a vector and fill it with random data.
    std::vector<BT> inputData(input.shape().size());
    FillVector(inputData);

    // Copy generated input data into input tensor
    CopyVectorIntoTensor(input, inputData);

    // Calculate golden output reference
    std::vector<BT> ref = GoldenGammaContrast<T>(inputData, batchSize, width, height, gamma);

    // Run roccv::Flip operator to obtain actual results
    hipStream_t stream;
    HIP_VALIDATE_NO_ERRORS(hipStreamCreate(&stream));

    GammaContrast op;
    op(stream, input, output, gamma, device);
    HIP_VALIDATE_NO_ERRORS(hipStreamSynchronize(stream));
    HIP_VALIDATE_NO_ERRORS(hipStreamDestroy(stream));

    // Copy data from output tensor into a host allocated vector
    std::vector<BT> result(output.shape().size());
    CopyTensorIntoVector(result, output);

    // Compare data in actual output versus the generated golden reference image
    //CompareVectorsNear(result, ref, inputData);
    CompareVectorsNear(result, ref);
}
} // namespace

eTestStatusType test_op_gamma_contrast(int argc, char** argv) {
    TEST_CASES_BEGIN();

    // GPU correctness tests
    
    TEST_CASE(TestCorrectness<uchar3>(1, 480, 360, 0.5f, FMT_RGB8, eDeviceType::GPU));
    TEST_CASE(TestCorrectness<uchar4>(2, 480, 120, 0.75f, FMT_RGBA8, eDeviceType::GPU));
    TEST_CASE(TestCorrectness<uchar1>(3, 360, 360, 2.2f, FMT_U8, eDeviceType::GPU));
    TEST_CASE(TestCorrectness<uchar3>(4, 134, 360, 0.4f, FMT_RGB8, eDeviceType::GPU));
    TEST_CASE(TestCorrectness<float1>(5, 134, 360, 0.9f, FMT_F32, eDeviceType::GPU));
    TEST_CASE(TestCorrectness<float3>(4, 134, 360, 1.2f, FMT_RGBf32, eDeviceType::GPU));
    TEST_CASE(TestCorrectness<float4>(2, 480, 120, 2.2f, FMT_RGBAf32, eDeviceType::GPU));
    TEST_CASE(TestCorrectness<uint1>(1, 134, 360, 1.8f, FMT_U32, eDeviceType::GPU));
    TEST_CASE(TestCorrectness<uint3>(1, 134, 360, 2.2f, FMT_RGB32, eDeviceType::GPU));
    TEST_CASE(TestCorrectness<uint4>(1, 134, 360, 0.5f, FMT_RGBA32, eDeviceType::GPU));
    TEST_CASE(TestCorrectness<ushort1>(5, 134, 360, 2.2f, FMT_U16, eDeviceType::GPU));
    TEST_CASE(TestCorrectness<ushort3>(1, 480, 360, 0.8f, FMT_RGB16, eDeviceType::GPU));
    TEST_CASE(TestCorrectness<ushort4>(2, 480, 120, 2.2f, FMT_RGBA16, eDeviceType::GPU));

    // CPU correctness tests
    TEST_CASE(TestCorrectness<uchar3>(1, 480, 360, 0.5f, FMT_RGB8, eDeviceType::CPU));
    TEST_CASE(TestCorrectness<uchar4>(2, 480, 120, 0.75f, FMT_RGBA8, eDeviceType::CPU));
    TEST_CASE(TestCorrectness<uchar1>(3, 360, 360, 2.2f, FMT_U8, eDeviceType::CPU));
    TEST_CASE(TestCorrectness<uchar3>(4, 134, 360, 0.4f, FMT_RGB8, eDeviceType::CPU));
    TEST_CASE(TestCorrectness<float1>(5, 134, 360, 0.9f, FMT_F32, eDeviceType::CPU));
    TEST_CASE(TestCorrectness<float3>(4, 134, 360, 1.2f, FMT_RGBf32, eDeviceType::CPU));
    TEST_CASE(TestCorrectness<float4>(2, 480, 120, 2.2f, FMT_RGBAf32, eDeviceType::CPU));
    TEST_CASE(TestCorrectness<uint1>(1, 134, 360, 1.8f, FMT_U32, eDeviceType::CPU));
    TEST_CASE(TestCorrectness<uint3>(1, 134, 360, 2.2f, FMT_RGB32, eDeviceType::CPU));
    TEST_CASE(TestCorrectness<uint4>(1, 134, 360, 0.5f, FMT_RGBA32, eDeviceType::CPU));
    TEST_CASE(TestCorrectness<ushort1>(5, 134, 360, 2.2f, FMT_U16, eDeviceType::CPU));
    TEST_CASE(TestCorrectness<ushort3>(1, 480, 360, 0.8f, FMT_RGB16, eDeviceType::CPU));
    TEST_CASE(TestCorrectness<ushort4>(2, 480, 120, 2.2f, FMT_RGBA16, eDeviceType::CPU));

    TEST_CASES_END();
}