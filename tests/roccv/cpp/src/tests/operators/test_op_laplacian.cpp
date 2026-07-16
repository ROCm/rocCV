/**
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

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

#include <atomic>
#include <cfenv>
#include <cmath>
#include <core/detail/casting.hpp>
#include <core/detail/type_traits.hpp>
#include <core/detail/vector_utils.hpp>
#include <core/wrappers/border_wrapper.hpp>
#include <core/wrappers/image_wrapper.hpp>
#include <op_laplacian.hpp>
#include <thread>

#include "test_helpers.hpp"

using namespace roccv;
using namespace roccv::detail;
using namespace roccv::tests;

namespace {

/**
 * @brief Golden model for the Laplacian operation.
 *
 * @tparam T Vectorized datatype of the image's pixels.
 * @tparam BorderMode Border pixel extrapolation method.
 * @tparam BT Base type of the image's data.
 * @param[in] input Input tensor containing image data.
 * @param[in] batchSize The number of images in the batch.
 * @param[in] width Image width.
 * @param[in] height Image height.
 * @param[in] ksize Aperture size. Must be 1 or 3.
 * @param[in] scale Scale factor for the Laplacian values.
 * @return Vector containing the results of the operation.
 */
template <typename T, eBorderType BorderMode, typename BT = detail::BaseType<T>>
std::vector<BT> GenerateGoldenLaplacian(std::vector<BT>& input, int32_t batchSize, int32_t width, int32_t height,
                                        int ksize, float scale) {
    std::vector<BT> output(input.size());
    BorderWrapper<T, BorderMode> src(ImageWrapper<T>(input, batchSize, width, height), SetAll<T>(0));
    ImageWrapper<T> dst(output, batchSize, width, height);

    using namespace roccv::detail;
    using worktype = MakeType<float, NumElements<T>>;

    std::array<float, 9> kernel;
    if (ksize == 1) {
        kernel = {0.0f, 1.0f, 0.0f, 1.0f, -4.0f, 1.0f, 0.0f, 1.0f, 0.0f};
    } else if (ksize == 3) {
        kernel = {2.0f, 0.0f, 2.0f, 0.0f, -8.0f, 0.0f, 2.0f, 0.0f, 2.0f};
    }

    worktype res;
    int kIdx;
    for (int b = 0; b < dst.batches(); b++) {
        for (int j = 0; j < dst.height(); j++) {
            for (int i = 0; i < dst.width(); i++) {
                res = SetAll<worktype>(0);
                kIdx = 0;
                for (int y = j - 1; y <= j + 1; y++) {
                    for (int x = i - 1; x <= i + 1; x++) {
                        res += StaticCast<worktype>(src.at(b, y, x, 0)) * kernel[kIdx++];
                    }
                }
                dst.at(b, j, i, 0) = SaturateCast<T>(res * scale);
            }
        }
    }
    return output;
}

/**
 * @brief Tests correctness of the Laplacian operator, comparing it against a generated golden result.
 *
 * @tparam T Underlying datatype of the image's pixels.
 * @tparam BorderMode Border pixel extrapolation method.
 * @tparam BT Base type of the image's data.
 * @param[in] batchSize Number of images in the batch.
 * @param[in] width Width of each image in the batch.
 * @param[in] height Height of each image in the batch.
 * @param[in] format Image format.
 * @param[in] ksize Aperture size. Must be 1 or 3.
 * @param[in] scale Scale factor for the Laplacian values.
 * @param[in] device Device this correctness test should be run on.
 */
template <typename T, eBorderType BorderMode, typename BT = detail::BaseType<T>>
void TestCorrectness(int batchSize, int width, int height, ImageFormat format, int ksize, float scale,
                     eDeviceType device) {
    // Create input and output tensor based on test parameters
    Tensor input(batchSize, {width, height}, format, device);
    Tensor output(batchSize, {width, height}, format, device);

    // Create a vector and fill it with random data.
    std::vector<BT> inputData(input.shape().size());
    FillVector(inputData);
    if constexpr (std::is_floating_point_v<BT>) {
        for (size_t i = 0; i < inputData.size(); i++) {
            inputData[i] *= static_cast<BT>(std::numeric_limits<ushort>::max());
        }
    }

    // Copy generated input data into input tensor
    CopyVectorIntoTensor(input, inputData);

    hipStream_t stream;
    HIP_VALIDATE_NO_ERRORS(hipStreamCreate(&stream));
    Laplacian op;
    op(stream, input, output, ksize, scale, BorderMode, device);
    HIP_VALIDATE_NO_ERRORS(hipStreamSynchronize(stream));
    HIP_VALIDATE_NO_ERRORS(hipStreamDestroy(stream));

    // Copy data from output tensor into a host allocated vector
    std::vector<BT> outputData(output.shape().size());
    CopyTensorIntoVector(outputData, output);

    // Calculate golden reference
    std::vector<BT> ref = GenerateGoldenLaplacian<T, BorderMode>(inputData, batchSize, width, height, ksize, scale);

    // Compare data in actual output versus the generated golden reference image
    // TODO check on delta
    CompareVectorsNear(outputData, ref, 1);
}

void TestNegativeLaplacian() {
    TensorShape validShape(TensorLayout(eTensorLayout::TENSOR_LAYOUT_NHWC), {1, 1, 1, 1});
    Tensor validGPUTensor(validShape, DataType(eDataType::DATA_TYPE_U8), eDeviceType::GPU);
    Tensor validCPUTensor(validShape, DataType(eDataType::DATA_TYPE_U8), eDeviceType::CPU);

    Laplacian op;

    {
        // Test wrong device
        EXPECT_EXCEPTION(op(nullptr, validCPUTensor, validGPUTensor, 3, 1, BORDER_TYPE_CONSTANT, eDeviceType::GPU),
                         eStatusType::INVALID_OPERATION);
        EXPECT_EXCEPTION(op(nullptr, validGPUTensor, validCPUTensor, 3, 1, BORDER_TYPE_CONSTANT, eDeviceType::GPU),
                         eStatusType::INVALID_COMBINATION);
    }

    {
        // Test unsupported/mismatch input/output data type
        Tensor invalidTensor(validGPUTensor.shape(), DataType(eDataType::DATA_TYPE_U32), eDeviceType::GPU);
        EXPECT_EXCEPTION(op(nullptr, invalidTensor, validGPUTensor, 3, 1, BORDER_TYPE_REFLECT, eDeviceType::GPU),
                         eStatusType::NOT_IMPLEMENTED);
        EXPECT_EXCEPTION(op(nullptr, validCPUTensor, invalidTensor, 3, 1, BORDER_TYPE_REFLECT, eDeviceType::CPU),
                         eStatusType::INVALID_COMBINATION);
        Tensor validGPUU16Tensor(validShape, DataType(eDataType::DATA_TYPE_U16), eDeviceType::GPU);
        EXPECT_EXCEPTION(op(nullptr, validGPUTensor, validGPUU16Tensor, 3, 1, BORDER_TYPE_REFLECT, eDeviceType::GPU),
                         eStatusType::INVALID_COMBINATION);
    }

    {
        // Test unsupported input/output layout
        TensorShape invalidLayoutShape(TensorLayout(eTensorLayout::TENSOR_LAYOUT_NC), {1, 1});
        Tensor invalidTensor(invalidLayoutShape, DataType(eDataType::DATA_TYPE_U8), eDeviceType::GPU);
        EXPECT_EXCEPTION(op(nullptr, invalidTensor, validGPUTensor, 3, 1, BORDER_TYPE_WRAP, eDeviceType::GPU),
                         eStatusType::INVALID_COMBINATION);
        EXPECT_EXCEPTION(op(nullptr, validGPUTensor, invalidTensor, 3, 1, BORDER_TYPE_WRAP, eDeviceType::GPU),
                         eStatusType::INVALID_COMBINATION);
    }

    {
        // Test input/output shape mismatch
        Tensor invalidTensor(TensorShape(validGPUTensor.layout(), {2, 2, 2, 2}), DataType(eDataType::DATA_TYPE_U8),
                             eDeviceType::GPU);
        EXPECT_EXCEPTION(op(nullptr, invalidTensor, validGPUTensor, 3, 1, BORDER_TYPE_REPLICATE, eDeviceType::GPU),
                         eStatusType::INVALID_COMBINATION);
    }

    {
        // Test bad ksize
        EXPECT_EXCEPTION(op(nullptr, validGPUTensor, validGPUTensor, 4, 1, BORDER_TYPE_CONSTANT, eDeviceType::GPU),
                         eStatusType::INVALID_VALUE);
        EXPECT_EXCEPTION(op(nullptr, validGPUTensor, validGPUTensor, 2, 1, BORDER_TYPE_CONSTANT, eDeviceType::GPU),
                         eStatusType::INVALID_VALUE);
        EXPECT_EXCEPTION(op(nullptr, validGPUTensor, validGPUTensor, 0, 1, BORDER_TYPE_CONSTANT, eDeviceType::GPU),
                         eStatusType::INVALID_VALUE);
    }
}
}  // namespace

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;
    TEST_CASES_BEGIN();

    // Test negative operator cases
    TEST_CASE(TestNegativeLaplacian());

    // GPU correctness tests
    TEST_CASE((TestCorrectness<uchar1, BORDER_TYPE_CONSTANT>(1, 20, 20, FMT_U8, 1, 1.0, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<ushort1, BORDER_TYPE_WRAP>(1, 20, 20, FMT_U16, 3, 1.0, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<float1, BORDER_TYPE_REPLICATE>(1, 24, 24, FMT_F32, 1, 2.0, eDeviceType::GPU)));

    TEST_CASE((TestCorrectness<uchar3, BORDER_TYPE_REFLECT101>(2, 20, 20, FMT_RGB8, 3, 2.0, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<ushort3, BORDER_TYPE_REFLECT>(1, 20, 20, FMT_RGB16, 1, -1.0, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<float3, BORDER_TYPE_WRAP>(2, 24, 24, FMT_RGBf32, 3, -1.0, eDeviceType::GPU)));

    TEST_CASE((TestCorrectness<uchar4, BORDER_TYPE_REPLICATE>(5, 64, 64, FMT_RGBA8, 1, 1.5, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<ushort4, BORDER_TYPE_REFLECT>(1, 20, 20, FMT_RGBA16, 3, 1.5, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<float4, BORDER_TYPE_CONSTANT>(2, 24, 24, FMT_RGBAf32, 3, 10, eDeviceType::GPU)));

    // CPU correctness tests
    TEST_CASE((TestCorrectness<uchar1, BORDER_TYPE_CONSTANT>(1, 20, 20, FMT_U8, 1, 1.0, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<ushort1, BORDER_TYPE_WRAP>(1, 20, 20, FMT_U16, 3, 1.0, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<float1, BORDER_TYPE_REPLICATE>(1, 24, 24, FMT_F32, 1, 2.0, eDeviceType::CPU)));

    TEST_CASE((TestCorrectness<uchar3, BORDER_TYPE_REFLECT101>(2, 20, 20, FMT_RGB8, 3, 2.0, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<ushort3, BORDER_TYPE_REFLECT>(1, 20, 20, FMT_RGB16, 1, -1.0, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<float3, BORDER_TYPE_WRAP>(2, 24, 24, FMT_RGBf32, 3, -1.0, eDeviceType::CPU)));

    TEST_CASE((TestCorrectness<uchar4, BORDER_TYPE_REPLICATE>(5, 64, 64, FMT_RGBA8, 1, 1.5, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<ushort4, BORDER_TYPE_REFLECT>(1, 20, 20, FMT_RGBA16, 3, 1.5, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<float4, BORDER_TYPE_CONSTANT>(2, 24, 24, FMT_RGBAf32, 3, 10, eDeviceType::CPU)));

    TEST_CASES_END();
}