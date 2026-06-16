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

#include <cfenv>
#include <cmath>
#include <core/detail/casting.hpp>
#include <core/detail/type_traits.hpp>
#include <core/detail/vector_utils.hpp>
#include <core/wrappers/border_wrapper.hpp>
#include <core/wrappers/image_wrapper.hpp>
#include <op_gaussian.hpp>

#include "test_helpers.hpp"

using namespace roccv;
using namespace roccv::detail;
using namespace roccv::tests;

namespace {

/**
 * @brief Golden model for the Gaussian operation.
 *
 * @tparam T Vectorized datatype of the image's pixels.
 * @tparam borderMode Border pixel extrapolation method.
 * @tparam BT Base type of the image's data.
 * @param[in] input Input tensor containing image data.
 * @param[in] batchSize The number of images in the batch.
 * @param[in] width Image width.
 * @param[in] height Image height.
 * @param[in] kernelWidth Kernel width.
 * @param[in] kernelHeight Kernel height.
 * @param[in] sigmaX Kernel standard deviation in X direction.
 * @param[in] sigmaY Kernel standard deviation in Y direction.
 * @return Vector containing the results of the operation.
 */
template <typename T, eBorderType borderMode, typename BT = detail::BaseType<T>>
std::vector<BT> GenerateGoldenGaussian(std::vector<BT>& input, int32_t batchSize, int32_t width, int32_t height,
                                       int kernelWidth, int kernelHeight, double sigmaX, double sigmaY) {
    std::vector<BT> output(input.size());
    BorderWrapper<T, borderMode> src(ImageWrapper<T>(input, batchSize, width, height), SetAll<T>(0));
    ImageWrapper<T> dst(output, batchSize, width, height);

    using namespace roccv::detail;
    using worktype = MakeType<float, NumElements<T>>;

    int halfX = kernelWidth / 2;
    int halfY = kernelHeight / 2;

    float squareSigX = sigmaX * sigmaX;
    float squareSigY = sigmaY * sigmaY;

    for (int b = 0; b < dst.batches(); b++) {
        for (int j = 0; j < dst.height(); j++) {
            for (int i = 0; i < dst.width(); i++) {
                worktype numerators = SetAll<worktype>(0.0f);
                float denominator = 0.0f;

                for (int y = j - halfY; y <= j + halfY; y++) {
                    for (int x = i - halfX; x <= i + halfX; x++) {
                        worktype workPixel = StaticCast<worktype>(src.at(b, y, x, 0));

                        float dx = x - i;
                        float dy = y - j;
                        float expWeight = exp(-((dx * dx) / squareSigX + (dy * dy) / squareSigY) / 2);

                        denominator += expWeight;
                        numerators += expWeight * workPixel;
                    }
                }
                dst.at(b, j, i, 0) = SaturateCast<T>(numerators / denominator);
            }
        }
    }
    return output;
}

/**
 * @brief Tests correctness of the Gaussian operator, comparing it against a generated golden result.
 *
 * @tparam T Underlying datatype of the image's pixels.
 * @tparam borderMode Border pixel extrapolation method.
 * @tparam BT Base type of the image's data.
 * @param[in] batchSize Number of images in the batch.
 * @param[in] width Width of each image in the batch.
 * @param[in] height Height of each image in the batch.
 * @param[in] format Image format.
 * @param[in] kernelWidth Kernel width.
 * @param[in] kernelHeight Kernel height.
 * @param[in] sigmaX Kernel standard deviation in X direction.
 * @param[in] sigmaY Kernel standard deviation in Y direction.
 * @param[in] device Device this correctness test should be run on.
 */
template <typename T, eBorderType BorderMode, typename BT = detail::BaseType<T>>
void TestCorrectness(int batchSize, int width, int height, ImageFormat format, int kernelWidth, int kernelHeight,
                     double sigmaX, double sigmaY, eDeviceType device) {
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

    // Golden infer ksize if nonpositive
    fesetround(FE_TONEAREST);
    auto compute_kdim = [&](int kdim, double sigma) {
        int multiplier = (input.dtype().etype() == DATA_TYPE_U8) ? 6 : 8;
        if (kdim <= 0) {
            float interm = sigma * multiplier + 1;
            kdim = static_cast<int>(std::rint(interm));
            if (kdim % 2 == 0) {
                kdim += 1;
            }
        }
        return kdim;
    };
    double effectiveSigmaY = (sigmaY <= 0) ? sigmaX : sigmaY;
    int effectiveKerWidth = compute_kdim(kernelWidth, sigmaX);
    int effectiveKerHeight = compute_kdim(kernelHeight, effectiveSigmaY);

    hipStream_t stream;
    HIP_VALIDATE_NO_ERRORS(hipStreamCreate(&stream));
    Gaussian op(effectiveKerWidth, effectiveKerHeight);
    op(stream, input, output, kernelWidth, kernelHeight, sigmaX, sigmaY, BorderMode, device);
    HIP_VALIDATE_NO_ERRORS(hipStreamSynchronize(stream));
    HIP_VALIDATE_NO_ERRORS(hipStreamDestroy(stream));

    // Copy data from output tensor into a host allocated vector
    std::vector<BT> outputData(output.shape().size());
    CopyTensorIntoVector(outputData, output);

    // Calculate golden reference
    std::vector<BT> ref = GenerateGoldenGaussian<T, BorderMode>(inputData, batchSize, width, height, effectiveKerWidth,
                                                                effectiveKerHeight, sigmaX, effectiveSigmaY);

    // Compare data in actual output versus the generated golden reference image
    // TODO check on delta, this matches bilateral filter and passes (looser does not)
    CompareVectorsNear(outputData, ref, 1);
}

void TestNegativeGaussian() {
    TensorShape validShape(TensorLayout(eTensorLayout::TENSOR_LAYOUT_NHWC), {1, 1, 1, 1});
    Tensor validGPUTensor(validShape, DataType(eDataType::DATA_TYPE_U8), eDeviceType::GPU);
    Tensor validCPUTensor(validShape, DataType(eDataType::DATA_TYPE_U8), eDeviceType::CPU);

    Gaussian op(3, 3);

    {
        // Test wrong device
        EXPECT_EXCEPTION(
            op(nullptr, validCPUTensor, validGPUTensor, 3, 3, 1, 1, BORDER_TYPE_CONSTANT, eDeviceType::GPU),
            eStatusType::INVALID_OPERATION);
        EXPECT_EXCEPTION(
            op(nullptr, validGPUTensor, validCPUTensor, 3, 3, 1, 1, BORDER_TYPE_CONSTANT, eDeviceType::GPU),
            eStatusType::INVALID_COMBINATION);
    }

    {
        // Test unsupported/mismatch input/output data type
        Tensor invalidTensor(validGPUTensor.shape(), DataType(eDataType::DATA_TYPE_U32), eDeviceType::GPU);
        EXPECT_EXCEPTION(op(nullptr, invalidTensor, validGPUTensor, 3, 3, 1, 1, BORDER_TYPE_REFLECT, eDeviceType::GPU),
                         eStatusType::NOT_IMPLEMENTED);
        EXPECT_EXCEPTION(op(nullptr, validCPUTensor, invalidTensor, 3, 3, 1, 1, BORDER_TYPE_REFLECT, eDeviceType::CPU),
                         eStatusType::INVALID_COMBINATION);
        Tensor validGPUS16Tensor(validShape, DataType(eDataType::DATA_TYPE_S16), eDeviceType::GPU);
        EXPECT_EXCEPTION(
            op(nullptr, validGPUTensor, validGPUS16Tensor, 3, 3, 1, 1, BORDER_TYPE_REFLECT, eDeviceType::GPU),
            eStatusType::INVALID_COMBINATION);
    }

    {
        // Test unsupported input/output layout
        TensorShape invalidLayoutShape(TensorLayout(eTensorLayout::TENSOR_LAYOUT_NC), {1, 1});
        Tensor invalidTensor(invalidLayoutShape, DataType(eDataType::DATA_TYPE_U8), eDeviceType::GPU);
        EXPECT_EXCEPTION(op(nullptr, invalidTensor, validGPUTensor, 3, 3, 1, 1, BORDER_TYPE_WRAP, eDeviceType::GPU),
                         eStatusType::INVALID_COMBINATION);
        EXPECT_EXCEPTION(op(nullptr, validGPUTensor, invalidTensor, 3, 3, 1, 1, BORDER_TYPE_WRAP, eDeviceType::GPU),
                         eStatusType::INVALID_COMBINATION);
    }

    {
        // Test input/output shape mismatch
        Tensor invalidTensor(TensorShape(validGPUTensor.layout(), {2, 2, 2, 2}), DataType(eDataType::DATA_TYPE_U8),
                             eDeviceType::GPU);
        EXPECT_EXCEPTION(
            op(nullptr, invalidTensor, validGPUTensor, 3, 3, 1, 1, BORDER_TYPE_REPLICATE, eDeviceType::GPU),
            eStatusType::INVALID_COMBINATION);
    }

    {
        // Test bad kernel size and sigma X
        // Kernel size exceeds max
        EXPECT_EXCEPTION(
            op(nullptr, validGPUTensor, validGPUTensor, 3, 5, 1, 1, BORDER_TYPE_CONSTANT, eDeviceType::GPU),
            eStatusType::INVALID_VALUE);
        EXPECT_EXCEPTION(
            op(nullptr, validGPUTensor, validGPUTensor, 5, 3, 1, 1, BORDER_TYPE_CONSTANT, eDeviceType::GPU),
            eStatusType::INVALID_VALUE);
        // Kernel size not odd
        EXPECT_EXCEPTION(
            op(nullptr, validGPUTensor, validGPUTensor, 1, 2, 1, 1, BORDER_TYPE_CONSTANT, eDeviceType::GPU),
            eStatusType::INVALID_VALUE);
        EXPECT_EXCEPTION(
            op(nullptr, validGPUTensor, validGPUTensor, 2, 1, 1, 1, BORDER_TYPE_CONSTANT, eDeviceType::GPU),
            eStatusType::INVALID_VALUE);
        // Sigma x not positive
        EXPECT_EXCEPTION(
            op(nullptr, validGPUTensor, validGPUTensor, 3, 3, 0, 1, BORDER_TYPE_CONSTANT, eDeviceType::GPU),
            eStatusType::INVALID_VALUE);
        EXPECT_EXCEPTION(
            op(nullptr, validGPUTensor, validGPUTensor, 3, 3, -0.1, 1, BORDER_TYPE_CONSTANT, eDeviceType::GPU),
            eStatusType::INVALID_VALUE);
    }
}

}  // namespace

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;
    TEST_CASES_BEGIN();

    // Test negative operator cases
    TEST_CASE(TestNegativeGaussian());

    // GPU correctness tests
    TEST_CASE((TestCorrectness<uchar1, BORDER_TYPE_CONSTANT>(1, 20, 20, FMT_U8, 3, 3, 1.0, 1.0, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<ushort1, BORDER_TYPE_WRAP>(1, 20, 20, FMT_U16, 3, 3, 1.0, 1.0, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<ushort1, BORDER_TYPE_REFLECT>(2, 20, 20, FMT_U16, 0, 0, 0.5, 0.5, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<short1, BORDER_TYPE_REPLICATE>(1, 20, 20, FMT_S16, 3, 3, 1.0, 1.0, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<int1, BORDER_TYPE_CONSTANT>(1, 32, 32, FMT_S32, 5, 3, 1.5, 0, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<int1, BORDER_TYPE_WRAP>(2, 32, 32, FMT_S32, 3, 3, 1.0, 1.0, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<float1, BORDER_TYPE_REPLICATE>(1, 24, 24, FMT_F32, 0, 0, 1.5, -1, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<float1, BORDER_TYPE_WRAP>(2, 24, 24, FMT_F32, 5, 5, 1.0, 1.0, eDeviceType::GPU)));

    TEST_CASE((TestCorrectness<uchar3, BORDER_TYPE_REPLICATE>(2, 20, 20, FMT_RGB8, -1, -1, 1.5, 0, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<uchar3, BORDER_TYPE_CONSTANT>(1, 20, 20, FMT_RGB8, 3, 3, 1.0, 1.0, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<ushort3, BORDER_TYPE_CONSTANT>(1, 20, 20, FMT_RGB16, 3, 3, 1.5, 1, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<short3, BORDER_TYPE_CONSTANT>(1, 20, 20, FMT_RGBs16, 0, 3, 1.5, 1.5, eDeviceType::GPU)));
    TEST_CASE(
        (TestCorrectness<short3, BORDER_TYPE_REFLECT101>(2, 20, 20, FMT_RGBs16, 3, 3, 0.5, 0.5, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<int3, BORDER_TYPE_REPLICATE>(1, 24, 24, FMT_RGBs32, 5, 5, 0.5, 0.5, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<float3, BORDER_TYPE_WRAP>(2, 24, 24, FMT_RGBf32, 3, 3, 1.0, 1.0, eDeviceType::GPU)));

    TEST_CASE((TestCorrectness<uchar4, BORDER_TYPE_WRAP>(1, 10, 10, FMT_RGBA8, 3, 3, 0.5, 0.5, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<uchar4, BORDER_TYPE_REPLICATE>(5, 64, 64, FMT_RGBA8, 7, 7, 1, 1.5, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<ushort4, BORDER_TYPE_REFLECT>(1, 20, 20, FMT_RGBA16, 3, 3, 0.5, 0.5, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<ushort4, BORDER_TYPE_WRAP>(2, 20, 20, FMT_RGBA16, 0, 3, 0.5, 0.5, eDeviceType::GPU)));
    TEST_CASE(
        (TestCorrectness<short4, BORDER_TYPE_REFLECT101>(2, 20, 20, FMT_RGBAs16, 3, 3, 0.5, 0.5, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<int4, BORDER_TYPE_REPLICATE>(1, 24, 24, FMT_RGBAs32, 5, 0, 0.5, 0, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<float4, BORDER_TYPE_WRAP>(2, 24, 24, FMT_RGBAf32, 5, 3, 0.5, 0.5, eDeviceType::GPU)));

    // CPU correctness tests
    TEST_CASE((TestCorrectness<uchar1, BORDER_TYPE_CONSTANT>(1, 20, 20, FMT_U8, 3, 3, 1.0, 1.0, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<ushort1, BORDER_TYPE_WRAP>(1, 20, 20, FMT_U16, 3, 3, 1.0, 1.0, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<ushort1, BORDER_TYPE_REFLECT>(2, 20, 20, FMT_U16, 0, 0, 0.5, 0.5, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<short1, BORDER_TYPE_REPLICATE>(1, 20, 20, FMT_S16, 3, 3, 1.0, 1.0, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<int1, BORDER_TYPE_CONSTANT>(1, 32, 32, FMT_S32, 5, 3, 1.5, 0, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<int1, BORDER_TYPE_WRAP>(2, 32, 32, FMT_S32, 3, 3, 1.0, 1.0, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<float1, BORDER_TYPE_REPLICATE>(1, 24, 24, FMT_F32, 0, 0, 1.5, -1, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<float1, BORDER_TYPE_WRAP>(2, 24, 24, FMT_F32, 5, 5, 1.0, 1.0, eDeviceType::CPU)));

    TEST_CASE((TestCorrectness<uchar3, BORDER_TYPE_REPLICATE>(2, 20, 20, FMT_RGB8, -1, -1, 1.5, 0, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<uchar3, BORDER_TYPE_CONSTANT>(1, 20, 20, FMT_RGB8, 3, 3, 1.0, 1.0, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<ushort3, BORDER_TYPE_CONSTANT>(1, 20, 20, FMT_RGB16, 3, 3, 1.5, 1, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<short3, BORDER_TYPE_CONSTANT>(1, 20, 20, FMT_RGBs16, 0, 3, 1.5, 1.5, eDeviceType::CPU)));
    TEST_CASE(
        (TestCorrectness<short3, BORDER_TYPE_REFLECT101>(2, 20, 20, FMT_RGBs16, 3, 3, 0.5, 0.5, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<int3, BORDER_TYPE_REPLICATE>(1, 24, 24, FMT_RGBs32, 5, 5, 0.5, 0.5, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<float3, BORDER_TYPE_WRAP>(2, 24, 24, FMT_RGBf32, 3, 3, 1.0, 1.0, eDeviceType::CPU)));

    TEST_CASE((TestCorrectness<uchar4, BORDER_TYPE_WRAP>(1, 10, 10, FMT_RGBA8, 3, 3, 0.5, 0.5, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<uchar4, BORDER_TYPE_REPLICATE>(5, 64, 64, FMT_RGBA8, 7, 7, 1, 1.5, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<ushort4, BORDER_TYPE_REFLECT>(1, 20, 20, FMT_RGBA16, 3, 3, 0.5, 0.5, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<ushort4, BORDER_TYPE_WRAP>(2, 20, 20, FMT_RGBA16, 0, 3, 0.5, 0.5, eDeviceType::CPU)));
    TEST_CASE(
        (TestCorrectness<short4, BORDER_TYPE_REFLECT101>(2, 20, 20, FMT_RGBAs16, 3, 3, 0.5, 0.5, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<int4, BORDER_TYPE_REPLICATE>(1, 24, 24, FMT_RGBAs32, 5, 0, 0.5, 0, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<float4, BORDER_TYPE_WRAP>(2, 24, 24, FMT_RGBAf32, 5, 3, 0.5, 0.5, eDeviceType::CPU)));

    TEST_CASES_END();
}
