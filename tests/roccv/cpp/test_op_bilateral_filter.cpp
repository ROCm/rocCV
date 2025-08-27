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
#include <core/wrappers/image_wrapper.hpp>
#include <core/wrappers/border_wrapper.hpp>
#include <core/detail/vector_utils.hpp>
#include <op_bilateral_filter.hpp>
#include "test_helpers.hpp"

using namespace roccv;
using namespace roccv::detail;
using namespace roccv::tests;

namespace {

/**
 * @brief Verified golden C++ model for the bilateral filtering operation on one image.
 *
 * @tparam T Vectorized datatype of the image's pixels.
 * @param[in] input Input tensor containing image data.
 * @param[out] output Output tensor containing normalized image data.
 * @param[in] diameter Diameter of the work region
 * @param[in] sigmaColor Sigma component of Gaussian exponent for color difference
 * @param[in] sigmaSpace Sigma component of Gaussian exponent for pixel spatial distance
 * @param[in] borderMode Border pixel extrapolation method
 * @param[in] borderValue Color for constant border mode
 * @return None.
 */
template <typename T, eBorderType borderMode, typename BT = detail::BaseType<T>>
void GenerateGoldenBilateral(std::vector<BT>& input, std::vector<BT>& output, int32_t batchSize, Size2D imageSize, int diameter, float sigmaColor, float sigmaSpace, T borderValue) {
    BorderWrapper<T, borderMode> src(ImageWrapper<T>(input, batchSize, imageSize.w, imageSize.h), borderValue);
    ImageWrapper<T> dst(output, batchSize, imageSize.w, imageSize.h);
    using namespace roccv::detail;
    using Worktype = MakeType<float, NumElements<T>>;

    if (sigmaColor <= 0) {
        sigmaColor = 1.0;
    }
    if (sigmaSpace <= 0) {
        sigmaSpace = 1.0;
    }

    int radius;
    if (diameter <= 0) {
        radius = std::roundf(sigmaSpace * 1.5f);
    } else {
        radius = diameter / 2;
    }
    if (radius < 1) {
        radius = 1;
    }

    float spaceParam = -1 / (2 * sigmaSpace * sigmaSpace);
    float colorParam = -1 / (2 * sigmaColor * sigmaColor);
    float radiusSquared = radius * radius;

    for (int b = 0; b < dst.batches(); b++) {
        for (int j = 0; j < dst.height(); j++) {
            for (int i = 0; i < dst.width(); i++) {
                Worktype numerators = SetAll<Worktype>(0.0f);
                float denominator = 0.0f;
                Worktype currPixel = StaticCast<Worktype>(src.at(b, j, i, 0));

                for (int y = j - radius; y <= j + radius; y++) {
                    for (int x = i - radius; x <= i + radius; x++) {
                        float spaceDistSquared = (x - i) * (x - i) + (y - j) * (y - j);
                        if (spaceDistSquared <= radiusSquared) {
                            Worktype workPixel = StaticCast<Worktype>(src.at(b, y, x, 0));
                            float spaceExp = spaceDistSquared * spaceParam;
                            float colorExp = 0.0f;
                            for (int c = 0; c < NumElements<Worktype>; c++) {
                                colorExp += std::abs(GetElement(currPixel, c) - GetElement(workPixel, c));
                            }
                            colorExp = colorExp * colorExp * colorParam;
                            float weight = std::exp(spaceExp + colorExp);
                            denominator += weight;
                            numerators += weight * workPixel;
                        }
                    }
                }

                dst.at(b, j, i, 0) = SaturateCast<T>(numerators / denominator);
            }
        }
    }
}

/**
 * @brief Tests correctness of the bilateral filter operator, comparing it against a generated golden result.
 *
 * @tparam T Underlying datatype of the image's pixels.
 * @tparam BT Base type of the image data.
 * @param[in] batchSize Number of images in the batch.
 * @param[in] width Width of each image in the batch.
 * @param[in] height Height of each image in the batch.
 * @param[in] format Image format.
 * @param[in] diameter Diameter of the work region
 * @param[in] sigmaColor Sigma component of Gaussian exponent for color difference
 * @param[in] sigmaSpace Sigma component of Gaussian exponent for pixel spatial distance
 * @param[in] borderColor Color for constant border mode
 * @param[in] device Device this correctness test should be run on.
 */
template <typename T, eBorderType BorderMode, typename BT = detail::BaseType<T>>
void TestCorrectness(int batchSize, int width, int height, ImageFormat format, int diameter, float sigmaColor, float sigmaSpace, float4 borderColor, eDeviceType device) {
    // Create input and output tensor based on test parameters
    Tensor input(batchSize, {width, height}, format, device);
    Tensor output(batchSize, {width, height}, format, device);

    // Create a vector and fill it with random data.
    std::vector<BT> inputData(input.shape().size());
    FillVector(inputData);
    if constexpr (std::is_floating_point_v<BT>) {
        for (int i = 0; i < inputData.size(); i++) {
            inputData[i] *= static_cast<BT>(std::numeric_limits<ushort>::max());
        }
    }

    // Copy generated input data into input tensor
    CopyVectorIntoTensor(input, inputData);

    hipStream_t stream;
    HIP_VALIDATE_NO_ERRORS(hipStreamCreate(&stream));
    BilateralFilter op;
    op(stream, input, output, diameter, sigmaColor, sigmaSpace, BorderMode, borderColor, device);
    HIP_VALIDATE_NO_ERRORS(hipStreamSynchronize(stream));
    HIP_VALIDATE_NO_ERRORS(hipStreamDestroy(stream));

    // Copy data from output tensor into a host allocated vector
    std::vector<BT> outputData(output.shape().size());
    CopyTensorIntoVector(outputData, output);

    // Calculate golden reference
    std::vector<BT> refData(output.shape().size());
    GenerateGoldenBilateral<T, BorderMode>(inputData, refData, batchSize, {width, height}, diameter, sigmaColor, sigmaSpace, detail::RangeCast<T>(borderColor));

    // Compare data in actual output versus the generated golden reference image
    CompareVectorsNear(outputData, refData, 1);
}

}

eTestStatusType test_op_bilateral_filter(int argc, char **argv) {
    TEST_CASES_BEGIN();

    // GPU correctness tests
    TEST_CASE((TestCorrectness<uchar, BORDER_TYPE_CONSTANT>(1, 20, 20, FMT_U8, 4, 50.0f, 3.0f, {0.0, 0.0, 0.0, 0.0}, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<uchar, BORDER_TYPE_REPLICATE>(4, 20, 20, FMT_U8, 4, 50.0f, 3.0f, {0.0, 0.0, 0.0, 0.0}, eDeviceType::GPU)));

    TEST_CASE((TestCorrectness<uchar3, BORDER_TYPE_CONSTANT>(1, 20, 20, FMT_RGB8, 4, 50.0f, 3.0f, {0.0, 0.0, 0.0, 0.0}, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<uchar3, BORDER_TYPE_REFLECT>(2, 20, 20, FMT_RGB8, 4, 50.0f, 5.0f, {100.0, 100.0, 0.0, 0.0}, eDeviceType::GPU)));

    TEST_CASE((TestCorrectness<uchar4, BORDER_TYPE_WRAP>(1, 10, 10, FMT_RGBA8, 5, 50.0f, 4.0f, {0.0, 0.0, 0.0, 0.0}, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<uchar4, BORDER_TYPE_REPLICATE>(5, 64, 64, FMT_RGBA8, 5, 50.0f, 4.0f, {0.0, 0.0, 0.0, 0.0}, eDeviceType::GPU)));

    TEST_CASE((TestCorrectness<char1, BORDER_TYPE_CONSTANT>(1, 20, 20, FMT_S8, 4, 50.0f, 3.0f, {100.0, 0.0, 100.0, 0.0}, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<char1, BORDER_TYPE_WRAP>(3, 30, 20, FMT_S8, 4, 50.0f, 3.0f, {100.0, 0.0, 100.0, 0.0}, eDeviceType::GPU)));

    TEST_CASE((TestCorrectness<char3, BORDER_TYPE_REPLICATE>(2, 22, 24, FMT_RGBs8, 5, 60.0f, 3.0f, {100.0, 0.0, 100.0, 0.0}, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<char3, BORDER_TYPE_WRAP>(5, 32, 24, FMT_RGBs8, 5, 60.0f, 5.0f, {100.0, 0.0, 100.0, 0.0}, eDeviceType::GPU)));

    TEST_CASE((TestCorrectness<char4, BORDER_TYPE_REFLECT>(1, 64, 24, FMT_RGBAs8, 5, 60.0f, 3.0f, {100.0, 0.0, 100.0, 0.0}, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<char4, BORDER_TYPE_CONSTANT>(5, 64, 24, FMT_RGBAs8, 5, 60.0f, 3.0f, {100.0, 100.0, 100.0, 0.0}, eDeviceType::GPU)));

    TEST_CASE((TestCorrectness<ushort1, BORDER_TYPE_CONSTANT>(1, 20, 20, FMT_U16, 4, 500.0f, 3.0f, {0.0, 0.0, 0.0, 0.0}, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<ushort1, BORDER_TYPE_REFLECT>(2, 20, 20, FMT_U16, 4, 500.0f, 3.0f, {0.0, 0.0, 0.0, 0.0}, eDeviceType::GPU)));

    TEST_CASE((TestCorrectness<ushort3, BORDER_TYPE_CONSTANT>(1, 20, 20, FMT_RGB16, 4, 500.0f, 3.0f, {500.0, 600.0, 0.0, 0.0}, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<ushort3, BORDER_TYPE_REFLECT>(2, 20, 20, FMT_RGB16, 4, 500.0f, 3.0f, {0.0, 0.0, 0.0, 0.0}, eDeviceType::GPU)));

    TEST_CASE((TestCorrectness<ushort4, BORDER_TYPE_REFLECT>(1, 20, 20, FMT_RGBA16, 4, 600.0f, 3.0f, {500.0, 600.0, 0.0, 0.0}, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<ushort4, BORDER_TYPE_WRAP>(2, 20, 20, FMT_RGBA16, 4, 500.0f, 3.0f, {0.0, 0.0, 0.0, 0.0}, eDeviceType::GPU)));

    TEST_CASE((TestCorrectness<short1, BORDER_TYPE_CONSTANT>(1, 20, 20, FMT_S16, 4, 500.0f, 3.0f, {500.0, 500.0, 0.0, 0.0}, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<short1, BORDER_TYPE_REFLECT>(3, 20, 20, FMT_S16, 4, 500.0f, 3.0f, {500.0, 500.0, 0.0, 0.0}, eDeviceType::GPU)));

    TEST_CASE((TestCorrectness<uint1, BORDER_TYPE_CONSTANT>(1, 20, 20, FMT_U32, 4, 500.0f, 3.0f, {500.0, 500.0, 0.0, 0.0}, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<uint1, BORDER_TYPE_REPLICATE>(2, 24, 16, FMT_U32, 4, 500.0f, 3.0f, {500.0, 500.0, 0.0, 0.0}, eDeviceType::GPU)));

    TEST_CASE((TestCorrectness<uint3, BORDER_TYPE_CONSTANT>(1, 20, 20, FMT_RGB32, 4, 500.0f, 3.0f, {500.0, 500.0, 0.0, 0.0}, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<uint3, BORDER_TYPE_WRAP>(2, 24, 16, FMT_RGB32, 4, 500.0f, 3.0f, {500.0, 500.0, 0.0, 0.0}, eDeviceType::GPU)));

    TEST_CASE((TestCorrectness<uint4, BORDER_TYPE_REPLICATE>(1, 20, 20, FMT_RGBA32, 4, 500.0f, 3.0f, {500.0, 500.0, 0.0, 0.0}, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<uint4, BORDER_TYPE_WRAP>(2, 24, 16, FMT_RGBA32, 4, 500.0f, 3.0f, {500.0, 500.0, 0.0, 0.0}, eDeviceType::GPU)));

    TEST_CASE((TestCorrectness<int1, BORDER_TYPE_CONSTANT>(1, 32, 32, FMT_S32, 4, 500.0f, 3.0f, {500.0, 500.0, 0.0, 0.0}, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<int1, BORDER_TYPE_WRAP>(2, 32, 32, FMT_S32, 4, 500.0f, 3.0f, {500.0, 500.0, 0.0, 0.0}, eDeviceType::GPU)));

    TEST_CASE((TestCorrectness<float1, BORDER_TYPE_REPLICATE>(1, 24, 24, FMT_F32, 4, 500.0f, 3.0f, {500.0, 500.0, 0.0, 0.0}, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<float1, BORDER_TYPE_WRAP>(2, 24, 24, FMT_F32, 4, 600.0f, 3.0f, {500.0, 500.0, 0.0, 0.0}, eDeviceType::GPU)));

    TEST_CASE((TestCorrectness<float3, BORDER_TYPE_REPLICATE>(1, 24, 24, FMT_RGBf32, 4, 500.0f, 3.0f, {500.0, 500.0, 0.0, 0.0}, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<float3, BORDER_TYPE_WRAP>(2, 24, 24, FMT_RGBf32, 4, 600.0f, 3.0f, {500.0, 500.0, 0.0, 0.0}, eDeviceType::GPU)));

    TEST_CASE((TestCorrectness<float4, BORDER_TYPE_REPLICATE>(1, 24, 24, FMT_RGBAf32, 4, 500.0f, 3.0f, {500.0, 500.0, 0.0, 0.0}, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<float4, BORDER_TYPE_WRAP>(2, 24, 24, FMT_RGBAf32, 4, 600.0f, 3.0f, {500.0, 500.0, 0.0, 0.0}, eDeviceType::GPU)));

    TEST_CASE((TestCorrectness<double1, BORDER_TYPE_REPLICATE>(1, 24, 24, FMT_F64, 4, 500.0f, 3.0f, {500.0, 500.0, 0.0, 0.0}, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<double1, BORDER_TYPE_WRAP>(2, 24, 24, FMT_F64, 4, 600.0f, 3.0f, {500.0, 500.0, 0.0, 0.0}, eDeviceType::GPU)));

    TEST_CASE((TestCorrectness<double3, BORDER_TYPE_REPLICATE>(1, 24, 24, FMT_RGBf64, 4, 500.0f, 3.0f, {500.0, 500.0, 0.0, 0.0}, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<double3, BORDER_TYPE_WRAP>(2, 24, 24, FMT_RGBf64, 4, 600.0f, 3.0f, {500.0, 500.0, 0.0, 0.0}, eDeviceType::GPU)));

    TEST_CASE((TestCorrectness<double4, BORDER_TYPE_REPLICATE>(1, 24, 24, FMT_RGBAf64, 4, 500.0f, 3.0f, {500.0, 500.0, 0.0, 0.0}, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<double4, BORDER_TYPE_WRAP>(2, 24, 24, FMT_RGBAf64, 4, 600.0f, 3.0f, {500.0, 500.0, 0.0, 0.0}, eDeviceType::GPU)));

    // CPU correctness tests
    TEST_CASE((TestCorrectness<uchar, BORDER_TYPE_CONSTANT>(1, 20, 20, FMT_U8, 4, 50.0f, 3.0f, {0.0, 0.0, 0.0, 0.0}, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<uchar, BORDER_TYPE_REPLICATE>(4, 20, 20, FMT_U8, 4, 50.0f, 3.0f, {0.0, 0.0, 0.0, 0.0}, eDeviceType::CPU)));

    TEST_CASE((TestCorrectness<uchar3, BORDER_TYPE_CONSTANT>(1, 20, 20, FMT_RGB8, 4, 50.0f, 3.0f, {0.0, 0.0, 0.0, 0.0}, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<uchar3, BORDER_TYPE_REFLECT>(2, 20, 20, FMT_RGB8, 4, 50.0f, 5.0f, {100.0, 100.0, 0.0, 0.0}, eDeviceType::CPU)));

    TEST_CASE((TestCorrectness<uchar4, BORDER_TYPE_WRAP>(1, 10, 10, FMT_RGBA8, 5, 50.0f, 4.0f, {0.0, 0.0, 0.0, 0.0}, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<uchar4, BORDER_TYPE_REPLICATE>(5, 64, 64, FMT_RGBA8, 5, 50.0f, 4.0f, {0.0, 0.0, 0.0, 0.0}, eDeviceType::CPU)));

    TEST_CASE((TestCorrectness<char1, BORDER_TYPE_CONSTANT>(1, 20, 20, FMT_S8, 4, 50.0f, 3.0f, {100.0, 0.0, 100.0, 0.0}, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<char1, BORDER_TYPE_WRAP>(3, 30, 20, FMT_S8, 4, 50.0f, 3.0f, {100.0, 0.0, 100.0, 0.0}, eDeviceType::CPU)));

    TEST_CASE((TestCorrectness<char3, BORDER_TYPE_REPLICATE>(2, 22, 24, FMT_RGBs8, 5, 60.0f, 3.0f, {100.0, 0.0, 100.0, 0.0}, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<char3, BORDER_TYPE_WRAP>(5, 32, 24, FMT_RGBs8, 5, 60.0f, 5.0f, {100.0, 0.0, 100.0, 0.0}, eDeviceType::CPU)));

    TEST_CASE((TestCorrectness<char4, BORDER_TYPE_REFLECT>(1, 64, 24, FMT_RGBAs8, 5, 60.0f, 3.0f, {100.0, 0.0, 100.0, 0.0}, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<char4, BORDER_TYPE_CONSTANT>(5, 64, 24, FMT_RGBAs8, 5, 60.0f, 3.0f, {100.0, 100.0, 100.0, 0.0}, eDeviceType::CPU)));

    TEST_CASE((TestCorrectness<ushort1, BORDER_TYPE_CONSTANT>(1, 20, 20, FMT_U16, 4, 500.0f, 3.0f, {0.0, 0.0, 0.0, 0.0}, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<ushort1, BORDER_TYPE_REFLECT>(2, 20, 20, FMT_U16, 4, 500.0f, 3.0f, {0.0, 0.0, 0.0, 0.0}, eDeviceType::CPU)));

    TEST_CASE((TestCorrectness<ushort3, BORDER_TYPE_CONSTANT>(1, 20, 20, FMT_RGB16, 4, 500.0f, 3.0f, {500.0, 600.0, 0.0, 0.0}, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<ushort3, BORDER_TYPE_REFLECT>(2, 20, 20, FMT_RGB16, 4, 500.0f, 3.0f, {0.0, 0.0, 0.0, 0.0}, eDeviceType::CPU)));

    TEST_CASE((TestCorrectness<ushort4, BORDER_TYPE_REFLECT>(1, 20, 20, FMT_RGBA16, 4, 600.0f, 3.0f, {500.0, 600.0, 0.0, 0.0}, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<ushort4, BORDER_TYPE_WRAP>(2, 20, 20, FMT_RGBA16, 4, 500.0f, 3.0f, {0.0, 0.0, 0.0, 0.0}, eDeviceType::CPU)));

    TEST_CASE((TestCorrectness<short1, BORDER_TYPE_CONSTANT>(1, 20, 20, FMT_S16, 4, 500.0f, 3.0f, {500.0, 500.0, 0.0, 0.0}, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<short1, BORDER_TYPE_REFLECT>(3, 20, 20, FMT_S16, 4, 500.0f, 3.0f, {500.0, 500.0, 0.0, 0.0}, eDeviceType::CPU)));

    TEST_CASE((TestCorrectness<uint1, BORDER_TYPE_CONSTANT>(1, 20, 20, FMT_U32, 4, 500.0f, 3.0f, {500.0, 500.0, 0.0, 0.0}, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<uint1, BORDER_TYPE_REPLICATE>(2, 24, 16, FMT_U32, 4, 500.0f, 3.0f, {500.0, 500.0, 0.0, 0.0}, eDeviceType::CPU)));

    TEST_CASE((TestCorrectness<uint3, BORDER_TYPE_CONSTANT>(1, 20, 20, FMT_RGB32, 4, 500.0f, 3.0f, {500.0, 500.0, 0.0, 0.0}, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<uint3, BORDER_TYPE_WRAP>(2, 24, 16, FMT_RGB32, 4, 500.0f, 3.0f, {500.0, 500.0, 0.0, 0.0}, eDeviceType::CPU)));

    TEST_CASE((TestCorrectness<uint4, BORDER_TYPE_REFLECT>(1, 20, 20, FMT_RGBA32, 4, 500.0f, 3.0f, {500.0, 500.0, 0.0, 0.0}, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<uint4, BORDER_TYPE_WRAP>(2, 24, 16, FMT_RGBA32, 4, 500.0f, 3.0f, {500.0, 500.0, 0.0, 0.0}, eDeviceType::CPU)));

    TEST_CASE((TestCorrectness<int1, BORDER_TYPE_CONSTANT>(1, 32, 32, FMT_S32, 4, 500.0f, 3.0f, {500.0, 500.0, 0.0, 0.0}, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<int1, BORDER_TYPE_WRAP>(2, 32, 32, FMT_S32, 4, 500.0f, 3.0f, {500.0, 500.0, 0.0, 0.0}, eDeviceType::CPU)));

    TEST_CASE((TestCorrectness<float1, BORDER_TYPE_REPLICATE>(1, 24, 24, FMT_F32, 4, 500.0f, 3.0f, {500.0, 500.0, 0.0, 0.0}, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<float1, BORDER_TYPE_WRAP>(2, 24, 24, FMT_F32, 4, 600.0f, 3.0f, {500.0, 500.0, 0.0, 0.0}, eDeviceType::CPU)));

    TEST_CASE((TestCorrectness<float3, BORDER_TYPE_REPLICATE>(1, 24, 24, FMT_RGBf32, 4, 500.0f, 3.0f, {500.0, 500.0, 0.0, 0.0}, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<float3, BORDER_TYPE_WRAP>(2, 24, 24, FMT_RGBf32, 4, 600.0f, 3.0f, {500.0, 500.0, 0.0, 0.0}, eDeviceType::CPU)));

    TEST_CASE((TestCorrectness<float4, BORDER_TYPE_REPLICATE>(1, 24, 24, FMT_RGBAf32, 4, 500.0f, 3.0f, {500.0, 500.0, 0.0, 0.0}, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<float4, BORDER_TYPE_WRAP>(2, 24, 24, FMT_RGBAf32, 4, 600.0f, 3.0f, {500.0, 500.0, 0.0, 0.0}, eDeviceType::CPU)));

    TEST_CASE((TestCorrectness<double1, BORDER_TYPE_REPLICATE>(1, 24, 24, FMT_F64, 4, 5.0f, 3.0f, {0.0, 500.0, 0.0, 0.0}, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<double1, BORDER_TYPE_WRAP>(2, 24, 24, FMT_F64, 4, 5.0f, 3.0f, {0.0, 500.0, 0.0, 0.0}, eDeviceType::CPU)));

    TEST_CASE((TestCorrectness<double1, BORDER_TYPE_REPLICATE>(1, 24, 24, FMT_F64, 4, 500.0f, 3.0f, {500.0, 500.0, 0.0, 0.0}, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<double1, BORDER_TYPE_WRAP>(2, 24, 24, FMT_F64, 4, 600.0f, 3.0f, {500.0, 500.0, 0.0, 0.0}, eDeviceType::CPU)));

    TEST_CASE((TestCorrectness<double3, BORDER_TYPE_REPLICATE>(1, 24, 24, FMT_RGBf64, 4, 500.0f, 3.0f, {500.0, 500.0, 0.0, 0.0}, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<double3, BORDER_TYPE_WRAP>(2, 24, 24, FMT_RGBf64, 4, 600.0f, 3.0f, {500.0, 500.0, 0.0, 0.0}, eDeviceType::CPU)));

    TEST_CASE((TestCorrectness<double4, BORDER_TYPE_REPLICATE>(1, 24, 24, FMT_RGBAf64, 4, 500.0f, 3.0f, {500.0, 500.0, 0.0, 0.0}, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<double4, BORDER_TYPE_WRAP>(2, 24, 24, FMT_RGBAf64, 4, 600.0f, 3.0f, {500.0, 500.0, 0.0, 0.0}, eDeviceType::CPU)));

    TEST_CASES_END();
}
