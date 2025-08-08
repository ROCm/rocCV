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
#include <op_normalize.hpp>
#include "test_helpers.hpp"

using namespace roccv;
using namespace roccv::detail;
using namespace roccv::tests;

namespace {

/**
 * @brief Verified golden C++ model for the normalize operation on one image.
 *
 * @tparam T Vectorized datatype of the image's pixels.
 * @tparam BT Base type of the image's data.
 * @param[in] input Input vector containing image data.
 * @param[out] output Output vector containing normalized image data.
 * @param[in] imageSize Image size.
 * @param[in] imgFormat Image format.
 * @param[in] base Base vector used to shift individual pixel value.
 * @param[in] baseSize Size of the base vector.
 * @param[in] baseFormat Base array format.
 * @param[in] scale Scale vector used for scaling individual pixel value.
 * @param[in] scaleSize Size of the scale vector.
 * @param[in] scaleFormat Scale array format.
 * @param[in] globalShift The value shift applid on all pixels.
 * @param[in] globalScale The scaling factor applid on all pixels.
 * @param[in] epsilon The quantity added to the standard deviaton in normalization using standard deviation (Z-score normalization or standardization).
 * @param[in] scaleIsStdDev The scaling factor is standard deviation.
 * @return None.
 */
template <typename T, typename BT = detail::BaseType<T>>
void GenerateGoldenNormalize(std::vector<BT>& input, std::vector<BT>& output, Size2D imageSize, ImageFormat imgFormat, std::vector<float>& base, Size2D baseSize, ImageFormat baseFormat, std::vector<float>& scale, Size2D scaleSize, ImageFormat scaleFormat, float globalShift, float globalScale, float epsilon, bool scaleIsStdDev) {
    for (int y = 0; y < imageSize.h; y++) {
        int base_y = baseSize.h == 1 ? 0 : y;
        int scale_y = scaleSize.h == 1 ? 0 : y;
        for (int x = 0; x < imageSize.w; x++) {
            int base_x = baseSize.w == 1 ? 0 : x;
            int scale_x = scaleSize.w == 1 ? 0 : x;
            for (int c = 0; c < imgFormat.channels(); c++) {
                int base_c = (baseFormat.channels() == 1 ? 0 : c);
                int scale_c = (scaleFormat.channels() == 1 ? 0 : c);
                float scaleFactor = scale.at(scale_y * scaleSize.w * scaleFormat.channels() + scale_x * scaleFormat.channels() + scale_c);;
                if (scaleIsStdDev) {
                    scaleFactor = 1.0 / std::sqrt(scaleFactor * scaleFactor + epsilon);
                }
                float result = (StaticCast<float>(input.at(y * imageSize.w * imgFormat.channels() + x * imgFormat.channels() + c)) - base.at(base_y * baseSize.w * baseFormat.channels() + base_x * baseFormat.channels() + base_c)) * scaleFactor * globalScale + globalShift;
                output.at(y * imageSize.w * imgFormat.channels() + x * imgFormat.channels() + c) = SaturateCast<BT>(result);
            }
        }
    }
}

/**
 * @brief Tests correctness of the normalize operator, comparing it against a generated golden result.
 *
 * @tparam T Underlying datatype of the image's pixels.
 * @tparam BT Base type of the image data.
 * @param[in] batchSize Number of images in the batch.
 * @param[in] imgSize Image size.
 * @param[in] imgFormat Image format.
 * @param[in] isScalarBase Flag to indicate if the base parameter is scalar or array with the same dimension as input image
 * @param[in] isScalarScale Flag to indicate if the scale parameter is scalar or array with the same dimension as input image
 * @param[in] globalShift The value shift applid on all pixels.
 * @param[in] globalScale The scaling factor applid on all pixels.
 * @param[in] epsilon The quantity added to the standard deviaton in normalization using standard deviation (Z-score normalization or standardization).
 * @param[in] flags The indicator for if the scaling factor is standard deviation.
 * @param[in] device Device this correctness test should be run on.
 */
template <typename T, typename BT = detail::BaseType<T>>
void TestCorrectness(int batchSize, Size2D imgSize, ImageFormat imgFormat, bool isScalarBase, bool isScalarScale, float globalShift, float globalScale, float epsilon, uint32_t flags, eDeviceType device) {
    Tensor input(batchSize, imgSize, imgFormat, device);
    Tensor output(batchSize, imgSize, imgFormat, device);
    Size2D baseSize, scaleSize;
    int baseBatchSize, scaleBatchSize;
    ImageFormat baseFormat, scaleFormat;

    // Create input tensor images.
    std::vector<BT> inputData(input.shape().size());
    FillVector(inputData, 1);
    // Copy generated input data into input tensor
    CopyVectorIntoTensor(input, inputData);

    // Create base tensor images.
    if (isScalarBase) {
        baseBatchSize = 1;
        baseSize = {1, 1};
    } else {
        baseBatchSize = batchSize;
        baseSize = imgSize;
    }
    if (imgFormat.channels() == 1) {
        baseFormat = FMT_F32;
    } else if (imgFormat.channels() == 3) {
        baseFormat = FMT_RGBf32;
    } else {
        baseFormat = FMT_RGBAf32;
    }
    Tensor baseTensor(baseBatchSize, baseSize, baseFormat, device);
    std::vector<float> baseData(baseTensor.shape().size());
    FillVector(baseData, 2);
    for (int i = 0; i < baseTensor.shape().size(); i++) {
        baseData[i] *= static_cast<float>(std::numeric_limits<BT>::max());
    }
    CopyVectorIntoTensor(baseTensor, baseData);

    // Create scale tensor images
    if (isScalarScale) {
        scaleBatchSize = 1;
        scaleSize = {1, 1};
    } else {
        scaleBatchSize = batchSize;
        scaleSize = imgSize;
    }
    if (imgFormat.channels() == 1) {
        scaleFormat = FMT_F32;
    } else if (imgFormat.channels() == 3) {
        scaleFormat = FMT_RGBf32;
    } else {
        scaleFormat = FMT_RGBAf32;
    }
    Tensor scaleTensor(scaleBatchSize, scaleSize, scaleFormat, device);
    std::vector<float> scaleData(scaleTensor.shape().size());
    FillVector(scaleData, 3);
    for (int i = 0; i < scaleTensor.shape().size(); i++) {
        scaleData[i] *= static_cast<float>(std::numeric_limits<BT>::max());
    }
    CopyVectorIntoTensor(scaleTensor, scaleData);

    hipStream_t stream;
    HIP_VALIDATE_NO_ERRORS(hipStreamCreate(&stream));

    Normalize op;
    op(stream, input, baseTensor, scaleTensor, output, globalScale, globalShift, epsilon, flags, device);

    HIP_VALIDATE_NO_ERRORS(hipStreamSynchronize(stream));
    HIP_VALIDATE_NO_ERRORS(hipStreamDestroy(stream));

    // Copy data from output tensor into a host allocated vector
    std::vector<BT> result(output.shape().size());
    CopyTensorIntoVector(result, output);

    // Calculate golden output reference and compare with the output, for each image in the batch.
    int imageVectorSize = output.shape().size() / batchSize;
    std::vector<BT> inputImage(imageVectorSize);
    std::vector<BT> ref(imageVectorSize);
    std::vector<BT> resultImage(imageVectorSize);
    std::vector<float> baseImageData(baseTensor.shape().size() / baseBatchSize);
    std::vector<float> scaleImageData(scaleTensor.shape().size() / scaleBatchSize);
    for (int i = 0; i < batchSize; i++) {
        int offset = i * imageVectorSize;
        std::copy(inputData.begin() + offset, inputData.begin() + offset + imageVectorSize, inputImage.begin());

        if (isScalarBase) {
            baseImageData = baseData;
        } else {
            std::copy(baseData.begin() + offset, baseData.begin() + offset + imageVectorSize, baseImageData.begin());
        }

        if (isScalarScale) {
            scaleImageData = scaleData;
        } else {
            std::copy(scaleData.begin() + offset, scaleData.begin() + offset + imageVectorSize, scaleImageData.begin());
        }

        GenerateGoldenNormalize<T>(inputImage, ref, imgSize, imgFormat, baseImageData, baseSize, baseFormat, scaleImageData, scaleSize, scaleFormat, globalShift, globalScale, epsilon, (flags & ROCCV_NORMALIZE_SCALE_IS_STDDEV));

        // Compare data in actual output versus the generated golden reference image
        std::copy(result.begin() + offset, result.begin() + offset + imageVectorSize, resultImage.begin());
        CompareVectorsNear(resultImage, ref);
    }
}

}

eTestStatusType test_op_normalize(int argc, char** argv) {
    TEST_CASES_BEGIN();

    // GPU correctness tests
    TEST_CASE(TestCorrectness<uchar>(1, {20, 23}, FMT_U8, true, true, 1.0f, 1.2f, 0.1f, 0, eDeviceType::GPU));
    TEST_CASE(TestCorrectness<uchar>(3, {33, 23}, FMT_U8, false, false, 1.0f, 1.2f, 0.1f, ROCCV_NORMALIZE_SCALE_IS_STDDEV, eDeviceType::GPU));

    TEST_CASE(TestCorrectness<uchar3>(1, {200, 200}, FMT_RGB8, false, true, 1.0f, 1.2f, 0.1f, 0, eDeviceType::GPU));
    TEST_CASE(TestCorrectness<uchar3>(2, {100, 25}, FMT_RGB8, true, true, 1.0f, 1.2f, 0.1f, ROCCV_NORMALIZE_SCALE_IS_STDDEV, eDeviceType::GPU))

    TEST_CASE(TestCorrectness<uchar4>(1, {55, 27}, FMT_RGBA8, true, true, 1.0f, 1.2f, 0.1f, 0, eDeviceType::GPU));
    TEST_CASE(TestCorrectness<uchar4>(4, {155, 27}, FMT_RGBA8, false, false, 1.0f, 1.2f, 0.1f, ROCCV_NORMALIZE_SCALE_IS_STDDEV, eDeviceType::GPU));

    TEST_CASE(TestCorrectness<char1>(5, {201, 20}, FMT_S8, true, false, 0.1f, 1.2f, 0.1f, 0, eDeviceType::GPU));
    TEST_CASE(TestCorrectness<char1>(2, {201, 20}, FMT_S8, true, true, 0.1f, 1.2f, 0.1f, ROCCV_NORMALIZE_SCALE_IS_STDDEV, eDeviceType::GPU));

    TEST_CASE(TestCorrectness<char3>(1, {22, 23}, FMT_RGBs8, true, true, 1.0f, 1.2f, 0.1f, 0, eDeviceType::GPU));
    TEST_CASE(TestCorrectness<char3>(3, {22, 23}, FMT_RGBs8, false, false, 1.0f, 1.2f, 0.1f, ROCCV_NORMALIZE_SCALE_IS_STDDEV, eDeviceType::GPU));

    TEST_CASE(TestCorrectness<char4>(1, {52, 42}, FMT_RGBAs8, true, false, 1.0f, 1.2f, 0.1f, 0, eDeviceType::GPU));
    TEST_CASE(TestCorrectness<char4>(7, {25, 25}, FMT_RGBAs8, false, false, 1.0f, 1.2f, 0.1f, ROCCV_NORMALIZE_SCALE_IS_STDDEV, eDeviceType::GPU));

    TEST_CASE(TestCorrectness<ushort1>(1, {200, 200}, FMT_U16, true, false, 1.0f, 1.2f, 0.1f, 0, eDeviceType::GPU));
    TEST_CASE(TestCorrectness<ushort1>(2, {250, 200}, FMT_U16, false, true, 1.1f, 1.0f, 0.3f, ROCCV_NORMALIZE_SCALE_IS_STDDEV, eDeviceType::GPU));

    TEST_CASE(TestCorrectness<ushort3>(1, {9, 8}, FMT_RGB16, true, false, 1.0f, 1.1f, 0.1f, ROCCV_NORMALIZE_SCALE_IS_STDDEV, eDeviceType::GPU));
    TEST_CASE(TestCorrectness<ushort3>(3, {19, 8}, FMT_RGB16, true, false, 1.0f, 1.1f, 0.1f, 0, eDeviceType::GPU));

    TEST_CASE(TestCorrectness<ushort4>(1, {29, 23}, FMT_RGBA16, true, true, 1.0f, 1.1f, 0.1f, 0, eDeviceType::GPU));
    TEST_CASE(TestCorrectness<ushort4>(5, {22, 25}, FMT_RGBA16, false, true, 1.0f, 1.1f, 0.1f, ROCCV_NORMALIZE_SCALE_IS_STDDEV, eDeviceType::GPU));

    TEST_CASE(TestCorrectness<short1>(1, {54, 22}, FMT_S16, false, true, 1.0f, 1.1f, 0.1f, 0, eDeviceType::GPU));
    TEST_CASE(TestCorrectness<short1>(2, {54, 22}, FMT_S16, false, true, 1.0f, 1.1f, 1.3f, ROCCV_NORMALIZE_SCALE_IS_STDDEV, eDeviceType::GPU));

    TEST_CASE(TestCorrectness<int1>(2, {100, 200}, FMT_S32, false, true, 1.0f, 1.1f, 0.1f, 0, eDeviceType::GPU));
    TEST_CASE(TestCorrectness<int1>(4, {100, 200}, FMT_S32, false, true, 1.0f, 1.1f, 0.1f, ROCCV_NORMALIZE_SCALE_IS_STDDEV, eDeviceType::GPU));

    TEST_CASE(TestCorrectness<float1>(2, {20, 2}, FMT_F32, false, true, 1.0f, 1.1f, 0.1f, 0, eDeviceType::GPU));
    TEST_CASE(TestCorrectness<float1>(1, {22, 25}, FMT_F32, false, false, 1.0f, 1.1f, 0.1f, ROCCV_NORMALIZE_SCALE_IS_STDDEV, eDeviceType::GPU));

    TEST_CASE(TestCorrectness<float3>(1, {222, 22}, FMT_RGBf32, false, true, 1.0f, 1.1f, 0.1f, 0, eDeviceType::GPU));
    TEST_CASE(TestCorrectness<float3>(20, {22, 20}, FMT_RGBf32, false, true, 1.0f, 1.1f, 0.1f, ROCCV_NORMALIZE_SCALE_IS_STDDEV, eDeviceType::GPU));

    TEST_CASE(TestCorrectness<float4>(1, {22, 29}, FMT_RGBAf32, true, true, 1.0f, 1.1f, 0.1f, 0, eDeviceType::GPU));
    TEST_CASE(TestCorrectness<float4>(2, {22, 24}, FMT_RGBAf32, false, false, 1.0f, 1.1f, 0.1f, ROCCV_NORMALIZE_SCALE_IS_STDDEV, eDeviceType::GPU));

    // CPU correctness tests
    TEST_CASE(TestCorrectness<uchar>(1, {20, 23}, FMT_U8, true, true, 1.0f, 1.2f, 0.1f, 0, eDeviceType::CPU));
    TEST_CASE(TestCorrectness<uchar>(3, {33, 23}, FMT_U8, false, false, 1.0f, 1.2f, 0.1f, ROCCV_NORMALIZE_SCALE_IS_STDDEV, eDeviceType::CPU));

    TEST_CASE(TestCorrectness<uchar3>(1, {200, 200}, FMT_RGB8, false, true, 1.0f, 1.2f, 0.1f, 0, eDeviceType::CPU));
    TEST_CASE(TestCorrectness<uchar3>(2, {100, 25}, FMT_RGB8, true, true, 1.0f, 1.2f, 0.1f, ROCCV_NORMALIZE_SCALE_IS_STDDEV, eDeviceType::CPU))

    TEST_CASE(TestCorrectness<uchar4>(1, {55, 27}, FMT_RGBA8, true, true, 1.0f, 1.2f, 0.1f, 0, eDeviceType::CPU));
    TEST_CASE(TestCorrectness<uchar4>(4, {155, 27}, FMT_RGBA8, false, false, 1.0f, 1.2f, 0.1f, ROCCV_NORMALIZE_SCALE_IS_STDDEV, eDeviceType::CPU));

    TEST_CASE(TestCorrectness<char1>(5, {201, 20}, FMT_S8, true, false, 0.1f, 1.2f, 0.1f, 0, eDeviceType::CPU));
    TEST_CASE(TestCorrectness<char1>(2, {201, 20}, FMT_S8, true, true, 0.1f, 1.2f, 0.1f, ROCCV_NORMALIZE_SCALE_IS_STDDEV, eDeviceType::CPU));

    TEST_CASE(TestCorrectness<char3>(1, {22, 23}, FMT_RGBs8, true, true, 1.0f, 1.2f, 0.1f, 0, eDeviceType::CPU));
    TEST_CASE(TestCorrectness<char3>(3, {22, 23}, FMT_RGBs8, false, false, 1.0f, 1.2f, 0.1f, ROCCV_NORMALIZE_SCALE_IS_STDDEV, eDeviceType::CPU));

    TEST_CASE(TestCorrectness<char4>(1, {52, 42}, FMT_RGBAs8, true, false, 1.0f, 1.2f, 0.1f, 0, eDeviceType::CPU));
    TEST_CASE(TestCorrectness<char4>(7, {25, 25}, FMT_RGBAs8, false, false, 1.0f, 1.2f, 0.1f, ROCCV_NORMALIZE_SCALE_IS_STDDEV, eDeviceType::CPU));

    TEST_CASE(TestCorrectness<ushort1>(1, {200, 200}, FMT_U16, true, false, 1.0f, 1.2f, 0.1f, 0, eDeviceType::CPU));
    TEST_CASE(TestCorrectness<ushort1>(2, {250, 200}, FMT_U16, false, true, 1.1f, 1.0f, 0.3f, ROCCV_NORMALIZE_SCALE_IS_STDDEV, eDeviceType::CPU));

    TEST_CASE(TestCorrectness<ushort3>(1, {9, 8}, FMT_RGB16, true, false, 1.0f, 1.1f, 0.1f, ROCCV_NORMALIZE_SCALE_IS_STDDEV, eDeviceType::CPU));
    TEST_CASE(TestCorrectness<ushort3>(3, {19, 8}, FMT_RGB16, true, false, 1.0f, 1.1f, 0.1f, 0, eDeviceType::CPU));

    TEST_CASE(TestCorrectness<ushort4>(1, {29, 23}, FMT_RGBA16, true, true, 1.0f, 1.1f, 0.1f, 0, eDeviceType::CPU));
    TEST_CASE(TestCorrectness<ushort4>(5, {22, 25}, FMT_RGBA16, false, true, 1.0f, 1.1f, 0.1f, ROCCV_NORMALIZE_SCALE_IS_STDDEV, eDeviceType::CPU));

    TEST_CASE(TestCorrectness<short1>(1, {54, 22}, FMT_S16, false, true, 1.0f, 1.1f, 0.1f, 0, eDeviceType::CPU));
    TEST_CASE(TestCorrectness<short1>(2, {54, 22}, FMT_S16, false, true, 1.0f, 1.1f, 1.3f, ROCCV_NORMALIZE_SCALE_IS_STDDEV, eDeviceType::CPU));

    TEST_CASE(TestCorrectness<int1>(2, {100, 200}, FMT_S32, false, true, 1.0f, 1.1f, 0.1f, 0, eDeviceType::CPU));
    TEST_CASE(TestCorrectness<int1>(4, {100, 200}, FMT_S32, false, true, 1.0f, 1.1f, 0.1f, ROCCV_NORMALIZE_SCALE_IS_STDDEV, eDeviceType::CPU));

    TEST_CASE(TestCorrectness<float1>(2, {20, 2}, FMT_F32, false, true, 1.0f, 1.1f, 0.1f, 0, eDeviceType::CPU));
    TEST_CASE(TestCorrectness<float1>(1, {22, 25}, FMT_F32, false, false, 1.0f, 1.1f, 0.1f, ROCCV_NORMALIZE_SCALE_IS_STDDEV, eDeviceType::CPU));

    TEST_CASE(TestCorrectness<float3>(1, {222, 22}, FMT_RGBf32, false, true, 1.0f, 1.1f, 0.1f, 0, eDeviceType::CPU));
    TEST_CASE(TestCorrectness<float3>(20, {22, 20}, FMT_RGBf32, false, true, 1.0f, 1.1f, 0.1f, ROCCV_NORMALIZE_SCALE_IS_STDDEV, eDeviceType::CPU));

    TEST_CASE(TestCorrectness<float4>(1, {22, 29}, FMT_RGBAf32, true, true, 1.0f, 1.1f, 0.1f, 0, eDeviceType::CPU));
    TEST_CASE(TestCorrectness<float4>(2, {22, 24}, FMT_RGBAf32, false, false, 1.0f, 1.1f, 0.1f, ROCCV_NORMALIZE_SCALE_IS_STDDEV, eDeviceType::CPU));

    TEST_CASES_END();
}
