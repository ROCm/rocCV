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
#include <optional>
#include "core/detail/casting.hpp"
#include "core/detail/type_traits.hpp"
#include "core/detail/math/vectorized_type_math.hpp"
#include <core/wrappers/image_wrapper.hpp>
#include <filesystem>
#include <iostream>
#include <op_thresholding.hpp>
#include <opencv2/opencv.hpp>
#include "operator_types.h"

#include "test_helpers.hpp"

using namespace roccv;
using namespace roccv::tests;
using namespace roccv::detail;

// Keep all non-entrypoint functions in an anonymous namespace to prevent redefinition errors across translation units.
namespace {

/**
 * @brief Verified golden C++ model for the Threshold operation.
 *
 * @tparam T Vectorized datatype of the image's pixels.
 * @tparam BT Base type of the image's data.
 * @param[in] input An input vector containing image data.
 * @param[in] batchSize The number of images in the batch.
 * @param[in] width Image width.
 * @param[in] height Image height.
 * @param[in] channels Number of channels in the image.
 * @param[in] thresh An input vector containing the threshold values for each image
 * @param[in] maxVal An input vector containing the maxval values for each image
 * @return Vector containing the results of the operation.
 */
template <typename T, typename BT = detail::BaseType<T>>
std::vector<BT> GoldenBinaryThreshold(std::vector<BT>& input, int32_t batchSize, int32_t width, int32_t height, std::vector<double> thresh, std::vector<double> maxVal) {
    
    // Create an output vector the same size as the input vector
    std::vector<BT> output(input.size());

    // Wrap input/output vectors for simplified data access
    ImageWrapper<T> src(input, batchSize, width, height);
    ImageWrapper<T> dst(output, batchSize, width, height);
    
    for (int b = 0; b < batchSize; ++b) {
        double th = thresh[b];
        double mv = maxVal[b];
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                T inputVal = src.at(b, y, x, 0);
                T outputVal;
                double outVal;
                for (int i = 0; i < dst.channels(); i++) {
                    double ip = StaticCast<double>(GetElement(inputVal, i));
                    outVal = ip > th ? mv : 0;
                    GetElement(outputVal, i) = StaticCast<BT>(outVal);
                }
                dst.at(b, y, x, 0) = outputVal;
            }
        }
    }
    return output;
}

template <typename T, typename BT = detail::BaseType<T>>
std::vector<BT> GoldenBinaryInvThreshold(std::vector<BT>& input, int32_t batchSize, int32_t width, int32_t height, std::vector<double> thresh, std::vector<double> maxVal) {
    // Create an output vector the same size as the input vector
    std::vector<BT> output(input.size());

    // Wrap input/output vectors for simplified data access
    ImageWrapper<T> src(input, batchSize, width, height);
    ImageWrapper<T> dst(output, batchSize, width, height);
    
    for (int b = 0; b < batchSize; ++b) {
        double th = thresh[b];
        double mv = maxVal[b];
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                T inputVal = src.at(b, y, x, 0);
                T outputVal;
                double outVal;
                for (int i = 0; i < dst.channels(); i++) {
                    double ip = StaticCast<double>(GetElement(inputVal, i));
                    outVal = ip > th ? 0 : mv;
                    GetElement(outputVal, i) = StaticCast<BT>(outVal);
                }
                dst.at(b, y, x, 0) = outputVal;
            }
        }
    }
    return output;
}

template <typename T, typename BT = detail::BaseType<T>>
std::vector<BT> GoldenTruncThreshold(std::vector<BT>& input, int32_t batchSize, int32_t width, int32_t height, std::vector<double> thresh) {
    // Create an output vector the same size as the input vector
    std::vector<BT> output(input.size());

    // Wrap input/output vectors for simplified data access
    ImageWrapper<T> src(input, batchSize, width, height);
    ImageWrapper<T> dst(output, batchSize, width, height);
    
    for (int b = 0; b < batchSize; ++b) {
        double th = thresh[b];
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                T inputVal = src.at(b, y, x, 0);
                T outputVal;
                double outVal;
                for (int i = 0; i < dst.channels(); i++) {
                    double ip = StaticCast<double>(GetElement(inputVal, i));
                    outVal = ip > th ? th : ip;
                    GetElement(outputVal, i) = StaticCast<BT>(outVal);
                }
                dst.at(b, y, x, 0) = outputVal;
            }
        }
    }
    return output;
}

template <typename T, typename BT = detail::BaseType<T>>
std::vector<BT> GoldenToZeroThreshold(std::vector<BT>& input, int32_t batchSize, int32_t width, int32_t height, std::vector<double> thresh) {
    // Create an output vector the same size as the input vector
    std::vector<BT> output(input.size());

    // Wrap input/output vectors for simplified data access
    ImageWrapper<T> src(input, batchSize, width, height);
    ImageWrapper<T> dst(output, batchSize, width, height);
    
    for (int b = 0; b < batchSize; ++b) {
        double th = thresh[b];
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                T inputVal = src.at(b, y, x, 0);
                T outputVal;
                double outVal;
                for (int i = 0; i < dst.channels(); i++) {
                    double ip = StaticCast<double>(GetElement(inputVal, i));
                    outVal = ip > th ? ip : 0;
                    GetElement(outputVal, i) = StaticCast<BT>(outVal);
                }
                dst.at(b, y, x, 0) = outputVal;
            }
        }
    }
    return output;
}

template <typename T, typename BT = detail::BaseType<T>>
std::vector<BT> GoldenToZeroInvThreshold(std::vector<BT>& input, int32_t batchSize, int32_t width, int32_t height, std::vector<double> thresh) {
    // Create an output vector the same size as the input vector
    std::vector<BT> output(input.size());

    // Wrap input/output vectors for simplified data access
    ImageWrapper<T> src(input, batchSize, width, height);
    ImageWrapper<T> dst(output, batchSize, width, height);
    
    for (int b = 0; b < batchSize; ++b) {
        double th = thresh[b];
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                T inputVal = src.at(b, y, x, 0);
                T outputVal;
                double outVal;
                for (int i = 0; i < dst.channels(); i++) {
                    double ip = StaticCast<double>(GetElement(inputVal, i));
                    outVal = ip > th ? 0 : ip;
                    GetElement(outputVal, i) = StaticCast<BT>(outVal);
                }
                dst.at(b, y, x, 0) = outputVal;
            }
        }
    }
    return output;
}

/**
 * @brief Tests correctness of the Threshold operator, comparing it against a generated golden result.
 *
 * @tparam T Underlying datatype of the image's pixels.
 * @tparam BT Base type of the image data.
 * @param[in] batchSize The number of images in the batch.
 * @param[in] width The width of each image in the batch.
 * @param[in] height The height of each image in the batch.
 * @param[in] thresh The threshold value for each image.
 * @param[in] maxVal The maximum value of each image, used for THRESH_BINARY and THRESH_BINARY_INV thresholding types.
 * @param[in] threshType The type of threshold to apply.
 * @param[in] format The image format.
 * @param[in] device The device this correctness test should be run on.
 */
template <typename T, typename BT = detail::BaseType<T>>
void TestCorrectness(int batchSize, int width, int height, double threshVal, double maxVal, eThresholdType threshType, ImageFormat format, eDeviceType device) {
    // Create input and output tensor based on test parameters
    Tensor input(batchSize, {width, height}, format, device);
    Tensor output(batchSize, {width, height}, format, device);
    
    // Create a vector and fill it with random data.
    std::vector<BT> inputData(input.shape().size());
    FillVector(inputData);

    // Copy generated input data into input tensor
    CopyVectorIntoTensor(input, inputData);

    std::vector<double> threshData;
    threshData.assign(batchSize, threshVal);
    std::vector<double> mvData;
    mvData.assign(batchSize, maxVal);

    // Calculate golden output reference
    std::vector<BT> ref;
    switch (threshType) {
            case THRESH_BINARY:
                ref = GoldenBinaryThreshold<T>(inputData, batchSize, width, height, threshData, mvData);
                break;
            case THRESH_BINARY_INV:
                ref = GoldenBinaryInvThreshold<T>(inputData, batchSize, width, height, threshData, mvData);
                break;
            case THRESH_TRUNC:
                ref = GoldenTruncThreshold<T>(inputData, batchSize, width, height, threshData);
                break;
            case THRESH_TOZERO:
                ref = GoldenToZeroThreshold<T>(inputData, batchSize, width, height, threshData);
                break;
            case THRESH_TOZERO_INV:
                ref = GoldenToZeroInvThreshold<T>(inputData, batchSize, width, height, threshData);
                break;
    }

    TensorShape thresh_param_shape(TensorLayout(eTensorLayout::TENSOR_LAYOUT_N), {batchSize});
    TensorShape maxval_param_shape(TensorLayout(eTensorLayout::TENSOR_LAYOUT_N), {batchSize});
    DataType param_dtype(eDataType::DATA_TYPE_F64);
    Tensor thresh(thresh_param_shape, param_dtype, device);
    Tensor maxval(maxval_param_shape, param_dtype, device);

    CopyVectorIntoTensor(thresh, threshData);
    CopyVectorIntoTensor(maxval, mvData);

    // Run roccv::Threshold operator to obtain actual results
    hipStream_t stream;
    HIP_VALIDATE_NO_ERRORS(hipStreamCreate(&stream));

    Threshold op(threshType, batchSize);
    op(stream, input, output, thresh, maxval, device);
    HIP_VALIDATE_NO_ERRORS(hipStreamSynchronize(stream));
    HIP_VALIDATE_NO_ERRORS(hipStreamDestroy(stream));

    // Copy data from output tensor into a host allocated vector
    std::vector<BT> result(output.shape().size());
    CopyTensorIntoVector(result, output);

    // Compare data in actual output versus the generated golden reference image
    CompareVectors(result, ref);
}
} // namespace

eTestStatusType test_op_thresholding(int argc, char** argv) {
    TEST_CASES_BEGIN();

    // GPU correctness tests    
    TEST_CASE(TestCorrectness<uchar1>(1, 360, 360, 100, 255, THRESH_BINARY, FMT_U8, eDeviceType::GPU));
    TEST_CASE(TestCorrectness<uchar3>(2, 480, 360, 100, 255, THRESH_BINARY, FMT_RGB8, eDeviceType::GPU));
    TEST_CASE(TestCorrectness<uchar4>(3, 480, 120, 100, 255, THRESH_BINARY, FMT_RGBA8, eDeviceType::GPU));
    TEST_CASE(TestCorrectness<uchar3>(4, 134, 360, 100, 255, THRESH_BINARY, FMT_RGB8, eDeviceType::GPU));
    TEST_CASE(TestCorrectness<ushort1>(5, 134, 360, 100, 255, THRESH_BINARY, FMT_U16, eDeviceType::GPU));
    TEST_CASE(TestCorrectness<ushort3>(1, 480, 360, 100, 255, THRESH_BINARY, FMT_RGB16, eDeviceType::GPU));
    TEST_CASE(TestCorrectness<ushort4>(2, 480, 120, 100, 255, THRESH_BINARY, FMT_RGBA16, eDeviceType::GPU));
    TEST_CASE(TestCorrectness<short1>(5, 134, 360, 100, 255, THRESH_BINARY, FMT_S16, eDeviceType::GPU));
    TEST_CASE(TestCorrectness<short3>(1, 480, 360, 100, 255, THRESH_BINARY, FMT_RGBs16, eDeviceType::GPU));
    TEST_CASE(TestCorrectness<short4>(2, 480, 120, 100, 255, THRESH_BINARY, FMT_RGBAs16, eDeviceType::GPU));
    TEST_CASE(TestCorrectness<float1>(5, 134, 360, 100, 255, THRESH_BINARY, FMT_F32, eDeviceType::GPU));
    TEST_CASE(TestCorrectness<float3>(4, 134, 360, 100, 255, THRESH_BINARY, FMT_RGBf32, eDeviceType::GPU));
    TEST_CASE(TestCorrectness<float4>(2, 480, 120, 100, 255, THRESH_BINARY, FMT_RGBAf32, eDeviceType::GPU));
    TEST_CASE(TestCorrectness<double1>(5, 134, 360, 100, 255, THRESH_BINARY, FMT_F64, eDeviceType::GPU));
    TEST_CASE(TestCorrectness<double3>(4, 134, 360, 100, 255, THRESH_BINARY, FMT_RGBf64, eDeviceType::GPU));
    TEST_CASE(TestCorrectness<double4>(2, 480, 120, 100, 255, THRESH_BINARY, FMT_RGBAf64, eDeviceType::GPU));
    
    TEST_CASE(TestCorrectness<uchar1>(1, 360, 360, 100, 255, THRESH_BINARY_INV, FMT_U8, eDeviceType::GPU));
    TEST_CASE(TestCorrectness<uchar3>(2, 480, 360, 100, 255, THRESH_BINARY_INV, FMT_RGB8, eDeviceType::GPU));
    TEST_CASE(TestCorrectness<uchar4>(3, 480, 120, 100, 255, THRESH_BINARY_INV, FMT_RGBA8, eDeviceType::GPU));
    TEST_CASE(TestCorrectness<uchar3>(4, 134, 360, 100, 255, THRESH_BINARY_INV, FMT_RGB8, eDeviceType::GPU));
    TEST_CASE(TestCorrectness<ushort1>(5, 134, 360, 100, 255, THRESH_BINARY_INV, FMT_U16, eDeviceType::GPU));
    TEST_CASE(TestCorrectness<ushort3>(1, 480, 360, 100, 255, THRESH_BINARY_INV, FMT_RGB16, eDeviceType::GPU));
    TEST_CASE(TestCorrectness<ushort4>(2, 480, 120, 100, 255, THRESH_BINARY_INV, FMT_RGBA16, eDeviceType::GPU));
    TEST_CASE(TestCorrectness<short1>(5, 134, 360, 100, 255, THRESH_BINARY_INV, FMT_S16, eDeviceType::GPU));
    TEST_CASE(TestCorrectness<short3>(1, 480, 360, 100, 255, THRESH_BINARY_INV, FMT_RGBs16, eDeviceType::GPU));
    TEST_CASE(TestCorrectness<short4>(2, 480, 120, 100, 255, THRESH_BINARY_INV, FMT_RGBAs16, eDeviceType::GPU));
    TEST_CASE(TestCorrectness<float1>(5, 134, 360, 100, 255, THRESH_BINARY_INV, FMT_F32, eDeviceType::GPU));
    TEST_CASE(TestCorrectness<float3>(4, 134, 360, 100, 255, THRESH_BINARY_INV, FMT_RGBf32, eDeviceType::GPU));
    TEST_CASE(TestCorrectness<float4>(2, 480, 120, 100, 255, THRESH_BINARY_INV, FMT_RGBAf32, eDeviceType::GPU));
    TEST_CASE(TestCorrectness<double1>(5, 134, 360, 100, 255, THRESH_BINARY_INV, FMT_F64, eDeviceType::GPU));
    TEST_CASE(TestCorrectness<double3>(4, 134, 360, 100, 255, THRESH_BINARY_INV, FMT_RGBf64, eDeviceType::GPU));
    TEST_CASE(TestCorrectness<double4>(2, 480, 120, 100, 255, THRESH_BINARY_INV, FMT_RGBAf64, eDeviceType::GPU));
    
    TEST_CASE(TestCorrectness<uchar1>(1, 360, 360, 100, 255, THRESH_TRUNC, FMT_U8, eDeviceType::GPU));
    TEST_CASE(TestCorrectness<uchar3>(2, 480, 360, 100, 255, THRESH_TRUNC, FMT_RGB8, eDeviceType::GPU));
    TEST_CASE(TestCorrectness<uchar4>(3, 480, 120, 100, 255, THRESH_TRUNC, FMT_RGBA8, eDeviceType::GPU));
    TEST_CASE(TestCorrectness<uchar3>(4, 134, 360, 100, 255, THRESH_TRUNC, FMT_RGB8, eDeviceType::GPU));
    TEST_CASE(TestCorrectness<ushort1>(5, 134, 360, 100, 255, THRESH_TRUNC, FMT_U16, eDeviceType::GPU));
    TEST_CASE(TestCorrectness<ushort3>(1, 480, 360, 100, 255, THRESH_TRUNC, FMT_RGB16, eDeviceType::GPU));
    TEST_CASE(TestCorrectness<ushort4>(2, 480, 120, 100, 255, THRESH_TRUNC, FMT_RGBA16, eDeviceType::GPU));
    TEST_CASE(TestCorrectness<short1>(5, 134, 360, 100, 255, THRESH_TRUNC, FMT_S16, eDeviceType::GPU));
    TEST_CASE(TestCorrectness<short3>(1, 480, 360, 100, 255, THRESH_TRUNC, FMT_RGBs16, eDeviceType::GPU));
    TEST_CASE(TestCorrectness<short4>(2, 480, 120, 100, 255, THRESH_TRUNC, FMT_RGBAs16, eDeviceType::GPU));
    TEST_CASE(TestCorrectness<float1>(5, 134, 360, 100, 255, THRESH_TRUNC, FMT_F32, eDeviceType::GPU));
    TEST_CASE(TestCorrectness<float3>(4, 134, 360, 100, 255, THRESH_TRUNC, FMT_RGBf32, eDeviceType::GPU));
    TEST_CASE(TestCorrectness<float4>(2, 480, 120, 100, 255, THRESH_TRUNC, FMT_RGBAf32, eDeviceType::GPU));
    TEST_CASE(TestCorrectness<double1>(5, 134, 360, 100, 255, THRESH_TRUNC, FMT_F64, eDeviceType::GPU));
    TEST_CASE(TestCorrectness<double3>(4, 134, 360, 100, 255, THRESH_TRUNC, FMT_RGBf64, eDeviceType::GPU));
    TEST_CASE(TestCorrectness<double4>(2, 480, 120, 100, 255, THRESH_TRUNC, FMT_RGBAf64, eDeviceType::GPU));

    TEST_CASE(TestCorrectness<uchar1>(1, 360, 360, 100, 255, THRESH_TOZERO, FMT_U8, eDeviceType::GPU));
    TEST_CASE(TestCorrectness<uchar3>(2, 480, 360, 100, 255, THRESH_TOZERO, FMT_RGB8, eDeviceType::GPU));
    TEST_CASE(TestCorrectness<uchar4>(3, 480, 120, 100, 255, THRESH_TOZERO, FMT_RGBA8, eDeviceType::GPU));
    TEST_CASE(TestCorrectness<uchar3>(4, 134, 360, 100, 255, THRESH_TOZERO, FMT_RGB8, eDeviceType::GPU));
    TEST_CASE(TestCorrectness<ushort1>(5, 134, 360, 100, 255, THRESH_TOZERO, FMT_U16, eDeviceType::GPU));
    TEST_CASE(TestCorrectness<ushort3>(1, 480, 360, 100, 255, THRESH_TOZERO, FMT_RGB16, eDeviceType::GPU));
    TEST_CASE(TestCorrectness<ushort4>(2, 480, 120, 100, 255, THRESH_TOZERO, FMT_RGBA16, eDeviceType::GPU));
    TEST_CASE(TestCorrectness<short1>(5, 134, 360, 100, 255, THRESH_TOZERO, FMT_S16, eDeviceType::GPU));
    TEST_CASE(TestCorrectness<short3>(1, 480, 360, 100, 255, THRESH_TOZERO, FMT_RGBs16, eDeviceType::GPU));
    TEST_CASE(TestCorrectness<short4>(2, 480, 120, 100, 255, THRESH_TOZERO, FMT_RGBAs16, eDeviceType::GPU));
    TEST_CASE(TestCorrectness<float1>(5, 134, 360, 100, 255, THRESH_TOZERO, FMT_F32, eDeviceType::GPU));
    TEST_CASE(TestCorrectness<float3>(4, 134, 360, 100, 255, THRESH_TOZERO, FMT_RGBf32, eDeviceType::GPU));
    TEST_CASE(TestCorrectness<float4>(2, 480, 120, 100, 255, THRESH_TOZERO, FMT_RGBAf32, eDeviceType::GPU));
    TEST_CASE(TestCorrectness<double1>(5, 134, 360, 100, 255, THRESH_TOZERO, FMT_F64, eDeviceType::GPU));
    TEST_CASE(TestCorrectness<double3>(4, 134, 360, 100, 255, THRESH_TOZERO, FMT_RGBf64, eDeviceType::GPU));
    TEST_CASE(TestCorrectness<double4>(2, 480, 120, 100, 255, THRESH_TOZERO, FMT_RGBAf64, eDeviceType::GPU));

    TEST_CASE(TestCorrectness<uchar1>(1, 360, 360, 100, 255, THRESH_TOZERO_INV, FMT_U8, eDeviceType::GPU));
    TEST_CASE(TestCorrectness<uchar3>(2, 480, 360, 100, 255, THRESH_TOZERO_INV, FMT_RGB8, eDeviceType::GPU));
    TEST_CASE(TestCorrectness<uchar4>(3, 480, 120, 100, 255, THRESH_TOZERO_INV, FMT_RGBA8, eDeviceType::GPU));
    TEST_CASE(TestCorrectness<uchar3>(4, 134, 360, 100, 255, THRESH_TOZERO_INV, FMT_RGB8, eDeviceType::GPU));
    TEST_CASE(TestCorrectness<ushort1>(5, 134, 360, 100, 255, THRESH_TOZERO_INV, FMT_U16, eDeviceType::GPU));
    TEST_CASE(TestCorrectness<ushort3>(1, 480, 360, 100, 255, THRESH_TOZERO_INV, FMT_RGB16, eDeviceType::GPU));
    TEST_CASE(TestCorrectness<ushort4>(2, 480, 120, 100, 255, THRESH_TOZERO_INV, FMT_RGBA16, eDeviceType::GPU));
    TEST_CASE(TestCorrectness<short1>(5, 134, 360, 100, 255, THRESH_TOZERO_INV, FMT_S16, eDeviceType::GPU));
    TEST_CASE(TestCorrectness<short3>(1, 480, 360, 100, 255, THRESH_TOZERO_INV, FMT_RGBs16, eDeviceType::GPU));
    TEST_CASE(TestCorrectness<short4>(2, 480, 120, 100, 255, THRESH_TOZERO_INV, FMT_RGBAs16, eDeviceType::GPU));
    TEST_CASE(TestCorrectness<float1>(5, 134, 360, 100, 255, THRESH_TOZERO_INV, FMT_F32, eDeviceType::GPU));
    TEST_CASE(TestCorrectness<float3>(4, 134, 360, 100, 255, THRESH_TOZERO_INV, FMT_RGBf32, eDeviceType::GPU));
    TEST_CASE(TestCorrectness<float4>(2, 480, 120, 100, 255, THRESH_TOZERO_INV, FMT_RGBAf32, eDeviceType::GPU));
    TEST_CASE(TestCorrectness<double1>(5, 134, 360, 100, 255, THRESH_TOZERO_INV, FMT_F64, eDeviceType::GPU));
    TEST_CASE(TestCorrectness<double3>(4, 134, 360, 100, 255, THRESH_TOZERO_INV, FMT_RGBf64, eDeviceType::GPU));
    TEST_CASE(TestCorrectness<double4>(2, 480, 120, 100, 255, THRESH_TOZERO_INV, FMT_RGBAf64, eDeviceType::GPU));
        
    // CPU correctness tests
    TEST_CASE(TestCorrectness<uchar1>(1, 360, 360, 100, 255, THRESH_BINARY, FMT_U8, eDeviceType::CPU));
    TEST_CASE(TestCorrectness<uchar3>(2, 480, 360, 100, 255, THRESH_BINARY, FMT_RGB8, eDeviceType::CPU));
    TEST_CASE(TestCorrectness<uchar4>(3, 480, 120, 100, 255, THRESH_BINARY, FMT_RGBA8, eDeviceType::CPU));
    TEST_CASE(TestCorrectness<uchar3>(4, 134, 360, 100, 255, THRESH_BINARY, FMT_RGB8, eDeviceType::CPU));
    TEST_CASE(TestCorrectness<float1>(5, 134, 360, 100, 255, THRESH_BINARY, FMT_F32, eDeviceType::CPU));
    TEST_CASE(TestCorrectness<float3>(4, 134, 360, 100, 255, THRESH_BINARY, FMT_RGBf32, eDeviceType::CPU));
    TEST_CASE(TestCorrectness<float4>(2, 480, 120, 100, 255, THRESH_BINARY, FMT_RGBAf32, eDeviceType::CPU));
    TEST_CASE(TestCorrectness<ushort1>(5, 134, 360, 100, 255, THRESH_BINARY, FMT_U16, eDeviceType::CPU));
    TEST_CASE(TestCorrectness<ushort3>(1, 480, 360, 100, 255, THRESH_BINARY, FMT_RGB16, eDeviceType::CPU));
    TEST_CASE(TestCorrectness<ushort4>(2, 480, 120, 100, 255, THRESH_BINARY, FMT_RGBA16, eDeviceType::CPU));
    TEST_CASE(TestCorrectness<short1>(5, 134, 360, 100, 255, THRESH_BINARY, FMT_S16, eDeviceType::CPU));
    TEST_CASE(TestCorrectness<short3>(1, 480, 360, 100, 255, THRESH_BINARY, FMT_RGBs16, eDeviceType::CPU));
    TEST_CASE(TestCorrectness<short4>(2, 480, 120, 100, 255, THRESH_BINARY, FMT_RGBAs16, eDeviceType::CPU));
    TEST_CASE(TestCorrectness<double1>(5, 134, 360, 100, 255, THRESH_BINARY, FMT_F64, eDeviceType::CPU));
    TEST_CASE(TestCorrectness<double3>(4, 134, 360, 100, 255, THRESH_BINARY, FMT_RGBf64, eDeviceType::CPU));
    TEST_CASE(TestCorrectness<double4>(2, 480, 120, 100, 255, THRESH_BINARY, FMT_RGBAf64, eDeviceType::CPU));

    TEST_CASE(TestCorrectness<uchar1>(1, 360, 360, 100, 255, THRESH_BINARY_INV, FMT_U8, eDeviceType::CPU));
    TEST_CASE(TestCorrectness<uchar3>(2, 480, 360, 100, 255, THRESH_BINARY_INV, FMT_RGB8, eDeviceType::CPU));
    TEST_CASE(TestCorrectness<uchar4>(3, 480, 120, 100, 255, THRESH_BINARY_INV, FMT_RGBA8, eDeviceType::CPU));
    TEST_CASE(TestCorrectness<uchar3>(4, 134, 360, 100, 255, THRESH_BINARY_INV, FMT_RGB8, eDeviceType::CPU));
    TEST_CASE(TestCorrectness<float1>(5, 134, 360, 100, 255, THRESH_BINARY_INV, FMT_F32, eDeviceType::CPU));
    TEST_CASE(TestCorrectness<float3>(4, 134, 360, 100, 255, THRESH_BINARY_INV, FMT_RGBf32, eDeviceType::CPU));
    TEST_CASE(TestCorrectness<float4>(2, 480, 120, 100, 255, THRESH_BINARY_INV, FMT_RGBAf32, eDeviceType::CPU));
    TEST_CASE(TestCorrectness<ushort1>(5, 134, 360, 100, 255, THRESH_BINARY_INV, FMT_U16, eDeviceType::CPU));
    TEST_CASE(TestCorrectness<ushort3>(1, 480, 360, 100, 255, THRESH_BINARY_INV, FMT_RGB16, eDeviceType::CPU));
    TEST_CASE(TestCorrectness<ushort4>(2, 480, 120, 100, 255, THRESH_BINARY_INV, FMT_RGBA16, eDeviceType::CPU));
    TEST_CASE(TestCorrectness<short1>(5, 134, 360, 100, 255, THRESH_BINARY_INV, FMT_S16, eDeviceType::CPU));
    TEST_CASE(TestCorrectness<short3>(1, 480, 360, 100, 255, THRESH_BINARY_INV, FMT_RGBs16, eDeviceType::CPU));
    TEST_CASE(TestCorrectness<short4>(2, 480, 120, 100, 255, THRESH_BINARY_INV, FMT_RGBAs16, eDeviceType::CPU));
    TEST_CASE(TestCorrectness<double1>(5, 134, 360, 100, 255, THRESH_BINARY_INV, FMT_F64, eDeviceType::CPU));
    TEST_CASE(TestCorrectness<double3>(4, 134, 360, 100, 255, THRESH_BINARY_INV, FMT_RGBf64, eDeviceType::CPU));
    TEST_CASE(TestCorrectness<double4>(2, 480, 120, 100, 255, THRESH_BINARY_INV, FMT_RGBAf64, eDeviceType::CPU));

    TEST_CASE(TestCorrectness<uchar1>(1, 360, 360, 100, 255, THRESH_TRUNC, FMT_U8, eDeviceType::CPU));
    TEST_CASE(TestCorrectness<uchar3>(2, 480, 360, 100, 255, THRESH_TRUNC, FMT_RGB8, eDeviceType::CPU));
    TEST_CASE(TestCorrectness<uchar4>(3, 480, 120, 100, 255, THRESH_TRUNC, FMT_RGBA8, eDeviceType::CPU));
    TEST_CASE(TestCorrectness<uchar3>(4, 134, 360, 100, 255, THRESH_TRUNC, FMT_RGB8, eDeviceType::CPU));
    TEST_CASE(TestCorrectness<ushort1>(5, 134, 360, 100, 255, THRESH_TRUNC, FMT_U16, eDeviceType::CPU));
    TEST_CASE(TestCorrectness<ushort3>(1, 480, 360, 100, 255, THRESH_TRUNC, FMT_RGB16, eDeviceType::CPU));
    TEST_CASE(TestCorrectness<ushort4>(2, 480, 120, 100, 255, THRESH_TRUNC, FMT_RGBA16, eDeviceType::CPU));
    TEST_CASE(TestCorrectness<short1>(5, 134, 360, 100, 255, THRESH_TRUNC, FMT_S16, eDeviceType::CPU));
    TEST_CASE(TestCorrectness<short3>(1, 480, 360, 100, 255, THRESH_TRUNC, FMT_RGBs16, eDeviceType::CPU));
    TEST_CASE(TestCorrectness<short4>(2, 480, 120, 100, 255, THRESH_TRUNC, FMT_RGBAs16, eDeviceType::CPU));
    TEST_CASE(TestCorrectness<float1>(5, 134, 360, 100, 255, THRESH_TRUNC, FMT_F32, eDeviceType::CPU));
    TEST_CASE(TestCorrectness<float3>(4, 134, 360, 100, 255, THRESH_TRUNC, FMT_RGBf32, eDeviceType::CPU));
    TEST_CASE(TestCorrectness<float4>(2, 480, 120, 100, 255, THRESH_TRUNC, FMT_RGBAf32, eDeviceType::CPU));
    TEST_CASE(TestCorrectness<double1>(5, 134, 360, 100, 255, THRESH_TRUNC, FMT_F64, eDeviceType::CPU));
    TEST_CASE(TestCorrectness<double3>(4, 134, 360, 100, 255, THRESH_TRUNC, FMT_RGBf64, eDeviceType::CPU));
    TEST_CASE(TestCorrectness<double4>(2, 480, 120, 100, 255, THRESH_TRUNC, FMT_RGBAf64, eDeviceType::CPU));

    TEST_CASE(TestCorrectness<uchar1>(1, 360, 360, 100, 255, THRESH_TOZERO, FMT_U8, eDeviceType::CPU));
    TEST_CASE(TestCorrectness<uchar3>(2, 480, 360, 100, 255, THRESH_TOZERO, FMT_RGB8, eDeviceType::CPU));
    TEST_CASE(TestCorrectness<uchar4>(3, 480, 120, 100, 255, THRESH_TOZERO, FMT_RGBA8, eDeviceType::CPU));
    TEST_CASE(TestCorrectness<uchar3>(4, 134, 360, 100, 255, THRESH_TOZERO, FMT_RGB8, eDeviceType::CPU));
    TEST_CASE(TestCorrectness<ushort1>(5, 134, 360, 100, 255, THRESH_TOZERO, FMT_U16, eDeviceType::CPU));
    TEST_CASE(TestCorrectness<ushort3>(1, 480, 360, 100, 255, THRESH_TOZERO, FMT_RGB16, eDeviceType::CPU));
    TEST_CASE(TestCorrectness<ushort4>(2, 480, 120, 100, 255, THRESH_TOZERO, FMT_RGBA16, eDeviceType::CPU));
    TEST_CASE(TestCorrectness<short1>(5, 134, 360, 100, 255, THRESH_TOZERO, FMT_S16, eDeviceType::CPU));
    TEST_CASE(TestCorrectness<short3>(1, 480, 360, 100, 255, THRESH_TOZERO, FMT_RGBs16, eDeviceType::CPU));
    TEST_CASE(TestCorrectness<short4>(2, 480, 120, 100, 255, THRESH_TOZERO, FMT_RGBAs16, eDeviceType::CPU));
    TEST_CASE(TestCorrectness<float1>(5, 134, 360, 100, 255, THRESH_TOZERO, FMT_F32, eDeviceType::CPU));
    TEST_CASE(TestCorrectness<float3>(4, 134, 360, 100, 255, THRESH_TOZERO, FMT_RGBf32, eDeviceType::CPU));
    TEST_CASE(TestCorrectness<float4>(2, 480, 120, 100, 255, THRESH_TOZERO, FMT_RGBAf32, eDeviceType::CPU));
    TEST_CASE(TestCorrectness<double1>(5, 134, 360, 100, 255, THRESH_TOZERO, FMT_F64, eDeviceType::CPU));
    TEST_CASE(TestCorrectness<double3>(4, 134, 360, 100, 255, THRESH_TOZERO, FMT_RGBf64, eDeviceType::CPU));
    TEST_CASE(TestCorrectness<double4>(2, 480, 120, 100, 255, THRESH_TOZERO, FMT_RGBAf64, eDeviceType::CPU));

    TEST_CASE(TestCorrectness<uchar1>(1, 360, 360, 100, 255, THRESH_TOZERO_INV, FMT_U8, eDeviceType::CPU));
    TEST_CASE(TestCorrectness<uchar3>(2, 480, 360, 100, 255, THRESH_TOZERO_INV, FMT_RGB8, eDeviceType::CPU));
    TEST_CASE(TestCorrectness<uchar4>(3, 480, 120, 100, 255, THRESH_TOZERO_INV, FMT_RGBA8, eDeviceType::CPU));
    TEST_CASE(TestCorrectness<uchar3>(4, 134, 360, 100, 255, THRESH_TOZERO_INV, FMT_RGB8, eDeviceType::CPU));
    TEST_CASE(TestCorrectness<ushort1>(5, 134, 360, 100, 255, THRESH_TOZERO_INV, FMT_U16, eDeviceType::CPU));
    TEST_CASE(TestCorrectness<ushort3>(1, 480, 360, 100, 255, THRESH_TOZERO_INV, FMT_RGB16, eDeviceType::CPU));
    TEST_CASE(TestCorrectness<ushort4>(2, 480, 120, 100, 255, THRESH_TOZERO_INV, FMT_RGBA16, eDeviceType::CPU));
    TEST_CASE(TestCorrectness<short1>(5, 134, 360, 100, 255, THRESH_TOZERO_INV, FMT_S16, eDeviceType::CPU));
    TEST_CASE(TestCorrectness<short3>(1, 480, 360, 100, 255, THRESH_TOZERO_INV, FMT_RGBs16, eDeviceType::CPU));
    TEST_CASE(TestCorrectness<short4>(2, 480, 120, 100, 255, THRESH_TOZERO_INV, FMT_RGBAs16, eDeviceType::CPU));
    TEST_CASE(TestCorrectness<float1>(5, 134, 360, 100, 255, THRESH_TOZERO_INV, FMT_F32, eDeviceType::CPU));
    TEST_CASE(TestCorrectness<float3>(4, 134, 360, 100, 255, THRESH_TOZERO_INV, FMT_RGBf32, eDeviceType::CPU));
    TEST_CASE(TestCorrectness<float4>(2, 480, 120, 100, 255, THRESH_TOZERO_INV, FMT_RGBAf32, eDeviceType::CPU));
    TEST_CASE(TestCorrectness<double1>(5, 134, 360, 100, 255, THRESH_TOZERO_INV, FMT_F64, eDeviceType::CPU));
    TEST_CASE(TestCorrectness<double3>(4, 134, 360, 100, 255, THRESH_TOZERO_INV, FMT_RGBf64, eDeviceType::CPU));
    TEST_CASE(TestCorrectness<double4>(2, 480, 120, 100, 255, THRESH_TOZERO_INV, FMT_RGBAf64, eDeviceType::CPU));
    
    TEST_CASES_END();
}

/*
eTestStatusType testCorrectness(const std::string &inputFile, uint8_t *expectedData, eThresholdType thresh_type,
                                const eDeviceType device) {
    std::vector<double> threshData;
    threshData.assign(1, 100);
    std::vector<double> mvData;
    mvData.assign(1, 255);

    Tensor input = createTensorFromImage(inputFile, DataType(eDataType::DATA_TYPE_U8), device);
    Tensor output(input.shape(), input.dtype(), input.device());

    TensorShape thresh_param_shape(TensorLayout(eTensorLayout::TENSOR_LAYOUT_N), {1});
    TensorShape maxval_param_shape(TensorLayout(eTensorLayout::TENSOR_LAYOUT_N), {1});
    DataType param_dtype(eDataType::DATA_TYPE_F64);
    Tensor thresh(thresh_param_shape, param_dtype, device);
    Tensor maxval(maxval_param_shape, param_dtype, device);

    auto threshTensorData = thresh.exportData<TensorDataStrided>();
    auto maxvalTensorData = maxval.exportData<TensorDataStrided>();

    if (device == eDeviceType::GPU) {
        hipStream_t stream;
        HIP_VALIDATE_NO_ERRORS(hipStreamCreate(&stream));

        size_t thresh_size = thresh.shape().size() * thresh.dtype().size();
        HIP_VALIDATE_NO_ERRORS(
            hipMemcpy(threshTensorData.basePtr(), threshData.data(), thresh_size, hipMemcpyHostToDevice));

        size_t maxval_size = maxval.shape().size() * maxval.dtype().size();
        HIP_VALIDATE_NO_ERRORS(
            hipMemcpy(maxvalTensorData.basePtr(), mvData.data(), maxval_size, hipMemcpyHostToDevice));

        Threshold op(thresh_type, 1);
        op(stream, input, output, thresh, maxval, device);

        // Move image data back to device
        size_t image_size = input.shape().size() * input.dtype().size();
        auto outputTensorData = output.exportData<TensorDataStrided>();
        std::vector<uint8_t> d_output(image_size);
        HIP_VALIDATE_NO_ERRORS(
            hipMemcpyAsync(d_output.data(), outputTensorData.basePtr(), image_size, hipMemcpyDeviceToHost, stream));

        HIP_VALIDATE_NO_ERRORS(hipStreamSynchronize(stream));
        HIP_VALIDATE_NO_ERRORS(hipStreamDestroy(stream));

        for (int i = 0; i < output.shape().size(); i++) {
            float err = std::abs(d_output.data()[i] - expectedData[i]);

            if (err > 1) {
                std::cout << "Threshold(DEVICE) failed at index: " << i << " with an error of: " << err << std::endl;
                return eTestStatusType::UNEXPECTED_VALUE;
            }
        }
    } else if (device == eDeviceType::CPU) {
        memcpy(threshTensorData.basePtr(), threshData.data(), thresh.shape().size() * thresh.dtype().size());
        memcpy(maxvalTensorData.basePtr(), mvData.data(), maxval.shape().size() * maxval.dtype().size());

        Threshold op(thresh_type, 1);
        op(nullptr, input, output, thresh, maxval, device);

        size_t image_size = input.shape().size() * input.dtype().size();
        auto outputTensorData = output.exportData<TensorDataStrided>();
        std::vector<uint8_t> d_output(image_size);
        memcpy(d_output.data(), outputTensorData.basePtr(), output.shape().size() * output.dtype().size());
        for (int i = 0; i < output.shape().size(); i++) {
            float err = std::abs(d_output.data()[i] - static_cast<double>(expectedData[i]));
            if (err > 1) {
                std::cout << "Threshold(HOST) failed at index: " << i << " with an error of: " << err << std::endl;
                return eTestStatusType::UNEXPECTED_VALUE;
            }
        }
    }
    return eTestStatusType::TEST_SUCCESS;
}

eTestStatusType test_op_thresholding(int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <test data path>" << std::endl;
        return eTestStatusType::TEST_FAILURE;
    }
    fs::path testDataPath = fs::path(argv[1]) / "tests" / "ops";
    fs::path test_image_filepath = testDataPath / "test_input.bmp";

    try {
        // cv::Mat testData = cv::imread(testDataPath / "test_input.bmp");
        cv::Mat expectedDataBinary = cv::imread(testDataPath / "expected_thresholding_binary.bmp");
        cv::Mat expectedDataBinaryInv = cv::imread(testDataPath / "expected_thresholding_binary_inv.bmp");
        cv::Mat expectedDataTrunc = cv::imread(testDataPath / "expected_thresholding_trunc.bmp");
        cv::Mat expectedDataTozero = cv::imread(testDataPath / "expected_thresholding_to_zero.bmp");
        cv::Mat expectedDataTozeroinv = cv::imread(testDataPath / "expected_thresholding_to_zero_inv.bmp");
        

        EXPECT_TEST_STATUS(
            testCorrectness(test_image_filepath, expectedDataBinary.data, THRESH_BINARY, eDeviceType::GPU),
            eTestStatusType::TEST_SUCCESS);
        EXPECT_TEST_STATUS(
            testCorrectness(test_image_filepath, expectedDataBinaryInv.data, THRESH_BINARY_INV, eDeviceType::GPU),
            eTestStatusType::TEST_SUCCESS);
        EXPECT_TEST_STATUS(
            testCorrectness(test_image_filepath, expectedDataTrunc.data, THRESH_TRUNC, eDeviceType::GPU),
            eTestStatusType::TEST_SUCCESS);
        EXPECT_TEST_STATUS(
            testCorrectness(test_image_filepath, expectedDataTozero.data, THRESH_TOZERO, eDeviceType::GPU),
            eTestStatusType::TEST_SUCCESS);
        EXPECT_TEST_STATUS(
            testCorrectness(test_image_filepath, expectedDataTozeroinv.data, THRESH_TOZERO_INV, eDeviceType::GPU),
            eTestStatusType::TEST_SUCCESS);
        
        EXPECT_TEST_STATUS(
            testCorrectness(test_image_filepath, expectedDataBinary.data, THRESH_BINARY, eDeviceType::CPU),
            eTestStatusType::TEST_SUCCESS);
        EXPECT_TEST_STATUS(
            testCorrectness(test_image_filepath, expectedDataBinaryInv.data, THRESH_BINARY_INV, eDeviceType::CPU),
            eTestStatusType::TEST_SUCCESS);
        EXPECT_TEST_STATUS(
            testCorrectness(test_image_filepath, expectedDataTrunc.data, THRESH_TRUNC, eDeviceType::CPU),
            eTestStatusType::TEST_SUCCESS);
        EXPECT_TEST_STATUS(
            testCorrectness(test_image_filepath, expectedDataTozero.data, THRESH_TOZERO, eDeviceType::CPU),
            eTestStatusType::TEST_SUCCESS);
        EXPECT_TEST_STATUS(
            testCorrectness(test_image_filepath, expectedDataTozeroinv.data, THRESH_TOZERO_INV, eDeviceType::CPU),
            eTestStatusType::TEST_SUCCESS);

    } catch (Exception e) {
        std::cout << "Exception: " << e.what() << std::endl;
        return eTestStatusType::TEST_FAILURE;
    }
    return eTestStatusType::TEST_SUCCESS;
}
*/