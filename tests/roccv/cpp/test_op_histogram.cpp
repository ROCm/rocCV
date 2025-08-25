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
#include <iostream>
#include <op_histogram.hpp>
#include "operator_types.h"

#include "test_helpers.hpp"

using namespace roccv;
using namespace roccv::tests;

// Keep all non-entrypoint functions in an anonymous namespace to prevent redefinition errors across translation units.
namespace {

/**
 * @brief Verified golden C++ model for the Histogram operation.
 *
 * @tparam T Vectorized datatype of the image's pixels.
 * @tparam BT Base type of the image's data.
 * @param[in] input An input vector containing image data.
 * @param[in] batchSize The number of images in the batch.
 * @param[in] width Image width.
 * @param[in] height Image height.
 * @param[in] channels Number of channels in the image.
 * @return Vector containing the results of the operation.
 */
template <typename T, typename BT = detail::BaseType<T>>
std::vector<BT> GoldenHistogram(std::vector<uchar>& input, int32_t batchSize, int32_t width, int32_t height) {
    // Create an output vector for the histogram
    std::vector<BT> output;
    std::vector<BT> local_histogram(256);

    // Wrap the input vector for simplified data access
    ImageWrapper<uchar> src(input, batchSize, width, height);

    for (int b = 0; b < batchSize; ++b) {
        std::fill(local_histogram.begin(), local_histogram.end(), 0);
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                auto hist_idx = src.at(b, y, x, 0);
                local_histogram[hist_idx]++;
            }
        }
        output.insert(output.end(), local_histogram.begin(), local_histogram.end());
    }
    return output;
}

/**
 * @brief Verified golden C++ model for the Histogram operation with the mask.
 *
 * @tparam T Vectorized datatype of the image's pixels.
 * @tparam BT Base type of the image's data.
 * @param[in] input An input vector containing image data.
 * @param[in] mask An input vector containing the mask data.
 * @param[in] batchSize The number of images in the batch.
 * @param[in] width Image width.
 * @param[in] height Image height.
 * @param[in] channels Number of channels in the image.
 * @return Vector containing the results of the operation.
 */

template <typename T, typename BT = detail::BaseType<T>>
std::vector<BT> GoldenHistogramMask(std::vector<uchar>& input, std::vector<uchar>& mask, int32_t batchSize, int32_t width, int32_t height) {
    // Create an output vector for the histogram
    std::vector<BT> output;
    std::vector<BT> local_histogram(256);

    // Wrap input/mask vectors for simplified data access
    ImageWrapper<uchar> src(input, batchSize, width, height);
    ImageWrapper<uchar> maskWrap(mask, batchSize, width, height);

    for (int b = 0; b < batchSize; ++b) {
        std::fill(local_histogram.begin(), local_histogram.end(), 0);
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                if (maskWrap.at(b, y, x, 0) != 0) {
                    auto hist_idx = src.at(b, y, x, 0);
                    local_histogram[hist_idx]++;
                }
            }
        }
        output.insert(output.end(), local_histogram.begin(), local_histogram.end());
    }
    return output;
}

/**
 * @brief Tests correctness of the Histogram operator, comparing it against a generated golden result.
 *
 * @tparam T Underlying datatype of the image's pixels.
 * @tparam BT Base type of the image data.
 * @param[in] batchSize The number of images in the batch.
 * @param[in] width The width of each image in the batch.
 * @param[in] height The height of each image in the batch.
 * @param[in] format The image format.
 * @param[in] device The device this correctness test should be run on.
 */
template <typename T, typename BT = detail::BaseType<T>>
void TestCorrectness(int batchSize, int width, int height, ImageFormat format, eDeviceType device) {
    // Create input and output tensor based on test parameters
    Tensor input(batchSize, {width, height}, format, device);
    Tensor mask(batchSize, {width, height}, format, device);
    Tensor histogram(TensorShape(TensorLayout(eTensorLayout::TENSOR_LAYOUT_HWC), {batchSize, 256, 1}),
                  DataType(eDataType::DATA_TYPE_S32), device);
    Tensor histogramWithMask(TensorShape(TensorLayout(eTensorLayout::TENSOR_LAYOUT_HWC), {batchSize, 256, 1}),
                  DataType(eDataType::DATA_TYPE_S32), device);
    
    // Create a vector and fill it with random data.
    std::vector<uchar> inputData(input.shape().size());
    FillVector(inputData);

    // Create a mask vector and fill it with random data.
    std::vector<uchar> maskData(mask.shape().size());
    FillVectorMask(maskData);

    // Copy generated input data into input tensor
    CopyVectorIntoTensor(input, inputData);

    // Copy generated mask data into mask tensor
    CopyVectorIntoTensor(mask, maskData);

    // Calculate golden output reference
    std::vector<BT> ref = GoldenHistogram<T>(inputData, batchSize, width, height);

    // Calculate golden output reference with mask
    std::vector<BT> maskRef = GoldenHistogramMask<T>(inputData, maskData, batchSize, width, height);

    // Run roccv::Histogram operator to obtain actual results
    hipStream_t stream;
    HIP_VALIDATE_NO_ERRORS(hipStreamCreate(&stream));

    Histogram op;
    op(stream, input, std::nullopt, histogram, device);
    HIP_VALIDATE_NO_ERRORS(hipStreamSynchronize(stream));
    HIP_VALIDATE_NO_ERRORS(hipStreamDestroy(stream));

    // Copy data from histogram tensor into a host allocated vector
    std::vector<BT> result(histogram.shape().size());
    CopyTensorIntoVector(result, histogram);

    // Run roccv::Histogram operator with mask to obtain actual results
    hipStream_t streamMask;
    HIP_VALIDATE_NO_ERRORS(hipStreamCreate(&streamMask));

    Histogram maskOp;
    maskOp(stream, input, mask, histogramWithMask, device);
    HIP_VALIDATE_NO_ERRORS(hipStreamSynchronize(streamMask));
    HIP_VALIDATE_NO_ERRORS(hipStreamDestroy(streamMask));

    // Copy data from histogramWithMask tensor into a host allocated vector
    std::vector<BT> maskResult(histogramWithMask.shape().size());
    CopyTensorIntoVector(maskResult, histogramWithMask);

    // Compare data in actual output versus the generated golden reference image
    CompareVectors(result, ref);
    CompareVectors(maskResult, maskRef);
}
} // namespace

eTestStatusType test_op_histogram(int argc, char** argv) {
    TEST_CASES_BEGIN();

    // GPU correctness tests
    TEST_CASE(TestCorrectness<uint1>(1, 360, 360, FMT_U8, eDeviceType::GPU));
    TEST_CASE(TestCorrectness<uint1>(2, 134, 360, FMT_U8, eDeviceType::GPU));
    TEST_CASE(TestCorrectness<uint1>(3, 480, 120, FMT_U8, eDeviceType::GPU));
    TEST_CASE(TestCorrectness<uint1>(4, 360, 360, FMT_U8, eDeviceType::GPU));
    TEST_CASE(TestCorrectness<uint1>(5, 134, 360, FMT_U8, eDeviceType::GPU));
    TEST_CASE(TestCorrectness<int1>(1, 360, 360, FMT_U8, eDeviceType::GPU));
    TEST_CASE(TestCorrectness<int1>(2, 134, 360, FMT_U8, eDeviceType::GPU));
    TEST_CASE(TestCorrectness<int1>(3, 480, 120, FMT_U8, eDeviceType::GPU));
    TEST_CASE(TestCorrectness<int1>(4, 360, 360, FMT_U8, eDeviceType::GPU));
    TEST_CASE(TestCorrectness<int1>(5, 134, 360, FMT_U8, eDeviceType::GPU));

    // CPU correctness tests
    TEST_CASE(TestCorrectness<uint1>(1, 360, 360, FMT_U8, eDeviceType::CPU));
    TEST_CASE(TestCorrectness<uint1>(2, 134, 360, FMT_U8, eDeviceType::CPU));
    TEST_CASE(TestCorrectness<uint1>(3, 480, 120, FMT_U8, eDeviceType::CPU));
    TEST_CASE(TestCorrectness<uint1>(4, 360, 360, FMT_U8, eDeviceType::CPU));
    TEST_CASE(TestCorrectness<uint1>(5, 134, 360, FMT_U8, eDeviceType::CPU));
    TEST_CASE(TestCorrectness<int1>(1, 360, 360, FMT_U8, eDeviceType::CPU));
    TEST_CASE(TestCorrectness<int1>(2, 134, 360, FMT_U8, eDeviceType::CPU));
    TEST_CASE(TestCorrectness<int1>(3, 480, 120, FMT_U8, eDeviceType::CPU));
    TEST_CASE(TestCorrectness<int1>(4, 360, 360, FMT_U8, eDeviceType::CPU));
    TEST_CASE(TestCorrectness<int1>(5, 134, 360, FMT_U8, eDeviceType::CPU));

    TEST_CASES_END();
}









































/*
namespace {
eTestStatusType testCorrectness(const std::string &inputFile, int32_t *expectedData, const eDeviceType device) {
    cv::Mat testData = cv::imread(inputFile);
    cv::cvtColor(testData, testData, cv::COLOR_BGR2GRAY);

    TensorShape shape(TensorLayout(eTensorLayout::TENSOR_LAYOUT_NHWC),
                      {1, testData.rows, testData.cols, testData.channels()});
    DataType dtype(eDataType::DATA_TYPE_U8);

    Tensor input(shape, dtype, device);
    size_t image_size = input.shape().size() * input.dtype().size();
    auto d_input_data = input.exportData<TensorDataStrided>();

    std::vector<int32_t> outData;
    outData.assign(256, 0);
    std::vector<int32_t> resultData;
    resultData.assign(256, 0);

    Tensor output(TensorShape(TensorLayout(eTensorLayout::TENSOR_LAYOUT_HWC), {1, 256, 1}),
                  DataType(eDataType::DATA_TYPE_S32), device);

    if (device == eDeviceType::GPU) {
        HIP_VALIDATE_NO_ERRORS(hipMemcpy(d_input_data.basePtr(), testData.data, image_size, hipMemcpyHostToDevice));

        hipStream_t stream;
        HIP_VALIDATE_NO_ERRORS(hipStreamCreate(&stream));

        Histogram op;
        op(stream, input, std::nullopt, output, device);

        auto outputTensorData = output.exportData<TensorDataStrided>();
        HIP_VALIDATE_NO_ERRORS(hipMemcpy(resultData.data(), outputTensorData.basePtr(),
                                         output.shape().size() * output.dtype().size(), hipMemcpyDeviceToHost));

        HIP_VALIDATE_NO_ERRORS(hipStreamSynchronize(stream));
        HIP_VALIDATE_NO_ERRORS(hipStreamDestroy(stream));

        for (int i = 0; i < output.shape().size(); i++) {
            float err = std::abs(static_cast<int32_t>(resultData[i] - expectedData[i]));

            if (err > 1) {
                std::cout << "OpHistogram (DEVICE) failed at index: " << i << " with an error of: " << err << std::endl;
                return eTestStatusType::UNEXPECTED_VALUE;
            }
        }
    } else if (device == eDeviceType::CPU) {
        memcpy(d_input_data.basePtr(), testData.data, image_size);

        Histogram op;
        op(nullptr, input, std::nullopt, output, device);

        auto outputTensorData = output.exportData<TensorDataStrided>();

        memcpy(resultData.data(), outputTensorData.basePtr(), output.shape().size() * output.dtype().size());

        for (int i = 0; i < output.shape().size(); i++) {
            float err = std::abs(static_cast<int32_t>(resultData[i] - expectedData[i]));

            if (err > 1) {
                std::cout << "OpHistogram (HOST) failed at index: " << i << " with an error of: " << err << std::endl;
                return eTestStatusType::UNEXPECTED_VALUE;
            }
        }
    }
    return eTestStatusType::TEST_SUCCESS;
}
}  // namespace

eTestStatusType test_op_histogram(int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <test data path>" << std::endl;
        return eTestStatusType::TEST_FAILURE;
    }
    fs::path testDataPath = fs::path(argv[1]) / "tests" / "ops";

    try {
        int32_t *expectedData;
        size_t expSize;

        const std::string histogramBinFile = testDataPath / "expected_histogram.bin";

        if (!rocCVBinaryIO::read_array_size(histogramBinFile, expSize)) {
            std::cerr << "OpHistogram : read_array_size() failed for : " << histogramBinFile << std::endl;
            return eTestStatusType::UNEXPECTED_VALUE;
        }

        if (expSize % 256 != 0 || expSize < 256) {
            std::cerr << "OpHistogram : Wrong number of bin size in expected output :" << expSize << std::endl;
            return eTestStatusType::UNEXPECTED_VALUE;
        }

        expectedData = (int32_t *)malloc(expSize * sizeof(int32_t));
        if (!rocCVBinaryIO::read_array(histogramBinFile, expectedData, expSize)) {
            std::cerr << "OpHistogram : read_array() failed for : " << histogramBinFile << std::endl;
            return eTestStatusType::UNEXPECTED_VALUE;
        }

        EXPECT_TEST_STATUS(testCorrectness(testDataPath / "test_input.bmp", expectedData, eDeviceType::GPU),
                           eTestStatusType::TEST_SUCCESS);
        EXPECT_TEST_STATUS(testCorrectness(testDataPath / "test_input.bmp", expectedData, eDeviceType::CPU),
                           eTestStatusType::TEST_SUCCESS);
        free(expectedData);
    } catch (Exception e) {
        std::cout << "Exception: " << e.what() << std::endl;
        return eTestStatusType::TEST_FAILURE;
    }
    return eTestStatusType::TEST_SUCCESS;
}*/