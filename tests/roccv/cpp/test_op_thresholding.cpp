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
#include <filesystem>
#include <iostream>
#include <op_thresholding.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

#include "test_helpers.hpp"

using namespace roccv;
using namespace roccv::tests;
namespace fs = std::filesystem;

// test_status_type testCorrectness(Tensor &input, uint8_t *expectedData, eThresholdType thresh_type) {
eTestStatusType testCorrectness(const std::string &inputFile, uint8_t *expectedData, eThresholdType thresh_type,
                                const eDeviceType device) {
    std::vector<uint8_t> threshData;
    threshData.assign(1, 100);
    std::vector<uint8_t> mvData;
    mvData.assign(1, 255);

    Tensor input = createTensorFromImage(inputFile, DataType(eDataType::DATA_TYPE_U8), device);
    Tensor output(input.shape(), input.dtype(), input.device());

    TensorShape thresh_param_shape(TensorLayout(eTensorLayout::TENSOR_LAYOUT_NW), {1, 1});
    TensorShape maxval_param_shape(TensorLayout(eTensorLayout::TENSOR_LAYOUT_NW), {1, 1});
    DataType param_dtype(eDataType::DATA_TYPE_U8);
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
            float err = std::abs(d_output.data()[i] - expectedData[i]);
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
        cv::Mat expectedDataTozero = cv::imread(testDataPath / "expected_thresholding_to_zero.bmp");
        cv::Mat expectedDataTozeroinv = cv::imread(testDataPath / "expected_thresholding_to_zero_inv.bmp");
        cv::Mat expectedDataTrunc = cv::imread(testDataPath / "expected_thresholding_trunc.bmp");

        EXPECT_TEST_STATUS(
            testCorrectness(test_image_filepath, expectedDataBinary.data, THRESH_BINARY, eDeviceType::GPU),
            eTestStatusType::TEST_SUCCESS);
        EXPECT_TEST_STATUS(
            testCorrectness(test_image_filepath, expectedDataBinaryInv.data, THRESH_BINARY_INV, eDeviceType::GPU),
            eTestStatusType::TEST_SUCCESS);
        EXPECT_TEST_STATUS(
            testCorrectness(test_image_filepath, expectedDataTozero.data, THRESH_TOZERO, eDeviceType::GPU),
            eTestStatusType::TEST_SUCCESS);
        EXPECT_TEST_STATUS(
            testCorrectness(test_image_filepath, expectedDataTozeroinv.data, THRESH_TOZERO_INV, eDeviceType::GPU),
            eTestStatusType::TEST_SUCCESS);
        EXPECT_TEST_STATUS(testCorrectness(test_image_filepath, expectedDataTrunc.data, THRESH_TRUNC, eDeviceType::GPU),
                           eTestStatusType::TEST_SUCCESS);

        EXPECT_TEST_STATUS(
            testCorrectness(test_image_filepath, expectedDataBinary.data, THRESH_BINARY, eDeviceType::CPU),
            eTestStatusType::TEST_SUCCESS);
        EXPECT_TEST_STATUS(
            testCorrectness(test_image_filepath, expectedDataBinaryInv.data, THRESH_BINARY_INV, eDeviceType::CPU),
            eTestStatusType::TEST_SUCCESS);
        EXPECT_TEST_STATUS(
            testCorrectness(test_image_filepath, expectedDataTozero.data, THRESH_TOZERO, eDeviceType::CPU),
            eTestStatusType::TEST_SUCCESS);
        EXPECT_TEST_STATUS(
            testCorrectness(test_image_filepath, expectedDataTozeroinv.data, THRESH_TOZERO_INV, eDeviceType::CPU),
            eTestStatusType::TEST_SUCCESS);
        EXPECT_TEST_STATUS(testCorrectness(test_image_filepath, expectedDataTrunc.data, THRESH_TRUNC, eDeviceType::CPU),
                           eTestStatusType::TEST_SUCCESS);

    } catch (Exception e) {
        std::cout << "Exception: " << e.what() << std::endl;
        return eTestStatusType::TEST_FAILURE;
    }
    return eTestStatusType::TEST_SUCCESS;
}
