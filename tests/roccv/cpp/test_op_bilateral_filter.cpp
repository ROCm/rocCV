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
#include <op_bilateral_filter.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

#include "test_helpers.hpp"

using namespace roccv;
using namespace roccv::tests;

eTestStatusType testCorrectness(const std::string &inputFile, uint8_t *expectedData, const eDeviceType device) {
    cv::Mat testData = cv::imread(inputFile);
    // Create input/output tensors for the image.
    Tensor input = createTensorFromImage(inputFile, DataType(eDataType::DATA_TYPE_U8), device);
    Tensor output(input.shape(), input.dtype(), device);
    eBorderType border_type = BORDER_TYPE_CONSTANT;
    const float4 border_value = {255, 0, 255, 0};

    if (device == eDeviceType::GPU) {
        hipStream_t stream;
        HIP_VALIDATE_NO_ERRORS(hipStreamCreate(&stream));

        BilateralFilter op;
        op(stream, input, output, 30, 75, 75, border_type, border_value, device);

        auto outputTensorData = output.exportData<TensorDataStrided>();
        std::vector<uint8_t> resultData;
        resultData.assign(output.shape().size(), 0);

        HIP_VALIDATE_NO_ERRORS(hipMemcpyAsync(resultData.data(), outputTensorData.basePtr(),
                                              output.shape().size() * output.dtype().size(), hipMemcpyDeviceToHost,
                                              stream));

        HIP_VALIDATE_NO_ERRORS(hipStreamSynchronize(stream));
        for (int i = 0; i < output.shape().size(); i++) {
            float err = std::abs(resultData[i] - expectedData[i]);

            if (err > 255 * 0.05f) {
                std::cout << "Bilateral Filter (DEVICE) failed at index: " << i << " with an error of: " << err
                          << std::endl;
                return eTestStatusType::UNEXPECTED_VALUE;
            }
        }
    } else if (device == eDeviceType::CPU) {
        BilateralFilter op;
        op(nullptr, input, output, 30, 75, 75, border_type, border_value, device);

        auto outputTensorData = output.exportData<TensorDataStrided>();
        std::vector<uint8_t> resultData;
        resultData.assign(output.shape().size(), 0);

        memcpy(resultData.data(), outputTensorData.basePtr(), output.shape().size() * output.dtype().size());

        for (int i = 0; i < output.shape().size(); i++) {
            float err = std::abs(resultData[i] - expectedData[i]);

            if (err > 255 * 0.05f) {
                std::cout << "Bilateral Filter (HOST) failed at index: " << i << " with an error of: " << err
                          << std::endl;
                return eTestStatusType::UNEXPECTED_VALUE;
            }
        }
    }
    return eTestStatusType::TEST_SUCCESS;
}

eTestStatusType test_op_bilateral_filter(int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <test data path>" << std::endl;
        return eTestStatusType::TEST_FAILURE;
    }
    std::filesystem::path testDataPath = std::filesystem::path(argv[1]) / "tests" / "ops";

    try {
        cv::Mat expectedData = cv::imread(testDataPath / "expected_bilateral_filter.bmp");

        EXPECT_TEST_STATUS(testCorrectness(testDataPath / "test_input.bmp", expectedData.data, eDeviceType::GPU),
                           eTestStatusType::TEST_SUCCESS);
        EXPECT_TEST_STATUS(testCorrectness(testDataPath / "test_input.bmp", expectedData.data, eDeviceType::CPU),
                           eTestStatusType::TEST_SUCCESS);
    } catch (Exception e) {
        std::cout << "Exception: " << e.what() << std::endl;
        return eTestStatusType::TEST_FAILURE;
    }
    return eTestStatusType::TEST_SUCCESS;
}
