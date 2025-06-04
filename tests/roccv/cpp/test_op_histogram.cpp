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
#include <op_histogram.hpp>
#include <opencv2/opencv.hpp>
#include <roccv_binio.hpp>
#include <vector>

#include "test_helpers.hpp"

using namespace roccv;
using namespace roccv::tests;
namespace fs = std::filesystem;

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
        op(stream, input, nullptr, output, device);

        auto outputTensorData = output.exportData<TensorDataStrided>();
        HIP_VALIDATE_NO_ERRORS(hipMemcpy(resultData.data(), outputTensorData.basePtr(),
                                         output.shape().size() * output.dtype().size(), hipMemcpyDeviceToHost));

        HIP_VALIDATE_NO_ERRORS(hipStreamSynchronize(stream));

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
        op(nullptr, input, nullptr, output, device);

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
}