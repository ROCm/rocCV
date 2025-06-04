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

#include <filesystem>
#include <iostream>
#include <op_cvt_color.hpp>
#include <opencv2/opencv.hpp>
#include <test_helpers.hpp>
#include <vector>

using namespace roccv;
using namespace roccv::tests;
namespace fs = std::filesystem;

namespace {
eTestStatusType testCorrectness(const std::string &inputFile, uint8_t *expectedData,
                                eColorConversionCode conversionCode, eDeviceType device) {
    cv::Mat image_data = cv::imread(inputFile);

    TensorShape shape(TensorLayout(eTensorLayout::TENSOR_LAYOUT_NHWC),
                      {1, image_data.rows, image_data.cols, image_data.channels()});

    TensorShape shape_grayscale(TensorLayout(eTensorLayout::TENSOR_LAYOUT_NHWC),
                                {1, image_data.rows, image_data.cols, 1});

    DataType dtype(eDataType::DATA_TYPE_U8);

    Tensor input(shape, dtype, device);
    Tensor output(shape, input.dtype(), input.device());
    Tensor output_grayscale(shape_grayscale, input.dtype(), input.device());

    if (device == eDeviceType::GPU) {
        hipStream_t stream;
        HIP_VALIDATE_NO_ERRORS(hipStreamCreate(&stream));

        size_t image_size = input.shape().size() * input.dtype().size();
        auto inputData = input.exportData<TensorDataStrided>();
        HIP_VALIDATE_NO_ERRORS(
            hipMemcpyAsync(inputData.basePtr(), image_data.data, image_size, hipMemcpyHostToDevice, stream));

        CvtColor op;
        if (conversionCode == eColorConversionCode::COLOR_RGB2GRAY ||
            conversionCode == eColorConversionCode::COLOR_BGR2GRAY) {
            op(stream, input, output_grayscale, conversionCode, device);
            size_t output_image_size = output_grayscale.shape().size() * output.dtype().size();
            auto outputTensorData = output_grayscale.exportData<TensorDataStrided>();
            std::vector<uint8_t> resultData(output_image_size);
            HIP_VALIDATE_NO_ERRORS(hipMemcpyAsync(resultData.data(), outputTensorData.basePtr(), output_image_size,
                                                  hipMemcpyDeviceToHost, stream));
            for (int i = 0; i < output_image_size; i++) {
                float err = std::abs(resultData[i] - expectedData[i]);
                if (err > 1) {
                    std::cout << "CvtColor (DEVICE) failed at index: " << i << " with an error of: " << err
                              << std::endl;
                    return eTestStatusType::UNEXPECTED_VALUE;
                }
            }
        } else {
            op(stream, input, output, conversionCode, device);
            size_t output_image_size = output.shape().size() * output.dtype().size();
            auto outputTensorData = output.exportData<TensorDataStrided>();
            std::vector<uint8_t> resultData(output_image_size);
            HIP_VALIDATE_NO_ERRORS(hipMemcpyAsync(resultData.data(), outputTensorData.basePtr(), output_image_size,
                                                  hipMemcpyDeviceToHost, stream));
            for (int i = 0; i < output_image_size; i++) {
                float err = std::abs(resultData[i] - expectedData[i]);
                if (err > 1) {
                    std::cout << "CvtColor (DEVICE) failed at index: " << i << " with an error of: " << err
                              << std::endl;
                    return eTestStatusType::UNEXPECTED_VALUE;
                }
            }
        }
        HIP_VALIDATE_NO_ERRORS(hipStreamSynchronize(stream));
    } else if (device == eDeviceType::CPU) {
        size_t image_size = input.shape().size() * input.dtype().size();
        auto inputData = input.exportData<TensorDataStrided>();
        memcpy(inputData.basePtr(), image_data.data, image_size);

        CvtColor op;
        if (conversionCode == eColorConversionCode::COLOR_RGB2GRAY ||
            conversionCode == eColorConversionCode::COLOR_BGR2GRAY) {
            op(nullptr, input, output_grayscale, conversionCode, device);
            size_t output_image_size = output_grayscale.shape().size() * output.dtype().size();
            auto outputTensorData = output_grayscale.exportData<TensorDataStrided>();
            std::vector<uint8_t> resultData(output_image_size);
            memcpy(resultData.data(), outputTensorData.basePtr(),
                   output_grayscale.shape().size() * output_grayscale.dtype().size());
            for (int i = 0; i < output_grayscale.shape().size(); i++) {
                float err = std::abs(resultData[i] - expectedData[i]);
                if (err > 1) {
                    std::cout << "CvtColor (HOST) failed at index: " << i << " with an error of: " << err << std::endl;
                    return eTestStatusType::UNEXPECTED_VALUE;
                }
            }
        } else {
            op(nullptr, input, output, conversionCode, device);
            size_t output_image_size = output.shape().size() * output.dtype().size();
            auto outputTensorData = output.exportData<TensorDataStrided>();
            std::vector<uint8_t> resultData(output_image_size);
            memcpy(resultData.data(), outputTensorData.basePtr(), output.shape().size() * output.dtype().size());
            for (int i = 0; i < output.shape().size(); i++) {
                float err = std::abs(resultData[i] - expectedData[i]);
                if (err > 1) {
                    std::cout << "CvtColor (HOST) failed at index: " << i << " with an error of: " << err << std::endl;
                    return eTestStatusType::UNEXPECTED_VALUE;
                }
            }
        }
    }
    return eTestStatusType::TEST_SUCCESS;
}
}  // namespace

eTestStatusType test_op_cvt_color(int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <test data path>" << std::endl;
        return eTestStatusType::TEST_FAILURE;
    }
    fs::path testDataPath = fs::path(argv[1]) / "tests" / "ops";

    try {
        cv::Mat expectedDataYUV = cv::imread(testDataPath / "expected_color_cvt_yuv.bmp");
        cv::Mat expectedDataBGR = cv::imread(testDataPath / "expected_color_cvt_bgr.bmp");
        cv::Mat expectedDataRGB = cv::imread(testDataPath / "expected_color_cvt_rgb.bmp");
        cv::Mat expectedDataGrayscale =
            cv::imread(testDataPath / "expected_color_cvt_grayscale.bmp", cv::IMREAD_UNCHANGED);

        EXPECT_TEST_STATUS(testCorrectness(testDataPath / "test_input.bmp", expectedDataYUV.data,
                                           eColorConversionCode::COLOR_BGR2YUV, eDeviceType::GPU),
                           eTestStatusType::TEST_SUCCESS);
        EXPECT_TEST_STATUS(testCorrectness(testDataPath / "expected_color_cvt_yuv.bmp", expectedDataBGR.data,
                                           eColorConversionCode::COLOR_YUV2BGR, eDeviceType::GPU),
                           eTestStatusType::TEST_SUCCESS);
        EXPECT_TEST_STATUS(testCorrectness(testDataPath / "test_input.bmp", expectedDataRGB.data,
                                           eColorConversionCode::COLOR_BGR2RGB, eDeviceType::GPU),
                           eTestStatusType::TEST_SUCCESS);
        EXPECT_TEST_STATUS(testCorrectness(testDataPath / "test_input.bmp", expectedDataGrayscale.data,
                                           eColorConversionCode::COLOR_BGR2GRAY, eDeviceType::GPU),
                           eTestStatusType::TEST_SUCCESS);
        EXPECT_TEST_STATUS(testCorrectness(testDataPath / "test_input.bmp", expectedDataYUV.data,
                                           eColorConversionCode::COLOR_BGR2YUV, eDeviceType::CPU),
                           eTestStatusType::TEST_SUCCESS);
        EXPECT_TEST_STATUS(testCorrectness(testDataPath / "expected_color_cvt_yuv.bmp", expectedDataBGR.data,
                                           eColorConversionCode::COLOR_YUV2BGR, eDeviceType::CPU),
                           eTestStatusType::TEST_SUCCESS);
        EXPECT_TEST_STATUS(testCorrectness(testDataPath / "test_input.bmp", expectedDataRGB.data,
                                           eColorConversionCode::COLOR_BGR2RGB, eDeviceType::CPU),
                           eTestStatusType::TEST_SUCCESS);
        EXPECT_TEST_STATUS(testCorrectness(testDataPath / "test_input.bmp", expectedDataGrayscale.data,
                                           eColorConversionCode::COLOR_BGR2GRAY, eDeviceType::CPU),
                           eTestStatusType::TEST_SUCCESS);
    } catch (Exception e) {
        std::cout << "Exception: " << e.what() << std::endl;
        return eTestStatusType::TEST_FAILURE;
    }
    return eTestStatusType::TEST_SUCCESS;
}
