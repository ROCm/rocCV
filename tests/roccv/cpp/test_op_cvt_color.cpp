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
eTestStatusType testCorrectness(const std::string &inputFile, uint8_t *expectedData, eColorConversionCode conversionCode, eDeviceType device) {

    cv::Mat image_data = cv::imread(inputFile);
    TensorShape shape_clr(TensorLayout(eTensorLayout::TENSOR_LAYOUT_NHWC), {1, image_data.rows, image_data.cols, image_data.channels()});
    TensorShape shape_gry(TensorLayout(eTensorLayout::TENSOR_LAYOUT_NHWC), {1, image_data.rows, image_data.cols, 1});
    DataType dtype(eDataType::DATA_TYPE_U8);
    Tensor input__clr(shape_clr, dtype, device);
    Tensor output_clr(shape_clr, input__clr.dtype(), input__clr.device());
    Tensor output_gry(shape_gry, input__clr.dtype(), input__clr.device());

    size_t image_size = input__clr.shape().size() * input__clr.dtype().size();
    auto inputData = input__clr.exportData<TensorDataStrided>();
    CvtColor op;
    int tst_limit = 0;
    hipStream_t stream = static_cast<hipStream_t>(nullptr);
    if(device == eDeviceType::GPU) {
        HIP_VALIDATE_NO_ERRORS(hipStreamCreate(&stream));
        HIP_VALIDATE_NO_ERRORS(hipMemcpyAsync(inputData.basePtr(), image_data.data, image_size, hipMemcpyHostToDevice, stream));
    } else { // CPU
        memcpy(inputData.basePtr(), image_data.data, image_size);
    }
    size_t output_image_size = 0;
    std::vector<uint8_t> resultData;
    std::optional<TensorDataStrided> outputTensorData;

    switch(conversionCode) {
        case eColorConversionCode::COLOR_RGB2GRAY:
        case eColorConversionCode::COLOR_BGR2GRAY:
        {
            op(stream, input__clr, output_gry, conversionCode, device);
            output_image_size = output_gry.shape().size() * output_clr.dtype().size();
            outputTensorData.emplace(output_gry.exportData<TensorDataStrided>());
            resultData.resize(output_image_size);
            if(device == eDeviceType::CPU) {
                auto* base = outputTensorData->basePtr();
                memcpy(resultData.data(), base, output_gry.shape().size() * output_gry.dtype().size());
                tst_limit = output_gry.shape().size();
            }
        }
        break;

        default: // COLOR = any other than GRAY
        {
            op(stream, input__clr, output_clr, conversionCode, device);
            output_image_size = output_clr.shape().size() * output_clr.dtype().size();
            outputTensorData.emplace(output_clr.exportData<TensorDataStrided>());
            resultData.resize(output_image_size);
            if(device == eDeviceType::CPU) {
                auto* base = outputTensorData->basePtr();
                memcpy(resultData.data(), base, output_clr.shape().size() * output_clr.dtype().size());
                tst_limit = output_clr.shape().size();
            }
        }
        break;
    }

    // test on GPU
    if(device == eDeviceType::GPU) {
        auto* base = outputTensorData->basePtr();
        HIP_VALIDATE_NO_ERRORS(hipMemcpyAsync(resultData.data(), base, output_image_size, hipMemcpyDeviceToHost, stream));
        HIP_VALIDATE_NO_ERRORS(hipStreamSynchronize(stream));
        tst_limit = output_image_size;
    }
    // final test
    for (int i = 0; i < tst_limit; i++) {
        float err = std::abs(resultData[i] - expectedData[i]);
        if (err > 1) {
            std::cout << "CvtColor " << ((device == eDeviceType::GPU) ? "(GPU)" : "(HOST)") << " failed at index: " << i << " with an error of: " << err << std::endl;
            return eTestStatusType::UNEXPECTED_VALUE;
        }
    }
    // clean up
    if(device == eDeviceType::GPU) {
        HIP_VALIDATE_NO_ERRORS(hipStreamDestroy(stream));
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
    fs::path tst_i = testDataPath / "test_input.bmp";
    fs::path tst_y = testDataPath / "expected_color_cvt_yuv.bmp";
    eTestStatusType success = eTestStatusType::TEST_SUCCESS;
    eDeviceType gpu = eDeviceType::GPU;
    eDeviceType cpu = eDeviceType::CPU;

    try {
        cv::Mat expectedDataYUV = cv::imread(testDataPath / "expected_color_cvt_yuv.bmp");
        cv::Mat expectedDataBGR = cv::imread(testDataPath / "expected_color_cvt_bgr.bmp");
        cv::Mat expectedDataRGB = cv::imread(testDataPath / "expected_color_cvt_rgb.bmp");
        cv::Mat expectedDataGrayscale = cv::imread(testDataPath / "expected_color_cvt_grayscale.bmp", cv::IMREAD_UNCHANGED);

        EXPECT_TEST_STATUS(testCorrectness(tst_i, expectedDataYUV.data, eColorConversionCode::COLOR_BGR2YUV, gpu), success);
        EXPECT_TEST_STATUS(testCorrectness(tst_y, expectedDataBGR.data, eColorConversionCode::COLOR_YUV2BGR, gpu), success);
        EXPECT_TEST_STATUS(testCorrectness(tst_i, expectedDataRGB.data, eColorConversionCode::COLOR_BGR2RGB, gpu), success);
        EXPECT_TEST_STATUS(testCorrectness(tst_i, expectedDataGrayscale.data, eColorConversionCode::COLOR_BGR2GRAY, gpu), success);
        EXPECT_TEST_STATUS(testCorrectness(tst_i, expectedDataYUV.data, eColorConversionCode::COLOR_BGR2YUV, cpu), success);
        EXPECT_TEST_STATUS(testCorrectness(tst_y, expectedDataBGR.data, eColorConversionCode::COLOR_YUV2BGR, cpu), success);
        EXPECT_TEST_STATUS(testCorrectness(tst_i, expectedDataRGB.data, eColorConversionCode::COLOR_BGR2RGB, cpu), success);
        EXPECT_TEST_STATUS(testCorrectness(tst_i, expectedDataGrayscale.data, eColorConversionCode::COLOR_BGR2GRAY, cpu), success);
    } catch (Exception e) {
        std::cout << "Exception: " << e.what() << std::endl;
        return eTestStatusType::TEST_FAILURE;
    }
    return success;
}
