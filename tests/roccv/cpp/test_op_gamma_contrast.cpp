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

#include <core/hip_assert.h>

#include <filesystem>
#include <iostream>
#include <op_gamma_contrast.hpp>
#include <opencv2/opencv.hpp>
#include <tuple>

#include "test_helpers.hpp"

using namespace roccv;
using namespace roccv::tests;
namespace fs = std::filesystem;

void test_gamma_contrast(const std::string& input_image, const std::string& expected_image, eDeviceType device) {
    Tensor input = createTensorFromImage(input_image, DataType(eDataType::DATA_TYPE_U8), device);
    Tensor output(input.shape(), input.dtype(), input.device());

    std::vector<float> gammaValues = {2.2};
    Tensor gammaTensor(TensorShape(TensorLayout(eTensorLayout::TENSOR_LAYOUT_N), {1}),
                       DataType(eDataType::DATA_TYPE_F32), device);
    auto gammaTensorData = gammaTensor.exportData<TensorDataStrided>();

    if (device == eDeviceType::GPU) {
        HIP_VALIDATE_NO_ERRORS(hipMemcpy(gammaTensorData.basePtr(), gammaValues.data(),
                                         gammaTensor.shape().size() * gammaTensor.dtype().size(),
                                         hipMemcpyHostToDevice));

        hipStream_t stream;
        HIP_VALIDATE_NO_ERRORS(hipStreamCreate(&stream));

        GammaContrast gamma_contrast;
        gamma_contrast(stream, input, output, gammaTensor, device);

        HIP_VALIDATE_NO_ERRORS(hipStreamSynchronize(stream));

        EXPECT_TEST_STATUS(compareImage(output, expected_image, 1.0f), eTestStatusType::TEST_SUCCESS);

        HIP_VALIDATE_NO_ERRORS(hipStreamDestroy(stream));
    } else if (device == eDeviceType::CPU) {
        memcpy(gammaTensorData.basePtr(), gammaValues.data(), gammaTensor.shape().size() * gammaTensor.dtype().size());

        GammaContrast gamma_contrast;
        gamma_contrast(nullptr, input, output, gammaTensor, device);

        EXPECT_TEST_STATUS(compareImage(output, expected_image, 1.0f), eTestStatusType::TEST_SUCCESS);
    }
}

eTestStatusType test_op_gamma_contrast(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <test data path>" << std::endl;
        return eTestStatusType::TEST_FAILURE;
    }
    fs::path test_data_path = fs::path(argv[1]) / "tests" / "ops";
    fs::path test_image_filepath = test_data_path / "test_input.bmp";
    fs::path expected_image_filepath = test_data_path / "expected_gamma_contrast.bmp";

    test_gamma_contrast(test_image_filepath, expected_image_filepath, eDeviceType::GPU);
    test_gamma_contrast(test_image_filepath, expected_image_filepath, eDeviceType::CPU);

    return eTestStatusType::TEST_SUCCESS;
}
