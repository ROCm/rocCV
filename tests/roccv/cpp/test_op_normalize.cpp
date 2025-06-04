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
#include <op_normalize.hpp>
#include <opencv2/opencv.hpp>
#include <tuple>

#include "test_helpers.hpp"

using namespace roccv;
using namespace roccv::tests;

eTestStatusType TestCorrectness(const std::string& input_image, const std::string& expected_image,
                                const std::vector<float>& base, const std::vector<float>& scale, float global_scale,
                                float shift, float epsilon, uint32_t flags, eDeviceType device) {
    Tensor input = createTensorFromImage(input_image, DataType(eDataType::DATA_TYPE_U8), device);
    Tensor output(input.shape(), input.dtype(), input.device());

    int64_t num_channels = input.shape(input.layout().channels_index());

    TensorShape paramShape(TensorLayout(eTensorLayout::TENSOR_LAYOUT_NHWC), {1, 1, 1, num_channels});
    DataType paramDtype(eDataType::DATA_TYPE_F32);
    Tensor base_tensor(paramShape, paramDtype, device);
    Tensor scale_tensor(paramShape, paramDtype, device);

    switch (device) {
        case eDeviceType::GPU: {
            hipMemcpy(base_tensor.exportData<TensorDataStrided>().basePtr(), base.data(), base.size() * sizeof(float),
                      hipMemcpyHostToDevice);
            hipMemcpy(scale_tensor.exportData<TensorDataStrided>().basePtr(), scale.data(),
                      scale.size() * sizeof(float), hipMemcpyHostToDevice);
            break;
        }
        case eDeviceType::CPU: {
            hipMemcpy(base_tensor.exportData<TensorDataStrided>().basePtr(), base.data(), base.size() * sizeof(float),
                      hipMemcpyHostToHost);
            hipMemcpy(scale_tensor.exportData<TensorDataStrided>().basePtr(), scale.data(),
                      scale.size() * sizeof(float), hipMemcpyHostToHost);
            break;
        }
    }

    hipStream_t stream;
    HIP_VALIDATE_NO_ERRORS(hipStreamCreate(&stream));

    Normalize op;
    op(stream, input, base_tensor, scale_tensor, output, global_scale, shift, epsilon, flags, device);

    HIP_VALIDATE_NO_ERRORS(hipStreamSynchronize(stream));
    HIP_VALIDATE_NO_ERRORS(hipStreamDestroy(stream));

    return compareImage(output, expected_image, 0.0f);
}

eTestStatusType test_op_normalize(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <test data path>" << std::endl;
        return eTestStatusType::TEST_FAILURE;
    }
    std::filesystem::path test_data_path = std::filesystem::path(argv[1]) / "tests" / "ops";
    std::filesystem::path test_image_filepath = test_data_path / "test_input.bmp";
    std::filesystem::path expected_image_filepath = test_data_path / "expected_normalize.bmp";

    try {
        EXPECT_TEST_STATUS(TestCorrectness(test_image_filepath, expected_image_filepath,
                                           std::vector<float>({121.816, 117.935, 98.395}),
                                           std::vector<float>({82.195, 62.885, 61.023}), 85.0f, 180.0f, 0.0f,
                                           ROCCV_NORMALIZE_SCALE_IS_STDDEV, eDeviceType::GPU),
                           eTestStatusType::TEST_SUCCESS);
        EXPECT_TEST_STATUS(TestCorrectness(test_image_filepath, expected_image_filepath,
                                           std::vector<float>({121.816, 117.935, 98.395}),
                                           std::vector<float>({82.195, 62.885, 61.023}), 85.0f, 180.0f, 0.0f,
                                           ROCCV_NORMALIZE_SCALE_IS_STDDEV, eDeviceType::CPU),
                           eTestStatusType::TEST_SUCCESS);

    } catch (Exception e) {
        fprintf(stderr, "Exception: %s\n", e.what());
        return eTestStatusType::TEST_FAILURE;
    }

    return eTestStatusType::TEST_SUCCESS;
}
