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
#include <op_resize.hpp>
#include <opencv2/opencv.hpp>

#include "test_helpers.hpp"

using namespace roccv;
using namespace roccv::tests;

eTestStatusType TestCorrectness(const std::string &input_file, const std::string &expected_file, float x_scale,
                                float y_scale, eInterpolationType interpolation, float error_threshold,
                                eDeviceType device) {
    Tensor input = createTensorFromImage(input_file, DataType(eDataType::DATA_TYPE_U8), device);
    auto input_data = input.exportData<TensorDataStrided>();

    int64_t output_width = input_data.shape(input.layout().width_index()) * x_scale;
    int64_t output_height = input_data.shape(input.layout().height_index()) * y_scale;
    TensorShape output_shape(input.layout(), {1, output_height, output_width, 3});

    Tensor output(output_shape, DataType(eDataType::DATA_TYPE_U8), device);

    Resize op;
    op(nullptr, input, output, interpolation, device);
    hipDeviceSynchronize();

    return compareImage(output, expected_file, error_threshold);
}

eTestStatusType test_op_resize(int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <test data path>" << std::endl;
        return eTestStatusType::TEST_FAILURE;
    }
    std::filesystem::path base_path = std::filesystem::path(argv[1]) / "tests" / "ops";

    try {
        // Test GPU correctness
        EXPECT_TEST_STATUS(TestCorrectness(base_path / "test_input.bmp", base_path / "expected_resize.bmp", 3.0f, 4.0f,
                                           eInterpolationType::INTERP_TYPE_NEAREST, 0.0f, eDeviceType::GPU),
                           eTestStatusType::TEST_SUCCESS);

        // Test CPU correctness
        EXPECT_TEST_STATUS(TestCorrectness(base_path / "test_input.bmp", base_path / "expected_resize.bmp", 3.0f, 4.0f,
                                           eInterpolationType::INTERP_TYPE_NEAREST, 0.0f, eDeviceType::CPU),
                           eTestStatusType::TEST_SUCCESS);

    } catch (Exception e) {
        std::cout << "Exception: " << e.what() << std::endl;
        return eTestStatusType::TEST_FAILURE;
    }

    return eTestStatusType::TEST_SUCCESS;
}
