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
#include <op_rotate.hpp>
#include <opencv2/opencv.hpp>

#include "test_helpers.hpp"

using namespace roccv;
using namespace roccv::tests;

double2 computeCenterShift(const int centerX, const int centerY, const double angle) {
    double xShift = (1 - cos(angle * M_PI / 180)) * centerX - sin(angle * M_PI / 180) * centerY;
    double yShift = sin(angle * M_PI / 180) * centerX + (1 - cos(angle * M_PI / 180)) * centerY;
    return {xShift, yShift};
}

eTestStatusType TestCorrectness(const std::string& input_file, const std::string& expected_file, double angle,
                                const eInterpolationType interpolation, float error_threshold, eDeviceType device) {
    Tensor input = createTensorFromImage(input_file, DataType(eDataType::DATA_TYPE_U8), device);
    Tensor output(input.shape(), input.dtype(), input.device());

    // Compute center shift based on rotation degrees and image dimensions
    int centerX = (input.shape(input.layout().width_index()) + 1) / 2,
        centerY = (input.shape(input.layout().height_index()) + 1) / 2;
    double2 shift = computeCenterShift(centerX, centerY, angle);
    Rotate op;
    op(nullptr, input, output, angle, shift, interpolation, device);
    hipDeviceSynchronize();

    return compareImage(output, expected_file, error_threshold);
}

eTestStatusType test_op_rotate(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <test data path>" << std::endl;
        return eTestStatusType::TEST_FAILURE;
    }
    std::filesystem::path base_path = std::filesystem::path(argv[1]) / "tests" / "ops";

    try {
        EXPECT_TEST_STATUS(
            TestCorrectness(base_path / "test_input.bmp", base_path / "rotate" / "expected_rotate_nearest.bmp", 315,
                            eInterpolationType::INTERP_TYPE_NEAREST, 0.0f, eDeviceType::GPU),
            eTestStatusType::TEST_SUCCESS);
        EXPECT_TEST_STATUS(
            TestCorrectness(base_path / "test_input.bmp", base_path / "rotate" / "expected_rotate_nearest.bmp", 315,
                            eInterpolationType::INTERP_TYPE_NEAREST, 0.0f, eDeviceType::CPU),
            eTestStatusType::TEST_SUCCESS);

    } catch (Exception e) {
        std::cout << "Exception: " << e.what() << std::endl;
        return eTestStatusType::TEST_FAILURE;
    }
    return eTestStatusType::TEST_SUCCESS;
}