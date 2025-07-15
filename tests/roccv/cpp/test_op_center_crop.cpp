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
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <op_center_crop.hpp>
#include <opencv2/opencv.hpp>

#include "test_helpers.hpp"

using namespace roccv;
using namespace roccv::tests;

void TestCorrectness(const std::string& inputFile, const std::string& expectedFile, int32_t cropWidth, int32_t cropHeight,
                     float errorThreshold, eDeviceType device) {
    Tensor input = createTensorFromImage(inputFile, DataType(eDataType::DATA_TYPE_U8), device);
    Tensor output(TensorShape(input.layout(), {1, cropHeight, cropWidth, 3}), input.dtype(), device);

    Size2D cropSize;
    cropSize.w = cropWidth;
    cropSize.h = cropHeight;

    CenterCrop op;
    op(nullptr, input, output, cropSize, device);
    hipDeviceSynchronize();

    EXPECT_TEST_STATUS(compareImage(output, expectedFile, errorThreshold), eTestStatusType::TEST_SUCCESS);
}

eTestStatusType test_op_center_crop(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <test data path>" << std::endl;
        return eTestStatusType::TEST_FAILURE;
    }
    std::filesystem::path testDataPath = std::filesystem::path(argv[1]) / "tests" / "ops";

    try {
        // Test GPU implementation correctness
        TestCorrectness(testDataPath / "test_input.bmp", testDataPath / "expected_center_crop.bmp",
                        300, 300, 0.0f, eDeviceType::GPU);

        // Test CPU implementation correctness
        TestCorrectness(testDataPath / "test_input.bmp", testDataPath / "expected_center_crop.bmp",
                        300, 300, 0.0f, eDeviceType::CPU);
    } catch (Exception e) {
        std::cout << "Exception: " << e.what() << std::endl;
        return eTestStatusType::TEST_FAILURE;
    }

    return eTestStatusType::TEST_SUCCESS;
}
