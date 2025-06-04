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
#include <op_warp_perspective.hpp>

#include "test_helpers.hpp"

using namespace roccv;
using namespace roccv::tests;

struct OpParams {
    PerspectiveTransform transform_matrix = {1, 0, 0, 0, 1, 0, -0.001, 0, 1};
    bool inverted = true;
    eInterpolationType interpolation_type = INTERP_TYPE_NEAREST;
    eBorderType border_type = BORDER_TYPE_CONSTANT;
    const float4 border_value = {255, 0, 255, 0};
};

void TestCorrectness(const std::string& inputFile, const std::string& expectedFile, const OpParams params,
                     const eDeviceType device, float errorThreshold) {
    Tensor input = createTensorFromImage(inputFile, DataType(eDataType::DATA_TYPE_U8), device);
    Tensor output(input.shape(), input.dtype(), device);

    roccv::WarpPerspective op;
    op(nullptr, input, output, params.transform_matrix, params.inverted, params.interpolation_type, params.border_type,
       params.border_value, device);
    hipDeviceSynchronize();

    EXPECT_TEST_STATUS(compareImage(output, expectedFile, errorThreshold), eTestStatusType::TEST_SUCCESS);
}

eTestStatusType test_op_warp_perspective(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <test data path>" << std::endl;
        return eTestStatusType::TEST_FAILURE;
    }
    std::filesystem::path testDataPath = std::filesystem::path(argv[1]) / "tests" / "ops";

    try {
        // Test GPU implementation correctness
        TestCorrectness(testDataPath / "test_input.bmp", testDataPath / "expected_warp_perspective.bmp", OpParams(),
                        eDeviceType::GPU, 1.0f);

        // Test CPU implementation correctness
        TestCorrectness(testDataPath / "test_input.bmp", testDataPath / "expected_warp_perspective.bmp", OpParams(),
                        eDeviceType::CPU, 1.0f);
    } catch (Exception e) {
        std::cout << "Exception: " << e.what() << std::endl;
        return eTestStatusType::TEST_FAILURE;
    }

    return eTestStatusType::TEST_SUCCESS;
}
