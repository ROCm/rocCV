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
#include <op_warp_affine.hpp>
#include <opencv2/opencv.hpp>

#include "test_helpers.hpp"

using namespace roccv;
using namespace roccv::tests;

eTestStatusType TestCorrectness(const std::string &inputFile, const std::string &expectedFile,
                                const AffineTransform xform, const bool isInverted,
                                const eInterpolationType interpolation, const eBorderType borderMode,
                                const float4 borderValue, float errorThreshold, const eDeviceType device) {
    Tensor input = createTensorFromImage(inputFile, DataType(eDataType::DATA_TYPE_U8), device);
    Tensor output(input.shape(), input.dtype(), device);

    WarpAffine op;
    op(nullptr, input, output, xform, isInverted, interpolation, borderMode, borderValue, device);
    hipDeviceSynchronize();

    return compareImage(output, expectedFile, errorThreshold);
}

eTestStatusType test_op_warp_affine(int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <test data path>" << std::endl;
        return eTestStatusType::TEST_FAILURE;
    }
    std::filesystem::path testDataPath = std::filesystem::path(argv[1]) / "tests" / "ops";

    try {
        AffineTransform affineMatrix = {1, 0, 0, 1, -1, 120};
        eInterpolationType interpolation = eInterpolationType::INTERP_TYPE_LINEAR;

        // Test GPU implementation correctness
        EXPECT_TEST_STATUS(TestCorrectness(testDataPath / "test_input.bmp", testDataPath / "expected_warp_affine.bmp",
                                           affineMatrix, false, interpolation, eBorderType::BORDER_TYPE_CONSTANT,
                                           make_float4(0, 0, 0, 0), 0.0f, eDeviceType::GPU),
                           eTestStatusType::TEST_SUCCESS);

        // Test CPU implementation correctness
        EXPECT_TEST_STATUS(TestCorrectness(testDataPath / "test_input.bmp", testDataPath / "expected_warp_affine.bmp",
                                           affineMatrix, false, interpolation, eBorderType::BORDER_TYPE_CONSTANT,
                                           make_float4(0, 0, 0, 0), 0.0f, eDeviceType::CPU),
                           eTestStatusType::TEST_SUCCESS);
    } catch (Exception e) {
        std::cout << "Exception: " << e.what() << std::endl;
        return eTestStatusType::TEST_FAILURE;
    }

    return eTestStatusType::TEST_SUCCESS;
}
