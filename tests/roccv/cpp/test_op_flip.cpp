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
#include <op_flip.hpp>
#include <opencv2/opencv.hpp>

#include "test_helpers.hpp"

using namespace roccv;
using namespace roccv::tests;

eTestStatusType TestErrorHandling() {
    TensorShape validShape(TensorLayout(eTensorLayout::TENSOR_LAYOUT_NHWC), {1, 1, 1, 1});
    Tensor validGPUTensor(validShape, DataType(eDataType::DATA_TYPE_U8), eDeviceType::GPU);
    Tensor validCPUTensor(validShape, DataType(eDataType::DATA_TYPE_U8), eDeviceType::CPU);
    Flip op;

    {
        // Test output tensor on CPU for GPU operation
        EXPECT_EXCEPTION(op(nullptr, validGPUTensor, validCPUTensor, 0, eDeviceType::GPU),
                         eStatusType::INVALID_OPERATION);
    }

    {
        // Test input tensor on CPU for GPU operation
        EXPECT_EXCEPTION(op(nullptr, validCPUTensor, validGPUTensor, 0, eDeviceType::GPU),
                         eStatusType::INVALID_OPERATION);
    }

    {
        // Test unsupported layout
        TensorShape invalidLayoutShape(TensorLayout(eTensorLayout::TENSOR_LAYOUT_NC), {1, 1});
        Tensor invalidTensor(invalidLayoutShape, DataType(eDataType::DATA_TYPE_U8), eDeviceType::GPU);
        EXPECT_EXCEPTION(op(nullptr, invalidTensor, validGPUTensor, 0, eDeviceType::GPU),
                         eStatusType::INVALID_COMBINATION);
    }

    {
        // Test unsupported data type
        Tensor invalidTensor(validGPUTensor.shape(), DataType(eDataType::DATA_TYPE_U32), eDeviceType::GPU);
        EXPECT_EXCEPTION(op(nullptr, invalidTensor, validGPUTensor, 0, eDeviceType::GPU), eStatusType::NOT_IMPLEMENTED);
    }

    {
        // Test input/output shape mismatch
        Tensor invalidTensor(TensorShape(validGPUTensor.layout(), {2, 2, 2, 2}), DataType(eDataType::DATA_TYPE_U8),
                             eDeviceType::GPU);
        EXPECT_EXCEPTION(op(nullptr, invalidTensor, validGPUTensor, 0, eDeviceType::GPU),
                         eStatusType::INVALID_COMBINATION);
    }

    return eTestStatusType::TEST_SUCCESS;
}

eTestStatusType TestCorrectness(const std::string& inputFile, const std::string& expectedFile, int32_t flipCode,
                                float errorThreshold, eDeviceType device) {
    Tensor input = createTensorFromImage(inputFile, DataType(eDataType::DATA_TYPE_U8), device);
    Tensor output(input.shape(), input.dtype(), device);

    Flip op;
    op(nullptr, input, output, flipCode, device);
    hipDeviceSynchronize();

    return compareImage(output, expectedFile, errorThreshold);
}

eTestStatusType test_op_flip(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <test data path>" << std::endl;
        return eTestStatusType::TEST_FAILURE;
    }
    std::filesystem::path testDataPath = std::filesystem::path(argv[1]) / "tests" / "ops";

    // Test that errors are being handled properly
    TestErrorHandling();

    try {
        // Test GPU implementation correctness
        EXPECT_TEST_STATUS(TestCorrectness(testDataPath / "test_input.bmp", testDataPath / "expected_flip.bmp", 1, 0.0f,
                                           eDeviceType::GPU),
                           eTestStatusType::TEST_SUCCESS);

        // Test CPU implementation correctness
        EXPECT_TEST_STATUS(TestCorrectness(testDataPath / "test_input.bmp", testDataPath / "expected_flip.bmp", 1, 0.0f,
                                           eDeviceType::CPU),
                           eTestStatusType::TEST_SUCCESS);
    } catch (Exception e) {
        std::cout << "Exception: " << e.what() << std::endl;
        return eTestStatusType::TEST_FAILURE;
    }
    return eTestStatusType::TEST_SUCCESS;
}
