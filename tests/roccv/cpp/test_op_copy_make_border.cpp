/*
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include <filesystem>
#include <op_copy_make_border.hpp>

#include "test_helpers.hpp"

using namespace roccv;
using namespace roccv::tests;

eTestStatusType TestCorrectness(const std::string& input_image, const std::string& golden_image, int32_t top,
                                int32_t left, eBorderType border_mode, float4 border_value, eDeviceType device,
                                float error_threshold) {
    Tensor input = createTensorFromImage(input_image, DataType(DATA_TYPE_U8), device);
    TensorShape o_shape(input.layout(), {1, input.shape(input.layout().height_index()) + top * 2,
                                         input.shape(input.layout().width_index()) + left * 2, 3});
    Tensor output(o_shape, input.dtype(), device);

    CopyMakeBorder op;
    op(nullptr, input, output, top, left, border_mode, border_value, device);
    hipDeviceSynchronize();

    return compareImage(output, golden_image, error_threshold);
}

eTestStatusType test_op_copy_make_border(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <test data path>" << std::endl;
        return eTestStatusType::TEST_FAILURE;
    }
    std::filesystem::path base_path = std::filesystem::path(argv[1]) / "tests" / "ops";

    try {
        // GPU implementation tests
        EXPECT_TEST_STATUS(TestCorrectness(base_path / "test_input.bmp",
                                           base_path / "copy_make_border" / "expected_copy_make_border_constant.bmp", 9,
                                           9, BORDER_TYPE_CONSTANT, make_float4(0, 0, 1.0, 0), eDeviceType::GPU, 0.0f),
                           eTestStatusType::TEST_SUCCESS);
        EXPECT_TEST_STATUS(
            TestCorrectness(base_path / "test_input.bmp",
                            base_path / "copy_make_border" / "expected_copy_make_border_replicate.bmp", 9, 9,
                            BORDER_TYPE_REPLICATE, make_float4(0, 0, 1.0, 0), eDeviceType::GPU, 0.0f),
            eTestStatusType::TEST_SUCCESS);
        EXPECT_TEST_STATUS(TestCorrectness(base_path / "test_input.bmp",
                                           base_path / "copy_make_border" / "expected_copy_make_border_reflect.bmp", 9,
                                           9, BORDER_TYPE_REFLECT, make_float4(0, 0, 1.0, 0), eDeviceType::GPU, 0.0f),
                           eTestStatusType::TEST_SUCCESS);
        EXPECT_TEST_STATUS(TestCorrectness(base_path / "test_input.bmp",
                                           base_path / "copy_make_border" / "expected_copy_make_border_wrap.bmp", 9, 9,
                                           BORDER_TYPE_WRAP, make_float4(0, 0, 1.0, 0), eDeviceType::GPU, 0.0f),
                           eTestStatusType::TEST_SUCCESS);

        // CPU implementation tests
        EXPECT_TEST_STATUS(TestCorrectness(base_path / "test_input.bmp",
                                           base_path / "copy_make_border" / "expected_copy_make_border_constant.bmp", 9,
                                           9, BORDER_TYPE_CONSTANT, make_float4(0, 0, 1.0, 0), eDeviceType::CPU, 0.0f),
                           eTestStatusType::TEST_SUCCESS);
        EXPECT_TEST_STATUS(
            TestCorrectness(base_path / "test_input.bmp",
                            base_path / "copy_make_border" / "expected_copy_make_border_replicate.bmp", 9, 9,
                            BORDER_TYPE_REPLICATE, make_float4(0, 0, 1.0, 0), eDeviceType::CPU, 0.0f),
            eTestStatusType::TEST_SUCCESS);
        EXPECT_TEST_STATUS(TestCorrectness(base_path / "test_input.bmp",
                                           base_path / "copy_make_border" / "expected_copy_make_border_reflect.bmp", 9,
                                           9, BORDER_TYPE_REFLECT, make_float4(0, 0, 1.0, 0), eDeviceType::CPU, 0.0f),
                           eTestStatusType::TEST_SUCCESS);
        EXPECT_TEST_STATUS(TestCorrectness(base_path / "test_input.bmp",
                                           base_path / "copy_make_border" / "expected_copy_make_border_wrap.bmp", 9, 9,
                                           BORDER_TYPE_WRAP, make_float4(0, 0, 1.0, 0), eDeviceType::CPU, 0.0f),
                           eTestStatusType::TEST_SUCCESS);
    } catch (Exception e) {
        printf("Exception occured: %s\n", e.what());
        return eTestStatusType::TEST_FAILURE;
    }
    return eTestStatusType::TEST_SUCCESS;
}