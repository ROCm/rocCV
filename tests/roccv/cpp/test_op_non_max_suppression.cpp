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
#include <fstream>
#include <iostream>
#include <iterator>
#include <op_non_max_suppression.hpp>
#include <vector>

#include "test_helpers.hpp"

using namespace roccv;
using namespace roccv::tests;

eTestStatusType testCorrectness(std::vector<short4> boxes_data, std::vector<float> scores_data,
                                std::vector<uint8_t> expected_data, float score_threshold, float iou_threshold,
                                eDeviceType device) {
    int numBoxes = scores_data.size();
    TensorShape shape(TensorLayout(TENSOR_LAYOUT_NW), {1, numBoxes});
    Tensor input(shape, DataType(DATA_TYPE_4S16), device);
    Tensor output(shape, DataType(DATA_TYPE_U8), device);
    Tensor scores(shape, DataType(DATA_TYPE_F32), device);

    copyData<short4>(input, boxes_data, device);
    copyData<float>(scores, scores_data, device);

    NonMaximumSuppression op;
    op(nullptr, input, output, scores, score_threshold, iou_threshold, device);
    hipDeviceSynchronize();

    return compareArray<uint8_t>(output, expected_data, 0.0f);
}

eTestStatusType test_op_non_max_suppression(int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <test data path>" << std::endl;
        return eTestStatusType::TEST_FAILURE;
    }
    std::filesystem::path test_data = std::filesystem::path(argv[1]) / "tests" / "ops" / "expected_nms.bin";
    std::ifstream file(test_data, std::ios::binary);
    if (!file) {
        std::cerr << "Error opening test file: " << test_data << std::endl;
        return eTestStatusType::TEST_FAILURE;
    }

    std::vector<uint8_t> test_data_vec(std::istreambuf_iterator<char>(file), {});
    file.close();

    // clang-format off
    std::vector<short4> boxesData = {
        make_short4(100, 100, 200, 200),
        make_short4(110, 110, 210, 210),
        make_short4(220, 220, 320, 320),
        make_short4(50, 50, 150, 150)
    };
    // clang-format on

    std::vector<float> scoresData = {0.9, 0.8, 0.85, 0.7};
    float iouThreshold = 0.5f;
    float scoreThreshold = 0.0f;

    try {
        EXPECT_TEST_STATUS(
            testCorrectness(boxesData, scoresData, test_data_vec, scoreThreshold, iouThreshold, eDeviceType::GPU),
            eTestStatusType::TEST_SUCCESS);
        EXPECT_TEST_STATUS(
            testCorrectness(boxesData, scoresData, test_data_vec, scoreThreshold, iouThreshold, eDeviceType::CPU),
            eTestStatusType::TEST_SUCCESS);
    } catch (Exception e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return eTestStatusType::TEST_FAILURE;
    }

    return eTestStatusType::TEST_SUCCESS;
}