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

#include <op_non_max_suppression.hpp>
#include <vector>

#include "test_helpers.hpp"

using namespace roccv;
using namespace roccv::tests;

namespace {

// Stores information about a box/confidence score pair
struct BoxInfo {
    short4 boxCoords;
    int idx;
    float score;
    unsigned char suppressed = 0;
};

/**
 * @brief Golden model for the non-maximum suppression operator. This calculates which boxes should be kept based on
 * input boxes, scores for each box, and a global score/iou threshold. Non-maximum suppression for each batch index is
 * calculated independently.
 *
 * @param[in] input A list of input boxes to check. Each short4 corresponds to the coordinates of a single box (x = x, y
 * = y, z = width, w = height).
 * @param[in] scores A list of scores corresponding to each box provided in input.
 * @param[in] scoreThreshold The score threshold to use when determining which boxes to keep and discard.
 * @param[in] iouThreshold The IoU threshold to use when comparing box intersections.
 * @param[in] batchSize The size of the batch.
 * @param[in] numBoxes The number of boxes per batch index.
 * @return A list of U8 values corresponding to each input box. A value of 0 means the box was suppressed, a value of 1
 * means the box is kept.
 */
std::vector<unsigned char> GoldenNonMaximumSuppression(std::vector<short4> input, std::vector<float> scores,
                                                       float scoreThreshold, float iouThreshold, int batchSize,
                                                       int numBoxes) {
    for (int b = 0; b < batchSize; b++) {
        int beginIdx = b * numBoxes;
        int endIdx = b * numBoxes + numBoxes;

        // Copy boxes/scores local to this batch index into a struct which keeps track of their score/original index.
        std::vector<BoxInfo> localBoxes;
        for (int i = beginIdx; i < endIdx; i++) {
            BoxInfo box;
            box.boxCoords = input[i];
            box.score = scores[i];
            box.idx = i - beginIdx;
            localBoxes.push_back(box);
        }

        // Sort boxes in descending order
        std::sort(localBoxes.begin(), localBoxes.end(), [](BoxInfo &a, BoxInfo &b) { return a.score > b.score; });

        // Now, we can begin determining which boxes can be suppressed or not
    }
}

}  // namespace

eTestStatusType test_op_non_max_suppression(int argc, char **argv) {
    TEST_CASES_BEGIN();

    TEST_CASES_END();
}

// eTestStatusType testCorrectness(std::vector<short4> boxes_data, std::vector<float> scores_data,
//                                 std::vector<uint8_t> expected_data, float score_threshold, float iou_threshold,
//                                 eDeviceType device) {
//     int numBoxes = scores_data.size();
//     TensorShape shape(TensorLayout(TENSOR_LAYOUT_NW), {1, numBoxes});
//     Tensor input(shape, DataType(DATA_TYPE_4S16), device);
//     Tensor output(shape, DataType(DATA_TYPE_U8), device);
//     Tensor scores(shape, DataType(DATA_TYPE_F32), device);

//     copyData<short4>(input, boxes_data, device);
//     copyData<float>(scores, scores_data, device);

//     NonMaximumSuppression op;
//     op(nullptr, input, output, scores, score_threshold, iou_threshold, device);
//     hipDeviceSynchronize();

//     return compareArray<uint8_t>(output, expected_data, 0.0f);
// }

// eTestStatusType test_op_non_max_suppression(int argc, char **argv) {
//     if (argc < 2) {
//         std::cerr << "Usage: " << argv[0] << " <test data path>" << std::endl;
//         return eTestStatusType::TEST_FAILURE;
//     }
//     std::filesystem::path test_data = std::filesystem::path(argv[1]) / "tests" / "ops" / "expected_nms.bin";
//     std::ifstream file(test_data, std::ios::binary);
//     if (!file) {
//         std::cerr << "Error opening test file: " << test_data << std::endl;
//         return eTestStatusType::TEST_FAILURE;
//     }

//     std::vector<uint8_t> test_data_vec(std::istreambuf_iterator<char>(file), {});
//     file.close();

//     // clang-format off
//     std::vector<short4> boxesData = {
//         make_short4(100, 100, 200, 200),
//         make_short4(110, 110, 210, 210),
//         make_short4(220, 220, 320, 320),
//         make_short4(50, 50, 150, 150)
//     };
//     // clang-format on

//     std::vector<float> scoresData = {0.9, 0.8, 0.85, 0.7};
//     float iouThreshold = 0.5f;
//     float scoreThreshold = 0.0f;

//     try {
//         EXPECT_TEST_STATUS(
//             testCorrectness(boxesData, scoresData, test_data_vec, scoreThreshold, iouThreshold, eDeviceType::GPU),
//             eTestStatusType::TEST_SUCCESS);
//         EXPECT_TEST_STATUS(
//             testCorrectness(boxesData, scoresData, test_data_vec, scoreThreshold, iouThreshold, eDeviceType::CPU),
//             eTestStatusType::TEST_SUCCESS);
//     } catch (Exception e) {
//         std::cerr << "Test failed with exception: " << e.what() << std::endl;
//         return eTestStatusType::TEST_FAILURE;
//     }

//     return eTestStatusType::TEST_SUCCESS;
// }