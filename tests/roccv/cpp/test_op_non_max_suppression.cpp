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
    short4 boxCoords;  // Coordinates of the box (x, y, w, h)
    int idx;           // The global idx of this box
    float score;       // The corresponding confidence score of this box
};

/**
 * @brief Computes the IoU between two boxes.
 *
 * @param[in] a Box A.
 * @param[in] b Box B.
 * @return The IoU between boxes A and B.
 */
float ComputeIoU(const short4 &a, const short4 &b) {
    float ax1 = a.x;
    float ay1 = a.y;
    float ax2 = a.x + a.z;
    float ay2 = a.y + a.w;

    float bx1 = b.x;
    float by1 = b.y;
    float bx2 = b.x + b.z;
    float by2 = b.y + b.w;

    float interX1 = std::max(ax1, bx1);
    float interY1 = std::max(ay1, by1);
    float interX2 = std::min(ax2, bx2);
    float interY2 = std::min(ay2, by2);

    float interW = std::max(0.0f, interX2 - interX1);
    float interH = std::max(0.0f, interY2 - interY1);
    float interA = interW * interH;

    float areaA = (ax2 - ax1) * (ay2 - ay1);
    float areaB = (bx2 - bx1) * (by2 - by1);
    float unionA = areaA + areaB - interA;
    if (unionA <= 0.0f) return 0.0f;
    return interA / unionA;
}

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
    // Initially, all boxes are suppressed.
    std::vector<unsigned char> output(batchSize * numBoxes, 0);

    for (int b = 0; b < batchSize; b++) {
        int beginIdx = b * numBoxes;
        int endIdx = b * numBoxes + numBoxes;

        // Copy boxes/scores local to this batch index into a struct which keeps track of their score/original index.
        std::vector<BoxInfo> localBoxes;
        for (int i = beginIdx; i < endIdx; i++) {
            if (scores[i] >= scoreThreshold) {
                // Only include this box if its score threshold is large enough
                BoxInfo box{input[i], i, scores[i]};
                localBoxes.push_back(box);
            }
        }

        // Sort boxes in descending order
        std::sort(localBoxes.begin(), localBoxes.end(), [](BoxInfo &a, BoxInfo &b) { return a.score > b.score; });
        std::vector<BoxInfo> keptBoxes;

        while (!localBoxes.empty()) {
            BoxInfo current = localBoxes.front();
            output[current.idx] = 1;

            std::vector<BoxInfo> remaining;
            remaining.reserve(localBoxes.size());
            for (size_t i = 1; i < localBoxes.size(); i++) {
                // Keep the box if it is equal to or less than the iouThreshold
                if (ComputeIoU(current.boxCoords, localBoxes[i].boxCoords) <= iouThreshold) {
                    remaining.push_back(localBoxes[i]);
                }

                // Otherwise, we do nothing. This box will be suppressed by dropping it from the remaining list.
            }

            localBoxes.swap(remaining);
        }
    }

    return output;
}

void TestCorrectness(int batchSize, int numBoxes, std::vector<short4> boxes, std::vector<float> scores,
                     float scoreThreshold, float iouThreshold, eDeviceType device) {
    // Create required input/output tensors for roccv::NonMaximumSuppression
    Tensor inputTensor(TensorShape(TensorLayout(TENSOR_LAYOUT_NWC), {batchSize, numBoxes, 4}), DataType(DATA_TYPE_S16),
                       device);
    Tensor scoresTensor(TensorShape(TensorLayout(TENSOR_LAYOUT_NWC), {batchSize, numBoxes, 1}), DataType(DATA_TYPE_F32),
                        device);
    Tensor outputTensor(TensorShape(TensorLayout(TENSOR_LAYOUT_NWC), {batchSize, numBoxes, 1}), DataType(DATA_TYPE_U8),
                        device);

    // Copy host-allocated vectors into tensors
    CopyVectorIntoTensor(inputTensor, boxes);
    CopyVectorIntoTensor(scoresTensor, scores);

    hipStream_t stream;
    HIP_VALIDATE_NO_ERRORS(hipStreamCreate(&stream));
    NonMaximumSuppression op;
    op(stream, inputTensor, outputTensor, scoresTensor, scoreThreshold, iouThreshold, device);
    HIP_VALIDATE_NO_ERRORS(hipStreamSynchronize(stream));
    HIP_VALIDATE_NO_ERRORS(hipStreamDestroy(stream));

    // Copy output tensor data into host allocated tensor
    std::vector<unsigned char> actualOutput(outputTensor.shape().size());
    CopyTensorIntoVector(actualOutput, outputTensor);

    // Generate golden output using defined model above
    std::vector<unsigned char> goldenOutput =
        GoldenNonMaximumSuppression(boxes, scores, scoreThreshold, iouThreshold, batchSize, numBoxes);

    CompareVectors(actualOutput, goldenOutput);
}

}  // namespace

eTestStatusType test_op_non_max_suppression(int argc, char **argv) {
    TEST_CASES_BEGIN();

    // GPU Tests
    TEST_CASE((TestCorrectness(1, 4,
                               {make_short4(100, 100, 200, 200), make_short4(110, 110, 210, 210),
                                make_short4(220, 220, 320, 320), make_short4(50, 50, 150, 150)},
                               {0.9, 0.8, 0.85, 0.7}, 0.0f, 0.5f, eDeviceType::GPU)));

    // CPU Tests
    TEST_CASE((TestCorrectness(1, 4,
                               {make_short4(100, 100, 200, 200), make_short4(110, 110, 210, 210),
                                make_short4(220, 220, 320, 320), make_short4(50, 50, 150, 150)},
                               {0.9, 0.8, 0.85, 0.7}, 0.0f, 0.5f, eDeviceType::CPU)));

    TEST_CASES_END();
}