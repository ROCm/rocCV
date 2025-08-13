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

/**
 * @brief Computes the area for a given box.
 *
 * @param[in] box The short4 representation of the box (x, y, width, height)
 * @return The computed area of the box.
 */
float GoldenArea(const short4 &box) { return box.z * box.w; }

/**
 * @brief Computes the IoU between two boxes.
 *
 * @param[in] a Box A.
 * @param[in] b Box B.
 * @return The IoU between boxes A and B.
 */
float GoldenIoU(const short4 &a, const short4 &b) {
    int aInterLeft = std::max(a.x, b.x);
    int bInterTop = std::max(a.y, b.y);
    int aInterRight = std::min(a.x + a.z, b.z + b.z);
    int bInterBottom = std::min(a.y + a.w, b.y + b.w);
    int widthInter = aInterRight - aInterLeft;
    int heightInter = bInterBottom - bInterTop;
    float interArea = widthInter * heightInter;
    float iou = 0.0f;

    if (widthInter > 0.0f && heightInter > 0.0f) {
        float unionArea = GoldenArea(a) + GoldenArea(b) - interArea;
        if (unionArea > 0.0f) {
            iou = interArea / unionArea;
        }
    }

    return iou;
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
    std::vector<unsigned char> output(batchSize * numBoxes);

    for (int batchIdx = 0; batchIdx < batchSize; batchIdx++) {
        for (int a = 0; a < numBoxes; a++) {
            // Starting index for this particular sample
            size_t beginIdx = batchIdx * numBoxes;

            const float &scoreA = scores.at(beginIdx + a);
            unsigned char &dst = output.at(beginIdx + a);

            // Discard this box immediately if its score is below the score threshold
            if (scoreA < scoreThreshold) {
                dst = 0u;
                continue;
            }

            const short4 &boxA = input.at(beginIdx + a);
            bool discard = false;

            // Compare this box to all other boxes to determine if it should be suppressed or not
            for (int b = 0; b < numBoxes; b++) {
                if (a == b) {
                    continue;
                }

                // For a box to be suppressed, the following conditions must be met
                // 1. Its intersection over union with another box must be greater than iouThreshold. If the IoU is over
                // the threshold, then check:
                //    - If its confidence score is less than the box being compared against, suppress it.
                //    - In the case of a tie breaker (scores are the same), suppress it if its area is less than the box
                //      being compared with.
                const short4 &boxB = input.at(beginIdx + b);
                if (GoldenIoU(boxA, boxB) > iouThreshold) {
                    const float &scoreB = scores.at(beginIdx + b);
                    if (scoreA < scoreB || (scoreA == scoreB && GoldenArea(boxA) < GoldenArea(boxB))) {
                        discard = true;
                        break;
                    }
                }
            }

            dst = discard ? 0u : 1u;
        }
    }

    return output;
}

/**
 * @brief Generate input boxes for NMS in a way that's more likely for them to intersect with each other.
 *
 * @param batchSize The number of samples in the batch.
 * @param numBoxes The number of boxes per sample.
 * @param seed A seed for random generation. Defaults to 12345.
 * @return A vector containing data for the generated boxes.
 */
std::vector<short4> GenerateBoxes(int batchSize, int numBoxes, int seed = 12345) {
    std::vector<short4> output;
    output.reserve(batchSize * numBoxes);

    std::mt19937 eng(seed);
    std::uniform_int_distribution<short> pos(0, 300);
    std::uniform_int_distribution<short> size(50, 100);

    for (int b = 0; b < batchSize; b++) {
        for (int i = 0; i < numBoxes; i++) {
            // Insert a randomly generated box (format is x, y, width, height)
            output.push_back(make_short4(pos(eng), pos(eng), size(eng), size(eng)));
        }
    }

    return output;
}

void TestCorrectness(int batchSize, int numBoxes, float scoreThreshold, float iouThreshold, eDeviceType device) {
    // Create required input/output tensors for roccv::NonMaximumSuppression
    Tensor inputTensor(TensorShape(TensorLayout(TENSOR_LAYOUT_NWC), {batchSize, numBoxes, 4}), DataType(DATA_TYPE_S16),
                       device);
    Tensor scoresTensor(TensorShape(TensorLayout(TENSOR_LAYOUT_NWC), {batchSize, numBoxes, 1}), DataType(DATA_TYPE_F32),
                        device);
    Tensor outputTensor(TensorShape(TensorLayout(TENSOR_LAYOUT_NWC), {batchSize, numBoxes, 1}), DataType(DATA_TYPE_U8),
                        device);

    // Generate data for input boxes and their associated scores
    std::vector<short4> boxes = GenerateBoxes(batchSize, numBoxes);
    std::vector<float> scores(batchSize * numBoxes);
    FillVector(scores);

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

    // GPU conformance tests
    TEST_CASE((TestCorrectness(1, 4, 0.0f, 0.5f, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness(3, 64, 0.25f, 0.5f, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness(20, 837, 0.5f, 0.45f, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness(1000, 34, 0.24f, 0.8f, eDeviceType::GPU)));

    // CPU conformance tests
    TEST_CASE((TestCorrectness(1, 4, 0.0f, 0.5f, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness(3, 64, 0.25f, 0.5f, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness(20, 837, 0.5f, 0.45f, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness(1000, 34, 0.24f, 0.8f, eDeviceType::CPU)));

    TEST_CASES_END();
}