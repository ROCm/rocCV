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

#pragma once

#include <operator_types.h>

#include "core/tensor.hpp"
#include "i_operator.hpp"

namespace roccv {
/**
 * @brief Class for managing the NMS operator.
 *
 */
class NonMaximumSuppression final : public IOperator {
   public:
    /**
     * @brief Construct a new Op Non Max Suppression object
     *
     */
    NonMaximumSuppression() {}

    /**
     * @brief Destroy the Op Non Max Suppression object
     *
     */
    ~NonMaximumSuppression() {}

    /**
     * @brief Executes the Non-Maximum Suppression operation. This object
     * performs Non-Maximum Suppression on bounding boxes based on scores a score threshold, and an
     * IoU threshold.
     *
     * Limitations:
     *
     * Input:
     *       Supported TensorLayout(s): [NW, NWC]
     *       Supported DataType(s):     [4S16, S16]
     *
     * Scores:
     *       Supported TensorLayout(s): [NW, NWC]
     *       Supported DataType(s):     [F32]
     *
     * Output:
     *       Supported TensorLayout(s): [NW, NWC]
     *       Supported DataType(s):     [U8]
     *
     * IoU Threshold:
     *       Range: (0.0, 1.0)
     *
     * Input/Output Dependency:
     *
     *       Property      |  Input == Output == Scores
     *      -------------- | -------------
     *       TensorLayout  | No
     *       DataType      | No
     *       Channels      | No
     *       Width         | Yes
     *       Height        | No
     *       Batch         | Yes
     *
     * @param[in] stream The HIP stream to run this operator on.
     * @param[in] input Batches of input boxes in the shape NW where N is the number of batches and W is the number of
     * boxes per batch. If using layout NW, datatype must be 4S16. Boxes are structured in memory as a short4 with (x=x,
     * y=y, z=width, w=height). If using layout NWC, datatype must be S16 with the final shape dimension being 4.
     * @param[out] output Output tensor is the output boolean mask which marks boxes as either kept (1) or suppressed
     * (0). If using NWC layout, the final shape dimension must be 1. The number of batches N and boxes per batch W must
     * match with the given input tensor.
     * @param[in] scores A Tensor containing the confidence scores for each box. If using layout NWC, the final shape
     * dimension must be 1. The number of batches N and boxes per batch W must match with the given input tensor.
     * @param[in] scoreThreshold The minimum score a box must have in order to be kept.
     * @param[in] iouThreshold IoU threshold to filter overlapping boxes.
     * @param[in] device The device to run this operator on. (Default: GPU)
     */
    void operator()(hipStream_t stream, const Tensor& input, const Tensor& output, const Tensor& scores,
                    float scoreThreshold, float iouThreshold, const eDeviceType device = eDeviceType::GPU) const;
};
}  // namespace roccv
