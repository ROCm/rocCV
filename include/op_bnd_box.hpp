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
#include <hip/hip_runtime.h>
#include <operator_types.h>

#include <i_operator.hpp>
#include <vector>

#include "core/tensor.hpp"

namespace roccv {
/**
 * @brief Class for managing the BndBox Operator
 *
 *
 */
class BndBox final : public IOperator {
   public:
    /**
     * @brief Constructs a BndBox object.
     *
     */

    BndBox();

    /**
     * @brief Destroy the BndBox object
     *
     */
    ~BndBox();

    /**
     * @brief Executes the BndBox operation. BndBox can be used
     * to draw bounding boxes on images in a tensor.
     *
     * Limitations:
     *
     * Input:
     *       Supported TensorLayout(s): [NHWC, HWC]
     *                        Channels: [3, 4]
     *       Supported DataType(s):     [U8]
     *
     * Output:
     *       Supported TensorLayout(s): [NHWC, HWC]
     *                        Channels: [3, 4]
     *       Supported DataType(s):     [U8]
     *
     * Input/Output dependency:
     *
     *       Property      |  Input == Output
     *      -------------- | -------------
     *       TensorLayout  | Yes
     *       DataType      | Yes
     *       Channels      | Yes
     *       Width         | Yes
     *       Height        | Yes
     *       Batch         | Yes
     *
     * @param[in] stream The HIP stream to run this operation on.
     * @param[in] input Input tensor.
     * @param[out] output Output tensor.
     * @param[in] bnd_boxes Bounding boxes to apply to input tensor.
     * @param[in] device The device which this operation should run on.
     * (Default: eDeviceType::GPU)
     *
     */
    void operator()(hipStream_t stream, const roccv::Tensor& input, const roccv::Tensor& output,
                    const BndBoxes_t bnd_boxes, eDeviceType device = eDeviceType::GPU);

   private:
    void generateRects(std::vector<Rect_t>& rects, const BndBoxes_t& bnd_boxes, int64_t height, int64_t width);
};
}  // namespace roccv
