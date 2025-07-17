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

#include <i_operator.hpp>

#include "core/tensor.hpp"

namespace roccv {
/**
 * @brief Class for managing the Flip operator.
 *
 */
class Flip final : public IOperator {
   public:
    /**
     * @brief Construct a new Op Flip object
     *
     */
    Flip() {}

    /**
     * @brief Destroy the Op Flip object
     *
     */
    ~Flip() {}

    /**
     * @brief construct a new Op Flip object. the object can be used to flip an
     * image batch about the horizontal, vertical or both axes
     *
     * Limitations:
     *
     * Input:
     *       Supported TensorLayout(s): [TENSOR_LAYOUT_NHWC, TENSOR_LAYOUT_HWC]
     *                        Channels: [1, 3, 4]
     *       Supported DataType(s):     [U8, S32, F32]
     *
     * Output:
     *       Supported TensorLayout(s): [TENSOR_LAYOUT_NHWC, TENSOR_LAYOUT_HWC]
     *                        Channels: [1, 3, 4]
     *       Supported TensorLayout(s)  [U8, S32, F32]
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
     * @param[in] stream HIP stream to run this operator on.
     * @param[in] input Input tensor with image batch data
     * @param[out] output Output tensor for storing modified image batch data
     * @param[in] flipCode Flip code used to determine how the images in the
     * batch will be flipped. 0 will flip along the x-axis, a positive number
     * (e.g. 1) will flip around the y-axis, and a negative number (e.g. -1)
     * will flip around both axis.
     * @param[in] device The device to run this operation on. (Default: GPU)
     */
    void operator()(hipStream_t stream, const Tensor &input, const Tensor &output, int32_t flipCode,
                    const eDeviceType device = eDeviceType::GPU) const;
};
}  // namespace roccv
