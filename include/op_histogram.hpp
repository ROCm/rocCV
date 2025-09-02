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

#include "core/tensor.hpp"


namespace roccv {
/**
 * @brief Class for managing the Histogram operator.
 *
 */
class Histogram final : public IOperator {
   public:
    /**
     * @brief Constructs a Histogram object.
     *
     */

    Histogram();

    /**
     * @brief Destroy the Histogram object
     *
     */
    ~Histogram();
    /**
     * @brief Executes the Histogram operation.
     * Shows how many times each pixel value occurs for a grayscale image.
     *
     * Limitations:
     *
     * Input:
     *       Supported TensorLayout(s): [NHWC, HWC]
     *                        Channels: [1]
     *       Supported DataType(s):     [U8]
     *
     * Output:
     *       Supported TensorLayout(s): [HWC]
     *                        Channels: [1]
     *       Supported DataTypes(s)     [U32, S32]
     *
     * Mask:
     *       Supported TensorLayout(s): [NHWC, HWC]
     *                        Channels: [1]
     *       Supported DataTypes(s)     [U8]
     *
     * Input/Output dependency:
     *
     *       Property      |  Input == Output
     *      -------------- | -------------
     *       TensorLayout  | No
     *       DataType      | No
     *       Channels      | No
     *       Width         | No
     *       Height        | No
     *       Batch         | No
     *
     * @param[in] stream The HIP stream to run this operation on.
     * @param[in] input Input tensor with image data.
     * @param[in] mask (Optional) Mask tensor with shape equal to the input
     * tensor shape and any value not equal 0 will be counted in histogram.
     * @param[out] histogram Histogram output tensor with width of 256 and a height equal to
     * the batch size of input (1 if HWC input).
     * @param[in] device The device which this operation should run on.
     * (Default: eDeviceType::GPU)
     *
     */
    void operator()(hipStream_t stream, const Tensor& input, std::optional<std::reference_wrapper<const Tensor>> mask, const Tensor& histogram,
                    const eDeviceType device = eDeviceType::GPU);
};
}  // namespace roccv