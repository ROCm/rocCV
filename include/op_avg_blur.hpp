/**
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

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

#include <i_operator.hpp>

#include "core/tensor.hpp"
#include "operator_types.h"

namespace roccv {
/**
 * @brief Class for managing the Average Blur operator
 *
 */
class AvgBlur final : public IOperator {
   public:
    /**
     * @brief Constructs an AvgBlur object.
     *
     */
    AvgBlur();

    /**
     * @brief Destroy the AvgBlur object
     *
     */
    ~AvgBlur();

    /**
     * @brief Construct a new AvgBlur object.
     * The object can be used to apply an average (mean) blur filter on images in a tensor.
     *
     * Limitations:
     *
     * Input:
     *       Supported TensorLayout(s): [NHWC, HWC]
     *                        Channels: [1, 3, 4]
     *       Supported DataType(s):     [U8, U16, S16, S32, F32]
     *
     * Output:
     *       Supported TensorLayout(s): [NHWC, HWC]
     *                        Channels: [1, 3, 4]
     *       Supported DataType(s):     [U8, U16, S16, S32, F32]
     *
     * Input/Output dependency
     *
     *       Property      |  Input == Output
     *      -------------- | -------------
     *       Data Layout   | Yes
     *       Data Type     | Yes
     *       Number        | Yes
     *       Channels      | Yes
     *       Width         | Yes
     *       Height        | Yes
     *
     *
     * @param[in] stream The HIP stream to run this operation on.
     * @param[in] input Input tensor with image batch data
     * @param[out] output Output tensor for storing modified image batch data
     * @param[in] kernelWidth Width of the averaging kernel.
     * @param[in] kernelHeight Height of the averaging kernel.
     * @param[in] kernelAnchorX Kernel anchor in X direction.
     * @param[in] kernelAnchorY Kernel anchor in Y direction.
     * @param[in] borderMode A border type to identify the pixel extrapolation
     * method (e.g. BORDER_TYPE_CONSTANT or BORDER_TYPE_REPLICATE)
     * @param[in] borderValue Set as 0 unless using a constant border.
     * @param[in] device The device which this operation should run on.
     * (Default: eDeviceType::GPU)
     *
     */
    void operator()(hipStream_t stream, const Tensor& input, const Tensor& output,
                    int kernelWidth, int kernelHeight, int kernelAnchorX, int kernelAnchorY,
                    eBorderType borderType, float4 borderValue, eDeviceType device = eDeviceType::GPU);
};
}  // namespace roccv