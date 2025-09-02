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

#include <i_operator.hpp>
#include <thread>
#include <vector>

#include "core/tensor.hpp"
#include "operator_types.h"

namespace roccv {
/**
 * @brief Class for managing the Bilateral filter operator
 *
 */
class BilateralFilter final : public IOperator {
   public:
    /**
     * @brief Constructs a BilateralFilter object.
     *
     */
    BilateralFilter();

    /**
     * @brief Destroy the BilateralFilter object
     *
     */
    ~BilateralFilter();

    /**
     * @brief Construct a new BilateralFilter object.
     * The object can be used to apply a bilateral filter on images in a tensor.
     *
     * Limitations:
     *
     * Input:
     *       Supported TensorLayout(s): [NHWC, HWC]
     *                        Channels: [1, 3, 4]
     *       Supported DataType(s):     [U8, S8, U16, S16, U32, S32, F32, F64]
     *
     * Output:
     *       Supported TensorLayout(s): [NHWC, HWC]
     *                        Channels: [1, 3, 4]
     *       Supported DataType(s):     [U8, S8, U16, S16, U32, S32, F32, F64]
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
     * @param[in] diameter bilateral filter diameter.
     * @param[in] sigmaColor Gaussian exponent for color difference, expected
     * to be positive, if it isn't, will be set to 1.0
     * @param[in] sigmaSpace Gaussian exponent for position difference expected
     * to be positive, if it isn't, will be set to 1.0
     * @param[in] borderMode A border type to identify the pixel extrapolation
     * method (e.g. BORDER_TYPE_CONSTANT or BORDER_TYPE_REPLICATE)
     * @param[in] borderValue Set as 0 unless using a constant border.
     * @param[in] device The device which this operation should run on.
     * (Default: eDeviceType::GPU)
     *
     */
    void operator()(hipStream_t stream, const roccv::Tensor& input, const roccv::Tensor& output, int diameter,
                    float sigmaColor, float sigmaSpace, const eBorderType borderMode,
                    const float4 borderValue = make_float4(0, 0, 0, 0), const eDeviceType device = eDeviceType::GPU);
};
}  // namespace roccv