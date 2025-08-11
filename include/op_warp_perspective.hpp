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
 * @brief Class for managing the Warp Perspective operator.
 *
 */
class WarpPerspective final : public IOperator {
   public:
    /**
     * @brief Construct a new Op Warp Perspective object. The object can be used
     * to apply a perspective transformation operation on an image.
     *
     * Limitations:
     *
     * Input:
     *       Supported TensorLayout(s): [HWC, NHWC]
     *                        Channels: [1, 3, 4]
     *       Supported DataType(s):     [U8, S8, U16, S16, U32, S32, F32, F64]
     *
     * Output:
     *       Supported TensorLayout(s): [HWC, NHWC]
     *                        Channels: [1, 3, 4]
     *       Supported DataType(s):     [U8, S8, U16, S16, U32, S32, F32, F64]
     *
     * Input/Output dependency:
     *
     *       Property      |  Input == Output
     *      -------------- | -------------
     *       TensorLayout  | Yes
     *       DataType      | Yes
     *       Channels      | Yes
     *       Width         | No
     *       Height        | No
     *
     * @param[in] stream The HIP stream to run this operator on.
     * @param[in] input Input tensor with image data.
     * @param[out] output  Output tensor for storing modified image data.
     * @param[in] transMatrix A 3x3 transformation matrix for the
     * perspective transformation.
     * @param[in] isInverted Marks whether transMatrix is the inverted transformation matrix or not.
     * @param[in] interpolation Interpolation type (e.g. Nearest neigbour,
     * linear).
     * @param[in] borderType A border type to identify the pixel extrapolation
     * method (e.g. BORDER_TYPE_CONSTANT or BORDER_TYPE_REPLICATE)
     * @param[in] borderValue Set as 0 unless using a constant border.
     * @param[in] device The device to run this operator on. (Default: GPU)
     */
    void operator()(hipStream_t stream, const roccv::Tensor &input, const roccv::Tensor &output,
                    const PerspectiveTransform transMatrix, bool isInverted, const eInterpolationType interpolation,
                    const eBorderType borderType, const float4 borderValue,
                    const eDeviceType device = eDeviceType::GPU) const;
};
}  // namespace roccv
