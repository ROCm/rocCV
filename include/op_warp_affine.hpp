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

#include "core/tensor.hpp"
#include "i_operator.hpp"
#include "op_warp_perspective.hpp"
#include "operator_types.h"

namespace roccv {
/**
 * @brief Class for managing the Warp Affine Operator
 *
 */
// Typedef for affine transformation matrix. Row-major.
typedef float AffineTransform[6];

class WarpAffine final : public IOperator {
   public:
    /**
     * @brief Construct a new WarpAffine object
     *
     */
    WarpAffine() : m_op() {}

    /**
     * @brief Destroy the WarpAffine object
     *
     */
    ~WarpAffine() {}

    /**
     * @brief Executes the WarpAffine operation on the given HIP stream. WarpAffine applies an affine transformation to
     * a given input image.
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
     *       Supported DataTypes(s)     [U8, S8, U16, S16, U32, S32, F32, F64]
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
     *       Batch size    | Yes
     *
     * @param[in] stream The HIP stream to run this operator on.
     * @param[in] input Input tensor with image data.
     * @param[out] output Output tensor for storing modified image data.
     * @param[in] xform Affine transformation matrix in row-major order.
     * @param[in] isInverted Flag defining whether the xform (transformation matrix) is the inverted transformation or
     * not.
     * @param[in] interp Interpolation method used for warp affine.
     * @param[in] borderMode The border mode to use for the affine transformation.
     * @param[in] borderValue The border value to use in the case of a CONSTANT border mode.
     * @param[in] device The device to run this operator on. (Default: GPU)
     */
    void operator()(hipStream_t stream, const Tensor& input, const Tensor& output, const AffineTransform xform,
                    const bool isInverted, const eInterpolationType interp, const eBorderType borderMode,
                    const float4 borderValue, const eDeviceType device = eDeviceType::GPU) const;

   private:
    // WarpPerspective op is used to execute the affine transformation, as affine transformation is a subset.
    WarpPerspective m_op;
};
}  // namespace roccv