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

#include "core/tensor.hpp"
#include "op_warp_affine.hpp"
#include "operator_types.h"

namespace roccv {

/**
 * @brief Class for managing the Rotate operator.
 *
 */

class Rotate final : public IOperator {
   public:
    /**
     * @brief Construct a new Op Rotate object
     *
     */
    Rotate() {}

    /**
     * @brief Destroy the Op Rotate object
     *
     */
    ~Rotate() {}

    /**
     * @brief Rotates a batch of images by a given angle in degrees clockwise.
     *
     * Limitations:
     *
     * Input:
     *       Supported TensorLayout(s): [TENSOR_LAYOUT_NHWC, TENSOR_LAYOUT_NCHW]
     *                        Channels: 1, 3
     *       Supported DataType(s):     [DATA_TYPE_U8, DATA_TYPE_F32,
     *                                  DATA_TYPE_S8]
     *
     * Output:
     *       Supported TensorLayout(s): [TENSOR_LAYOUT_NHWC, TENSOR_LAYOUT_NCHW]
     *                        Channels: 1, 3
     *       Supported DataType(s)     [DATA_TYPE_U8, DATA_TYPE_F32,
     *                                 DATA_TYPE_S8]
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
     * @param[in] stream The HIP stream to run this operator on.
     * @param[in] input Input tensor with image batch data
     * @param[out] output Output tensor for storing modified image batch data
     * @param[in] angle_deg The angle in degrees for which images are rotated
     * by.
     * @param[in] shift x and y coordinates to perform a shift after a rotation.
     * @param[in] interpolation The interpolation method to be applied to the
     * images.
     * @param[in] device The device to run this operation on. (Default: GPU)
     */
    void operator()(hipStream_t stream, const roccv::Tensor &input, const roccv::Tensor &output, const double angle_deg,
                    const double2 shift, const eInterpolationType interpolation,
                    const eDeviceType device = eDeviceType::GPU) const;

   private:
    WarpAffine m_op;
};
}  // namespace roccv