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

#include "core/tensor.hpp"
#include "i_operator.hpp"
#include "operator_types.h"

namespace roccv {

// Flag to interpret the scale tensor as the standard deviation for normalize.
#define ROCCV_NORMALIZE_SCALE_IS_STDDEV 1

/**
 * @brief Class for managing the Normalize operator.
 *
 */
class Normalize final : public IOperator {
   public:
    /**
     * @brief Constructs an OpNormalize object.
     *
     */
    Normalize() {}

    /**
     * @brief Destroy the Op Normalize object
     */
    ~Normalize() {}

    /**
     * @brief Executes the Normalize operation on the given HIP stream.
     *
     * Normalization subtracts the base and multiplies by a provided scale value. It is calculated as follows:
     *
     * output[idx] = (input[idx] - base[param_idx]) * scale[param_idx] * global_scale + shift
     *
     * Optionally, if the ROCCV_NORMALIZE_SCALE_IS_STDDEV flag is set, the scale tensor is interpreted as the standard
     * deviation and will be calculated as follows (where epsilon is provided to ensure numerical stability):
     *
     * output[idx] = (input[idx] - base[param_idx]) * (1 / sqrt(sqr(scale[param_idx]) + epsilon)) * global_scale +
     * shift.
     *
     * Limitations:
     *
     * Input:
     *       Supported TensorLayout(s): [NHWC, HWC]
     *       Supported DataType(s):     [U8, S8, S16, U32, S32, F32]
     *
     * Output:
     *       Supported TensorLayout(s): [NHWC, HWC]
     *       Supported DataType(s):     [U8, S8, S16, U32, S32, F32]
     *
     * base:
     *      Supported TensorLayout(s): [NHWC, HWC]
     *      Supported DataType(s):     [F32]
     *
     * scale:
     *      Supported TensorLayout(s): [NHWC, HWC]
     *      Supported DataType(s):     [F32]
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
     * @param[in] base A tensor containing base values. Layout must match that of the input/output tensors. Shape
     * dimensions can either be 1, signifying that the base will be masked across that dimension, or match the
     * corresponding dimension of the input/output tensor's shape in the image. For example, a shape of [1, 1, 1, C]
     * will apply the base value across the channel dimension of the image, and a shape of [1, 1, W, C] will apply the
     * base value across the width and channel dimensions (where W corresponds to the image width and C corresponds to
     * the channel count). The channel count must match that of the input/output images.
     * @param[in] scale A tensor containing scale values. This tensor exhibits the same rules as the base tensor.
     * @param[out] output Output tensor.
     * @param[in] global_scale Scales the output by a constant value after normalization.
     * @param[in] shift Shifts the value after scaling.
     * @param[in] epsilon An epsilon value to add to the standard deviation to ensure numerical stability.
     * @param[in] flags Flags to specify the behavior of the normalization. If ROCCV_NORMALIZE_SCALE_IS_STDDEV is set,
     * the scale tensor will be interpreted at the standard deviation values rather than the scale.
     * @param[in] device The device which this operation should run on. (Default: eDeviceType::GPU)
     */
    void operator()(hipStream_t stream, const Tensor& input, const Tensor& base, const Tensor& scale,
                    const Tensor& output, float global_scale, float shift, float epsilon, uint32_t flags,
                    const eDeviceType device = eDeviceType::GPU) const;
};
}  // namespace roccv