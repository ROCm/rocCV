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
#include "operator_types.h"

namespace roccv {
/**
 * @brief Class for managing the Resize operator.
 *
 */
class Resize final : public IOperator {
   public:
    /**
     * @brief Construct a new Op Resize object
     */
    Resize() {}

    /**
     * @brief Destroy the Op Resize object
     *
     */
    ~Resize() {}

    /**
     * @brief Resizes input images to the shape of the output images using an interpolation mode for
     * upscaling/downscaling.
     *
     * Limitations:
     *
     * Input:
     *       Supported TensorLayout(s): [NHWC, HWC]
     *                        Channels: [1, 3, 4]
     *       Supported DataType(s):     [U8, F32]
     *
     * Output:
     *       Supported TensorLayout(s): [NHWC, HWC]
     *                        Channels: [1, 3, 4]
     *       Supported DataType(s)      [U8, F32]
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
     *       Batch         | Yes
     *
     * Supported interpolation modes: [NEAREST, LINEAR]
     *
     * @param[in] stream The HIP stream to run this operator on.
     * @param[in] input Input tensor with image batch data.
     * @param[out] output Output tensor for storing modified image batch data.
     * @param[in] interpolation The interpolation method used when resizing images.
     * @param[in] device The device to run this operator on. (Default: GPU).
     */
    void operator()(hipStream_t stream, const Tensor &in, const Tensor &output, const eInterpolationType interpolation,
                    const eDeviceType device = eDeviceType::GPU) const;
};
}  // namespace roccv
