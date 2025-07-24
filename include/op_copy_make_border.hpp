/*
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#pragma once

#include "core/tensor.hpp"
#include "core/util_enums.h"
#include "i_operator.hpp"
#include "operator_types.h"

namespace roccv {
/**
 * @brief Class for managing the CopyMakeBorder operator.
 *
 */
class CopyMakeBorder final : public IOperator {
   public:
    explicit CopyMakeBorder() {}
    ~CopyMakeBorder() {}

    /**
     * @brief Executes the CopyMakeBorder operation on the given HIP stream and device. This operation will create a
     * border around the images based on the given border mode. The output tensor's height must be of size (input.height
     * + top * 2), the width must be of size (input.width + left * 2) to create a uniform border around the input
     * images.
     *
     * Limitations:
     *
     * Input:
     *       Supported TensorLayout(s): [NHWC, HWC]
     *                        Channels: [1, 3, 4]
     *       Supported DataType(s):     [U8, S8, U32, S32, F32]
     *
     * Output:
     *       Supported TensorLayout(s): [NHWC, HWC]
     *                        Channels: [1, 3, 4]
     *       Supported DataType(s):     [U8, S8, U32, S32, F32]
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
     * @param stream The HIP stream to execute this operation on.
     * @param input Tensor representing the input images.
     * @param output Tensor representing the output images.
     * @param top The top-most pixel of the output images where the border should end.
     * @param left The left-most pixel of the output images where the border should end.
     * @param border_mode The border mode used for creating the image border.
     * @param border_value The color of the border is a constant border type is used.
     * @param device The device to execute this operation on. Defaults to GPU.
     */
    void operator()(hipStream_t stream, const Tensor& input, const Tensor& output, int32_t top, int32_t left,
                    eBorderType border_mode, float4 border_value, const eDeviceType device = eDeviceType::GPU) const;
};
}  // namespace roccv