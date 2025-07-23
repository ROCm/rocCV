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
#include "i_operator.hpp"

namespace roccv {
class Composite final : public IOperator {
   public:
    Composite() {}
    ~Composite() {}

    /**
     * @brief Executes the Composite operation on the given HIP stream. Compositing blends the foreground image into the
     * background image using the values in a grayscale alpha mask.
     *
     *
     * Limitations:
     *
     * Foreground:
     *       Supported TensorLayout(s): [NHWC, HWC]
     *                        Channels: [3]
     *       Supported DataType(s):     [U8, F32]
     *       Notes:                     Must be the same shape and datatype as the background tensor.
     *
     * Background:
     *       Supported TensorLayout(s): [NHWC, HWC]
     *                        Channels: [3]
     *       Supported DataTypes(s):    [U8, F32]
     *       Notes:                     Must be the same shape and datatype as the foreground tensor.
     *
     * Mask:
     *       Supported TensorLayout(s): [NHWC, HWC]
     *                        Channels: [1]
     *       Supported DataTypes(s):    [U8, F32]
     *       Notes:                     Must be the same shape as the foreground/background tensor.
     *                                  Can use any of the supported datatypes.
     *
     * Output:
     *       Supported TensorLayout(s): [NHWC, HWC]
     *                        Channels: [3, 4]
     *       Supported DataTypes(s):    [U8, F32]
     *       Notes:                     Must be the same layout as the foreground/background/mask tensor. If 4 channels are selected,
     *                                  an alpha channel will be created with its value at 100%. Any of the supported datatypes can
     *                                  be used. This will convert the foreground/background images into the output datatype if they
     *                                  differ.
     *
     * Input/Output dependency:
     *
     *       Property      |  Input == Output
     *      -------------- | -------------
     *       TensorLayout  | Yes
     *       DataType      | No
     *       Channels      | No
     *       Width         | Yes
     *       Height        | Yes
     *       Batch         | Yes
     *
     * @param[in] stream The HIP stream to run this operation on.
     * @param[in] foreground Foreground images.
     * @param[in] background Background images.
     * @param[in] mask Greyscale mask image for the foreground. Must be a 1-channel grayscale image matching the
     * datatype of the foreground and background images.
     * @param[out] output Output tensor. If 4-channel, the alpha channel will be set to the max value, resulting in a
     * RGBA/BGRA image (depending on the layout of the input images).
     * @param[in] device The device to run this operation on. Default is GPU.
     */
    void operator()(hipStream_t stream, const Tensor& foreground, const Tensor& background, const Tensor& mask,
                    const Tensor& output, const eDeviceType device = eDeviceType::GPU) const;
};
}  // namespace roccv