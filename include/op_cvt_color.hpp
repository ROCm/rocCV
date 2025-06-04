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
/**
 * @brief Class for managing the Color Conversion operator.
 *
 */
namespace roccv {
class CvtColor final : public IOperator {
   public:
    /**
     * @brief Constructs a Color Conversion object. The object can be used to
     * convert the desired image from BGR or RGB to YUV and vice-versa. It can
     * also convert from BGR or RGB to a single channel grayscale image
     *
     */

    CvtColor();

    /**
     * @brief Destroy the Color Conversion object
     *
     */
    ~CvtColor();

    /**
     * Limitations:
     *
     * Input:
     *       Supported TensorLayout(s): [NHWC, HWC]
     *                        Channels: [1, 3, 4]
     *       Supported DataType(s):     [U8]
     *
     * Output:
     *       Supported TensorLayout(s): [NHWC, HWC]
     *                        Channels: [1, 3, 4]
     *       Supported DataType(s):     [U8]
     *
     * Input/Output dependency:
     *
     *       Property      |  Input == Output
     *      -------------- | -------------
     *       TensorLayout  | Yes
     *       DataType      | Yes
     *       Channels      | No
     *       Width         | Yes
     *       Height        | Yes
     *       Batch         | Yes
     *       Channel Type  | No
     *
     * Supported Color Conversion Codes:
     *
     *    - COLOR_RGB2YUV
     *    - COLOR_BGR2YUV
     *    - COLOR_YUV2RGB
     *    - COLOR_YUV2BGR
     *    - COLOR_RGB2BGR
     *    - COLOR_BGR2RGB
     *    - COLOR_RGB2GRAY
     *    - COLOR_BGR2GRAY
     *
     * @param[in] stream The HIP stream to run this operation on.
     * @param[in] input Input tensor with image data.
     * @param[out] output Output tensor for storing modified image data.
     * @param[in] conversionCode The color conversion code.
     * All supported color conversions are in the documentation.
     * @param[in] device The device which this operation should run on.
     * (Default: eDeviceType::GPU)
     */
    void operator()(hipStream_t stream, const Tensor &input, Tensor &output, eColorConversionCode conversionCode,
                    eDeviceType device = eDeviceType::GPU);
};
}  // namespace roccv