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

#include <hip/hip_runtime.h>
#include <operator_types.h>

#include <i_operator.hpp>

#include "core/tensor.hpp"

namespace roccv {
/**
 * @brief Class for managing the Advanced Color Conversion operator.
 *
 */
class AdvCvtColor final : public IOperator {
   public:
    /**
     * @brief Constructs the Advanced Color Conversion operator.
     */
    AdvCvtColor();

    /**
     * @brief Destroys the Advanced Color Conversion operator.
     */
    ~AdvCvtColor();

    /**
     * @brief Executes advanced color conversions with explicit color-spec coefficients.
     *
     * Limitations:
     *
     * Input:
     *       Supported TensorLayout(s): [NHWC, HWC]
     *       Supported DataType(s):     [U8]
     *
     * Output:
     *       Supported TensorLayout(s): [NHWC, HWC]
     *       Supported DataType(s):     [U8]
     *
     * Input/Output dependency:
     *
     *       Property      |  Input == Output
     *      -------------- | -------------
     *       TensorLayout  | Yes
     *       DataType      | Yes
     *       Width         | Yes
     *       Batch         | Yes
     *
     * Supported Color Conversion Codes:
     *
     *    - COLOR_RGB2YUV
     *    - COLOR_BGR2YUV
     *    - COLOR_YUV2RGB
     *    - COLOR_YUV2BGR
     *    - COLOR_YUV2RGB_NV12
     *    - COLOR_YUV2BGR_NV12
     *    - COLOR_YUV2RGB_NV21
     *    - COLOR_YUV2BGR_NV21
     *    - COLOR_RGB2YUV_NV12
     *    - COLOR_BGR2YUV_NV12
     *    - COLOR_RGB2YUV_NV21
     *    - COLOR_BGR2YUV_NV21
     *
     * Supported color specifications:
     *
     *    - BT601
     *    - BT709
     *    - BT2020
     *
     * @param[in] stream The HIP stream to run this operation on.
     * @param[in] input Input tensor.
     * @param[out] output Output tensor.
     * @param[in] conversionCode Color conversion code.
     * @param[in] colorSpec Color specification for conversion matrix coefficients.
     * @param[in] device Device to execute on (GPU or CPU).
     */
    void operator()(hipStream_t stream, const Tensor &input, Tensor &output, eColorConversionCode conversionCode,
                    eColorSpec colorSpec, eDeviceType device = eDeviceType::GPU);
};
}  // namespace roccv
