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
 * @brief Class for managing the Laplacian operator.
 *
 */
class Laplacian final : public IOperator {
   public:
    /**
     * @brief Construct a new Laplacian object.
     *
     * Limitations:
     *
     * Input:
     *       Supported TensorLayout(s): [HWC, NHWC]
     *                        Channels: [1, 3, 4]
     *       Supported DataType(s):     [U8, U16, F32]
     *
     * Output:
     *       Supported TensorLayout(s): [HWC, NHWC]
     *                        Channels: [1, 3, 4]
     *       Supported DataType(s):     [U8, U16, F32]
     *
     * Parameter requirements:
     *       ksize: Must be 1 or 3.
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
     *
     * @param[in] stream The HIP stream to run this operator on.
     * @param[in] input Input tensor with image data.
     * @param[out] output Output tensor for storing modified image data.
     * @param[in] ksize Aperture size used to compute the second derivative filters. Must be 1 or 3.
     * @param[in] scale Scale factor for the Laplacian values.
     * @param[in] borderMode A border type to identify the pixel extrapolation method.
     * @param[in] device The device to run this operator on. (Default: GPU)
     */
    void operator()(hipStream_t stream, const roccv::Tensor &input, const roccv::Tensor &output, int32_t ksize,
                    float scale, eBorderType borderMode, eDeviceType device = eDeviceType::GPU) const;
};
}  // namespace roccv