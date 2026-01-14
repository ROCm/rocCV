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

#include "core/tensor.hpp"
#include "i_operator.hpp"

namespace roccv {
class Reformat final : public IOperator {
   public:
    /**
     * @brief Constructs a Reformat object.
     *
     */
    explicit Reformat() {}

    /**
     * @brief Destroys a Reformat object.
     *
     */
    ~Reformat() {}

    /**
     * @brief Executes the Reformat operation on the given HIP stream.
     *
     * Reformat changes the layout of the input tensor to that of the output tensor by rearranging memory. Supported
     * layout conversions include:
     * - NHWC to NCHW (and vice versa)
     *
     * Limitations:
     *
     * Input:
     *       Supported TensorLayout(s): [NHWC, NCHW, HWC]
     *                        Channels: [1, 3, 4]
     *       Supported DataType(s):     [U8, S8, U16, S16, U32, S32, F32, F64]
     *
     * Output:
     *       Supported TensorLayout(s): [NHWC, NCHW, HWC]
     *                        Channels: [1, 3, 4]
     *       Supported DataType(s):     [U8, S8, U16, S16, U32, S32, F32, F64]
     *
     * Input/Output dependency:
     *
     *       Property      |  Input == Output
     *      -------------- | -------------
     *       TensorLayout  | No
     *       DataType      | Yes
     *       Channels      | Yes
     *       Width         | Yes
     *       Height        | Yes
     *       Batch Size    | Yes
     *
     * @param[in] stream The HIP stream to run this operation on.
     * @param[in] input The input tensor to reformat.
     * @param[out] output The output tensor to store the result.
     * @param[in] device The device to run this operation on. Default is GPU.
     */
    void operator()(hipStream_t stream, const Tensor& input, const Tensor& output,
                    const eDeviceType device = eDeviceType::GPU) const;
};
}  // namespace roccv