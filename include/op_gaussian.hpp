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

#include "core/detail/allocators/default_allocator.hpp"
#include "core/tensor.hpp"
#include "i_operator.hpp"
#include "operator_types.h"

namespace roccv {

/**
 * @brief Class for managing the Gaussian operator.
 *
 */
class Gaussian final : public IOperator {
   public:
    /**
     * @brief Constructs a Gaussian object.
     *
     */
    Gaussian(int32_t maxKernelWidth, int32_t maxKernelHeight);

    /**
     * @brief Destroy the Gaussian object
     *
     */
    ~Gaussian();

    /**
     * @brief Executes the Gaussian operation on the given HIP stream.
     *
     *
     * Limitations:
     *
     * Input:
     *       Supported TensorLayout(s): [NHWC, HWC]
     *                        Channels: [1, 3, 4]
     *       Supported DataType(s):     [U8, U16, S16, S32, F32]
     *
     * Output:
     *       Supported TensorLayout(s): [NHWC, HWC]
     *                        Channels: [1, 3, 4]
     *       Supported DataType(s):     [U8, U16, S16, S32, F32]
     *
     * // TODO: clear documentation about when valid, when inferred from sigma, relation to max, etc.
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
     * @param[out] output Output tensor.
     * @param[in] kernelWidth Gaussian kernel width.
     * @param[in] kernelHeight Gaussian kernel height.
     * @param[in] sigmaX Gaussian kernel standard deviation in X direction.
     * @param[in] sigmaY Gaussian kernel standard deviation in Y direction.
     * @param[in] borderMode A border type to identify the pixel extrapolation
     * method (e.g. BORDER_TYPE_CONSTANT or BORDER_TYPE_REPLICATE).
     * @param[in] device The device which this operation should run on. (Default: eDeviceType::GPU)
     */
    void operator()(hipStream_t stream, const Tensor& input, Tensor& output, int kernelWidth, int kernelHeight,
                    double sigmaX, double sigmaY, eBorderType borderMode, eDeviceType device = eDeviceType::GPU) const;

   private:
    int32_t m_maxKernelWidth;
    int32_t m_maxKernelHeight;
    float* m_hostKernelMem = nullptr;
    float* m_deviceKernelMem = nullptr;
    DefaultAllocator m_allocator;
};
}  // namespace roccv