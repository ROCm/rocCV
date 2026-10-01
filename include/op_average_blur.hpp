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

#include <mutex>

#include "core/detail/allocators/default_allocator.hpp"
#include "core/tensor.hpp"
#include "i_operator.hpp"
#include "operator_types.h"

namespace roccv {
/**
 * @brief Class for managing the Average Blur operator.
 *
 */
class AverageBlur : public IOperator {
   public:
    /**
     * @brief Constructs an AverageBlur object.
     *
     * @param maxKernelWidth [in] The maximum kernel width that will be used by the operator. Must be positive (> 0).
     * @param maxKernelHeight [in] The maximum kernel height that will be used by the operator. Must be positive(> 0).
     */
    AverageBlur(int32_t maxKernelWidth, int32_t maxKernelHeight);

    /**
     * @brief Destroy the AverageBlur object
     *
     */
    ~AverageBlur();

    /**
     * @brief Executes the Average Blur operation on the given HIP stream.
     * Note that the Average Blur operator is stateful, and has concurrency control such that back-to-back calls on the
     * same AverageBlur operator object will be serialized.
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
     * @param[out] output Output tensor.
     * @param[in] kernelWidth AverageBlur kernel width. Must be positive, odd, and less than m_maxKernelWidth.
     * @param[in] kernelHeight AverageBlur kernel height. Must be positive, odd, and less than m_maxKernelHeight.
     * @param[in] anchorX Kernel anchor in X direction. Must be nonnegative and less than kernelWidth, or -1 to indicate
     * center.
     * @param[in] anchorY Kernel anchor in Y direction. Must be nonnegative and less than kernelHeight, or -1 to
     * indicate center.
     * @param[in] borderMode A border type to identify the pixel extrapolation
     * method (e.g. BORDER_TYPE_CONSTANT or BORDER_TYPE_REPLICATE).
     * @param[in] device The device which this operation should run on. (Default: eDeviceType::GPU)
     */
    void operator()(hipStream_t stream, const Tensor& input, Tensor& output, int kernelWidth, int kernelHeight,
                    int anchorX, int anchorY, eBorderType borderMode, eDeviceType device = eDeviceType::GPU);

   private:
    int32_t m_maxKernelWidth;
    int32_t m_maxKernelHeight;
    DefaultAllocator m_allocator;
    std::mutex m_bufferMutex;
    bool m_use2D = false;
    float* m_hostKernelMem = nullptr;
    float* m_hostKernelMemH = nullptr;
    float* m_hostKernelMemV = nullptr;
    float* m_deviceKernelMem = nullptr;
    float* m_deviceKernelMemH = nullptr;
    float* m_deviceKernelMemV = nullptr;
    hipEvent_t m_completionEvent = nullptr;
};
}  // namespace roccv
