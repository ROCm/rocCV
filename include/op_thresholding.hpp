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

#include <operator_types.h>

#include <i_operator.hpp>

#include "core/tensor.hpp"


namespace roccv {
/**
 * @brief Class for managing the Threshold operator.
 *
 */
class Threshold final : public IOperator {
   public:
    /**
     * @brief Constructs a Threshold object.
     *
     */

    Threshold(eThresholdType threshType, int32_t maxBatchSize);

    /**
     * @brief Destroy the Threshold object
     *
     */
    ~Threshold();
    /**
     * @brief Executes the Threshold operation.
     * Applies a global threshold value that is the same for all pixels across
     * each image.
     *
     * Limitations:
     *
     * Input:
     *       Supported TensorLayout(s): [HWC, NHWC]
     *                        Channels: [1, 3, 4]
     *       Supported DataType(s):     [U8, U16, S16, F32, F64]
     *
     * Output:
     *       Supported TensorLayout(s): [HWC, NHWC]
     *                        Channels: [1, 3, 4]
     *       Supported DataType(s):     [U8, U16, S16, F32, F64]
     *
     * Input/Output dependency
     *
     *       Property      |  Input == Output
     *      -------------- | -------------
     *       Data Layout   | Yes
     *       Data Type     | Yes
     *       Number        | Yes
     *       Channels      | Yes
     *       Width         | Yes
     *       Height        | Yes
     *
     * thresh Tensor
     *
     *      Must be of layout type 'N' (dim = 1) with N equal to the number of
     *      images in the input tensor. Data Type must be a 64bit float (DATA_TYPE_F64).
     *
     * maxVal Tensor
     *
     *      Must be of layout type 'N' (dim = 1) with N equal to the number of
     *      images in the input tensor. Data Type must be a 64bit float (DATA_TYPE_F64).
     *      Required for all threshold types, but only used by THRESH_BINARY and
     *      THRESH_BINARY_INV.
     *
     * thresh and maxVal are expected to be contiguous in memory and located on
     * the device this operation runs on.
     *
     * Current Supported Threshold Types (threshType)
     * - THRESH_BINARY
     * - THRESH_BINARY_INV
     * - THRESH_TRUNC
     * - THRESH_TOZERO
     * - THRESH_TOZERO_INV
     *
     *
     * @param[in] stream The HIP stream to run this operation on.
     * @param[in] input Input tensor with image batch data
     * @param[out] output Output tensor for storing modified image batch data
     * @param[in] thresh An F64 tensor of layout N with one threshold value per
     * image in the input tensor.
     * @param[in] maxVal An F64 tensor of layout N with one maximum value per
     * image in the input tensor, used with the THRESH_BINARY and THRESH_BINARY_INV thresholding types.
     * @param[in] device The device which this operation should run on.
     * (Default: eDeviceType::GPU)
     */
    void operator()(hipStream_t stream, const roccv::Tensor& input, const roccv::Tensor& output,
                    const roccv::Tensor& thresh, const roccv::Tensor& maxVal, eDeviceType device = eDeviceType::GPU);

   private:
    eThresholdType m_threshType;
    int32_t m_maxBatchSize;
};
}  // namespace roccv