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

namespace roccv {
/**
 * @brief Class for managing the remap operator.
 *
 */
class Remap final : public IOperator {
   public:
    /**
     * @brief Constructs an Remap object.
     *
     */

    Remap();

    /**
     * @brief Destroy the Remap object
     *
     */
    ~Remap();
    /**
     * @brief Construct a new Remap object.
     * The object can be used to remap the pixels in an image according
     * to a new mapping given by a map tensor containing pixel coordinates.
     *
     * Limitations:
     *
     * Input:
     *       Supported TensorLayout(s): [NHWC, HWC]
     *                        Channels: [1 , 3, 4]
     *       Supported DataType(s):     [U8]
     *
     * Output:
     *       Supported TensorLayout(s): [NHWC, HWC]
     *                        Channels: [1 , 3, 4]
     *       Supported DataType(s):     [U8]
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
     *       Batch Size    | Yes
     *
     *  Currently supported remap types:
     *      - REMAP_ABSOLUTE
     *
     *
     * @param[in] stream The HIP stream to run this operation on.
     * @param[in] input Input tensor with image batch data
     * @param[out] output Output tensor for storing modified image batch data
     * @param[in] map Map tensor containing absolute or relative positions for how to
     * remap the pixels of the input tensor to the output tensor
     * @param[in] inInterpolation  Interpolation type to be used when getting values from the input tensor.
     * @param[in] mapInterpolation Interpolation type to be used when getting indices from the map tensor.
     * @param[in] mapValueType Determines how the values in the map are interpreted.
     * @param[in] alignCorners Set to true if corner values are aligned to center points of corner pixels
     * and set to false if they are aligned by the corner points of the corner pixels.
     * @param[in] borderType A border type to identify the pixel extrapolation
     * method (e.g. BORDER_TYPE_CONSTANT or BORDER_TYPE_REPLICATE)
     * @param[in] borderValue Set as 0 unless using a constant border.
     * @param[in] device The device which this operation should run on.
     * (Default: eDeviceType::GPU)
     */
    void operator()(hipStream_t stream, const roccv::Tensor& input, const roccv::Tensor& output,
                    const roccv::Tensor& map, const eInterpolationType inInterpolation,
                    const eInterpolationType mapInterpolation, const eRemapType mapValueType, const bool alignCorners,
                    const eBorderType borderType, const float4 borderValue, eDeviceType device = eDeviceType::GPU);
};
}  // namespace roccv