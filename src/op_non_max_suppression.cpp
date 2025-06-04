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
#include "op_non_max_suppression.hpp"

#include <hip/hip_runtime.h>

#include "common/validation_helpers.hpp"
#include "core/hip_assert.h"
#include "kernels/device/non_max_suppression_device.hpp"
#include "kernels/host/non_max_suppression_host.hpp"

namespace roccv {
void NonMaximumSuppression::operator()(hipStream_t stream, const Tensor& input, const Tensor& output,
                                       const Tensor& scores, float scoreThreshold, float iouThreshold,
                                       const eDeviceType device) const {
    // Validate input tensor
    CHECK_TENSOR_DEVICE(input, device);
    CHECK_TENSOR_LAYOUT(input, TENSOR_LAYOUT_NW, TENSOR_LAYOUT_NWC);
    CHECK_TENSOR_DATATYPES(input, DATA_TYPE_4S16, DATA_TYPE_S16);

    if (input.layout() == TENSOR_LAYOUT_NW) {
        if (input.dtype() != DATA_TYPE_4S16) {
            throw Exception("Input tensor with layout NW, must have datatype 4S16.", eStatusType::INVALID_COMBINATION);
        }
    }

    else if (input.layout() == TENSOR_LAYOUT_NWC) {
        if (input.dtype() != DATA_TYPE_S16) {
            throw Exception("Input tensor with layout NWC must have datatype S16.", eStatusType::INVALID_COMBINATION);
        }

        if (input.shape(2) != 4) {
            throw Exception("Input tensor with layout NWC must have last shape dimension as 4.",
                            eStatusType::INVALID_COMBINATION);
        }
    }

    // Validate output tensor
    CHECK_TENSOR_DEVICE(output, device);
    CHECK_TENSOR_LAYOUT(output, TENSOR_LAYOUT_NW, TENSOR_LAYOUT_NWC);
    CHECK_TENSOR_DATATYPES(output, DATA_TYPE_U8);

    if (output.layout() == TENSOR_LAYOUT_NWC) {
        if (output.shape(2) != 1) {
            throw Exception("Output tensor with layout NWC must have last shape dimension as 1.",
                            eStatusType::INVALID_COMBINATION);
        }
    }

    // Validate scores tensor
    CHECK_TENSOR_DEVICE(scores, device);
    CHECK_TENSOR_LAYOUT(scores, TENSOR_LAYOUT_NW, TENSOR_LAYOUT_NWC);
    CHECK_TENSOR_DATATYPES(scores, DATA_TYPE_F32);

    if (scores.layout() == TENSOR_LAYOUT_NWC) {
        if (scores.shape(2) != 1) {
            throw Exception("Scores tensor with layout NWC must have last shape dimension as 1.",
                            eStatusType::INVALID_COMBINATION);
        }
    }

    // Validate other parameters
    if (iouThreshold <= 0.0f || iouThreshold > 1.0f) {
        throw Exception("IoU threshold must be between (0, 1.0]", eStatusType::INVALID_VALUE);
    }

    int numBoxes = input.shape(1);
    int numBatches = input.shape(0);

    // Ensure the number of samples and boxes per sample matches across all tensors.
    CHECK_TENSOR_COMPARISON(output.shape(1) == numBoxes);
    CHECK_TENSOR_COMPARISON(output.shape(0) == numBatches);

    CHECK_TENSOR_COMPARISON(scores.shape(1) == numBoxes);
    CHECK_TENSOR_COMPARISON(scores.shape(0) == numBatches);

    // Create tensor views to conform to the expected data types of the NMS kernel. These shapes should be validated,
    // but typically involve going from non-vectorized to vectorized shapes. These shapes should be validated
    // beforehand. For example: an input tensor with shape and datatype [NWC, <batches, boxes, 4>, S16] will be
    // reinterpreted as [NW, <batches, boxes, 4S16>]. In any case, the underlying data is structured the same.
    Tensor inputReshaped =
        input.reshape(TensorShape(TensorLayout(TENSOR_LAYOUT_NW), {numBatches, numBoxes}), DataType(DATA_TYPE_4S16));
    Tensor outputReshaped =
        output.reshape(TensorShape(TensorLayout(TENSOR_LAYOUT_NW), {numBatches, numBoxes}), DataType(DATA_TYPE_U8));
    Tensor scoresReshaped =
        scores.reshape(TensorShape(TensorLayout(TENSOR_LAYOUT_NW), {numBatches, numBoxes}), DataType(DATA_TYPE_F32));

    // Launch nms kernel
    switch (device) {
        case eDeviceType::GPU: {
            dim3 block(256, 1, 1);
            dim3 grid((numBoxes + block.x - 1) / block.x, 1, numBatches);
            Kernels::Device::non_maximum_suppression<<<grid, block, 0, stream>>>(
                GenericTensorWrapper<short4>(inputReshaped), GenericTensorWrapper<uint8_t>(outputReshaped),
                GenericTensorWrapper<float>(scoresReshaped), numBoxes, scoreThreshold, iouThreshold);
            break;
        }

        case eDeviceType::CPU: {
            Kernels::Host::non_maximum_suppression(
                GenericTensorWrapper<short4>(inputReshaped), GenericTensorWrapper<uint8_t>(outputReshaped),
                GenericTensorWrapper<float>(scoresReshaped), numBoxes, scoreThreshold, iouThreshold);
            break;
        }
    }
}
}  // namespace roccv
