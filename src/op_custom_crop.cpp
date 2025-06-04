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
#include "op_custom_crop.hpp"

#include <hip/hip_runtime.h>

#include "common/strided_data_wrap.hpp"
#include "common/validation_helpers.hpp"
#include "kernels/device/custom_crop_device.hpp"
#include "kernels/host/custom_crop_host.hpp"

namespace roccv {
void CustomCrop::operator()(hipStream_t stream, const Tensor& input, const Tensor& output, const Box_t cropRect,
                            const eDeviceType device) const {
    CHECK_TENSOR_DEVICE(input, device);
    CHECK_TENSOR_LAYOUT(input, eTensorLayout::TENSOR_LAYOUT_NHWC);
    CHECK_TENSOR_DATATYPES(input, eDataType::DATA_TYPE_U8);

    size_t batchSize = input.shape(input.layout().batch_index());
    size_t channels = input.shape(input.layout().channels_index());

    CHECK_TENSOR_COMPARISON(channels > 0 && channels <= 4);

    CHECK_TENSOR_DEVICE(output, device);
    CHECK_TENSOR_COMPARISON(input.layout() == output.layout());
    CHECK_TENSOR_COMPARISON(input.dtype() == output.dtype());
    CHECK_TENSOR_COMPARISON(output.shape(output.layout().channels_index()) == channels);
    CHECK_TENSOR_COMPARISON(output.shape(output.layout().batch_index()) == batchSize);

    switch (device) {
        case eDeviceType::GPU: {
            dim3 blockDim(16, 16, channels);
            dim3 gridDim((cropRect.width + blockDim.x - 1) / blockDim.x,
                         (cropRect.height + blockDim.y - 1) / blockDim.y, batchSize);

            Kernels::Device::custom_crop<uint8_t><<<gridDim, blockDim, 0, stream>>>(
                detail::get_sdwrapper<eTensorLayout::TENSOR_LAYOUT_NHWC>(input),
                detail::get_sdwrapper<eTensorLayout::TENSOR_LAYOUT_NHWC>(output), cropRect, channels, batchSize);
            break;
        }

        case eDeviceType::CPU: {
            Kernels::Host::custom_crop<uint8_t>(detail::get_sdwrapper<eTensorLayout::TENSOR_LAYOUT_NHWC>(input),
                                                detail::get_sdwrapper<eTensorLayout::TENSOR_LAYOUT_NHWC>(output),
                                                cropRect, channels, batchSize);
            break;
        }
    }
}
}  // namespace roccv