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

#include "op_gamma_contrast.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstring>
#include <iostream>
#include <vector>

#include "common/array_wrapper.hpp"
#include "common/math_vector.hpp"
#include "common/strided_data_wrap.hpp"
#include "common/validation_helpers.hpp"
#include "core/tensor.hpp"
#include "kernels/device/gamma_contrast_device.hpp"
#include "kernels/host/gamma_contrast_host.hpp"

namespace roccv {
GammaContrast::GammaContrast() {}

GammaContrast::~GammaContrast() {}

void GammaContrast::operator()(hipStream_t stream, const roccv::Tensor &input, const roccv::Tensor &output,
                               const roccv::Tensor &gamma, eDeviceType device) {
    CHECK_TENSOR_DEVICE(input, device);
    CHECK_TENSOR_DEVICE(output, device);
    CHECK_TENSOR_DEVICE(gamma, device);
    CHECK_TENSOR_DATATYPES(input, eDataType::DATA_TYPE_U8);
    CHECK_TENSOR_DATATYPES(output, eDataType::DATA_TYPE_U8);
    CHECK_TENSOR_DATATYPES(gamma, eDataType::DATA_TYPE_F32);
    CHECK_TENSOR_LAYOUT(input, eTensorLayout::TENSOR_LAYOUT_NHWC, eTensorLayout::TENSOR_LAYOUT_HWC);
    CHECK_TENSOR_LAYOUT(output, eTensorLayout::TENSOR_LAYOUT_NHWC, eTensorLayout::TENSOR_LAYOUT_HWC);
    CHECK_TENSOR_LAYOUT(gamma, eTensorLayout::TENSOR_LAYOUT_N);

    CHECK_TENSOR_COMPARISON(input.layout() == output.layout());
    CHECK_TENSOR_COMPARISON(input.shape() == output.shape());

    const auto batch_i = input.shape().layout().batch_index();
    const auto batch = (batch_i >= 0) ? input.shape()[batch_i] : 1;
    const auto height = input.shape()[input.shape().layout().height_index()];
    const auto width = input.shape()[input.shape().layout().width_index()];
    const auto channels = input.shape()[input.shape().layout().channels_index()];

    if (channels < 1 || channels > 4) {
        throw Exception("Invalid channel size: must be between 1 and 4.", eStatusType::INVALID_COMBINATION);
    }

    auto gamma_data = gamma.exportData<roccv::TensorDataStrided>();

    if (device == eDeviceType::GPU) {
        dim3 block(64, 16);
        dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y, batch);
       
        Kernels::Device::gamma_contrast_wrapped_u8<uchar3>
            <<<grid, block, 0, stream>>>(detail::get_sdwrapper<TENSOR_LAYOUT_NHWC>(input),
                detail::get_sdwrapper<TENSOR_LAYOUT_NHWC>(output), batch, width,
                height, static_cast<float *>(gamma_data.basePtr()));
    }
    else if (device == eDeviceType::CPU) {
        Kernels::Host::gamma_contrast_wrapped_u8<uchar3>(detail::get_sdwrapper<TENSOR_LAYOUT_NHWC>(input),
            detail::get_sdwrapper<TENSOR_LAYOUT_NHWC>(output),
            batch, width, height,
            static_cast<float *>(gamma_data.basePtr()));
        }
}
}  // namespace roccv