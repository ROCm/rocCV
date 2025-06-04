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
#include "op_thresholding.hpp"

#include <hip/hip_runtime.h>

#include "common/strided_data_wrap.hpp"
#include "common/validation_helpers.hpp"
#include "core/exception.hpp"
#include "core/status_type.h"
#include "kernels/device/thresholding_device.hpp"
#include "kernels/host/thresholding_host.hpp"

namespace roccv {
Threshold::Threshold(eThresholdType threshType, int32_t maxBatchSize)
    : m_threshType(threshType), m_maxBatchSize(maxBatchSize) {}

Threshold::~Threshold() {}

void Threshold::operator()(hipStream_t stream, const roccv::Tensor &input, const roccv::Tensor &output,
                           const roccv::Tensor &thresh, const roccv::Tensor &maxVal, eDeviceType device) {
    // Verify that the tensors are located on the right device (CPU or GPU).
    CHECK_TENSOR_DEVICE(input, device);
    CHECK_TENSOR_DEVICE(output, device);

    // Ensure all tensors are using supported datatypes
    CHECK_TENSOR_DATATYPES(input, eDataType::DATA_TYPE_U8);
    CHECK_TENSOR_DATATYPES(output, eDataType::DATA_TYPE_U8);

    // Ensure all tensors are using supported layouts.
    CHECK_TENSOR_LAYOUT(input, eTensorLayout::TENSOR_LAYOUT_NHWC, eTensorLayout::TENSOR_LAYOUT_HWC);
    CHECK_TENSOR_LAYOUT(output, eTensorLayout::TENSOR_LAYOUT_NHWC, eTensorLayout::TENSOR_LAYOUT_HWC);

    // Ensure the layout and shapes for the input/output tensor match
    CHECK_TENSOR_COMPARISON(input.layout() == output.layout());
    CHECK_TENSOR_COMPARISON(input.shape() == output.shape());

    const auto o_batch_i = output.shape().layout().batch_index();
    const auto o_batch = (o_batch_i >= 0) ? output.shape()[o_batch_i] : 1;
    const auto o_channels = output.shape()[output.shape().layout().channels_index()];

    const auto i_batch_i = input.shape().layout().batch_index();
    const auto i_batch = (i_batch_i >= 0) ? input.shape()[i_batch_i] : 1;
    const auto i_height = input.shape()[input.shape().layout().height_index()];
    const auto i_width = input.shape()[input.shape().layout().width_index()];
    const auto i_channels = input.shape()[input.shape().layout().channels_index()];

    if (o_batch != i_batch) {
        throw Exception("Invalid batch size: input != output.", eStatusType::INVALID_COMBINATION);
    }

    if (o_channels != i_channels) {
        throw Exception("Invalid channel size: input != output.", eStatusType::INVALID_COMBINATION);
    }

    if (o_channels > 4) {
        throw Exception("Invalid channel size: cannot be greater than 4.", eStatusType::OUT_OF_BOUNDS);
    }

    if (m_maxBatchSize != i_batch) {
        throw Exception("Invalid batch size: Input Batch != maxBatchSize", eStatusType::INVALID_COMBINATION);
    }
    auto channels = i_channels;
    auto height = i_height;
    auto width = i_width;
    auto batch_size = m_maxBatchSize;

    auto input_data = input.exportData<roccv::TensorDataStrided>();
    auto output_data = output.exportData<roccv::TensorDataStrided>();
    auto thresh_data = thresh.exportData<roccv::TensorDataStrided>();
    auto maxVal_data = maxVal.exportData<roccv::TensorDataStrided>();

    if (device == eDeviceType::GPU) {
        dim3 block(64, 16);
        dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y, batch_size);

        switch (m_threshType) {
            case THRESH_BINARY:
                Kernels::Device::binary_generic_kernel<uint8_t><<<grid, block, 0, stream>>>(
                    detail::get_sdwrapper<TENSOR_LAYOUT_NHWC>(input), detail::get_sdwrapper<TENSOR_LAYOUT_NHWC>(output),
                    height, width, channels, static_cast<uint8_t *>(thresh_data.basePtr()),
                    static_cast<uint8_t *>(maxVal_data.basePtr()), batch_size);
                break;
            case THRESH_BINARY_INV:
                Kernels::Device::binary_inv_generic_kernel<uint8_t><<<grid, block, 0, stream>>>(
                    detail::get_sdwrapper<TENSOR_LAYOUT_NHWC>(input), detail::get_sdwrapper<TENSOR_LAYOUT_NHWC>(output),
                    height, width, channels, static_cast<uint8_t *>(thresh_data.basePtr()),
                    static_cast<uint8_t *>(maxVal_data.basePtr()), batch_size);
                break;
            case THRESH_TRUNC:
                Kernels::Device::trunc_generic_kernel<uint8_t><<<grid, block, 0, stream>>>(
                    detail::get_sdwrapper<TENSOR_LAYOUT_NHWC>(input), detail::get_sdwrapper<TENSOR_LAYOUT_NHWC>(output),
                    height, width, channels, static_cast<uint8_t *>(thresh_data.basePtr()),
                    static_cast<uint8_t *>(maxVal_data.basePtr()), batch_size);
                break;
            case THRESH_TOZERO:
                Kernels::Device::tozero_generic_kernel<uint8_t><<<grid, block, 0, stream>>>(
                    detail::get_sdwrapper<TENSOR_LAYOUT_NHWC>(input), detail::get_sdwrapper<TENSOR_LAYOUT_NHWC>(output),
                    height, width, channels, static_cast<uint8_t *>(thresh_data.basePtr()),
                    static_cast<uint8_t *>(maxVal_data.basePtr()), batch_size);
                break;
            case THRESH_TOZERO_INV:
                Kernels::Device::tozeroinv_generic_kernel<uint8_t><<<grid, block, 0, stream>>>(
                    detail::get_sdwrapper<TENSOR_LAYOUT_NHWC>(input), detail::get_sdwrapper<TENSOR_LAYOUT_NHWC>(output),
                    height, width, channels, static_cast<uint8_t *>(thresh_data.basePtr()),
                    static_cast<uint8_t *>(maxVal_data.basePtr()), batch_size);
                break;
        }
    } else if (device == eDeviceType::CPU) {
        switch (m_threshType) {
            case THRESH_BINARY:
                Kernels::Host::binary_generic_kernel<uint8_t>(
                    detail::get_sdwrapper<TENSOR_LAYOUT_NHWC>(input), detail::get_sdwrapper<TENSOR_LAYOUT_NHWC>(output),
                    height, width, channels, static_cast<uint8_t *>(thresh_data.basePtr()),
                    static_cast<uint8_t *>(maxVal_data.basePtr()), batch_size);
                break;
            case THRESH_BINARY_INV:
                Kernels::Host::binary_inv_generic_kernel<uint8_t>(
                    detail::get_sdwrapper<TENSOR_LAYOUT_NHWC>(input), detail::get_sdwrapper<TENSOR_LAYOUT_NHWC>(output),
                    height, width, channels, static_cast<uint8_t *>(thresh_data.basePtr()),
                    static_cast<uint8_t *>(maxVal_data.basePtr()), batch_size);
                break;
            case THRESH_TRUNC:
                Kernels::Host::trunc_generic_kernel<uint8_t>(
                    detail::get_sdwrapper<TENSOR_LAYOUT_NHWC>(input), detail::get_sdwrapper<TENSOR_LAYOUT_NHWC>(output),
                    height, width, channels, static_cast<uint8_t *>(thresh_data.basePtr()),
                    static_cast<uint8_t *>(maxVal_data.basePtr()), batch_size);
                break;
            case THRESH_TOZERO:
                Kernels::Host::tozero_generic_kernel<uint8_t>(
                    detail::get_sdwrapper<TENSOR_LAYOUT_NHWC>(input), detail::get_sdwrapper<TENSOR_LAYOUT_NHWC>(output),
                    height, width, channels, static_cast<uint8_t *>(thresh_data.basePtr()),
                    static_cast<uint8_t *>(maxVal_data.basePtr()), batch_size);
                break;
            case THRESH_TOZERO_INV:
                Kernels::Host::tozeroinv_generic_kernel<uint8_t>(
                    detail::get_sdwrapper<TENSOR_LAYOUT_NHWC>(input), detail::get_sdwrapper<TENSOR_LAYOUT_NHWC>(output),
                    height, width, channels, static_cast<uint8_t *>(thresh_data.basePtr()),
                    static_cast<uint8_t *>(maxVal_data.basePtr()), batch_size);
                break;
        }
    }
}
}  // namespace roccv