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
#include "op_histogram.hpp"

#include <hip/hip_runtime.h>
#include <stdio.h>

#include <cstring>

#include "common/array_wrapper.hpp"
#include "common/strided_data_wrap.hpp"
#include "common/validation_helpers.hpp"
#include "kernels/device/histogram_device.hpp"
#include "kernels/host/histogram_host.hpp"

namespace roccv {
Histogram::Histogram() {}

Histogram::~Histogram() {}

void Histogram::operator()(hipStream_t stream, const Tensor& input, const Tensor* mask, const Tensor& histogram,
                           eDeviceType device) {
    // Verify that the tensors are located on the right device (CPU or GPU).
    CHECK_TENSOR_DEVICE(input, device);
    CHECK_TENSOR_DEVICE(histogram, device);

    // Ensure all tensors are using supported datatypes
    CHECK_TENSOR_DATATYPES(input, eDataType::DATA_TYPE_U8);
    CHECK_TENSOR_DATATYPES(histogram, eDataType::DATA_TYPE_U32, eDataType::DATA_TYPE_S32);

    // Ensure all tensors are using supported layouts.
    CHECK_TENSOR_LAYOUT(input, eTensorLayout::TENSOR_LAYOUT_NHWC, eTensorLayout::TENSOR_LAYOUT_HWC);
    CHECK_TENSOR_LAYOUT(histogram, eTensorLayout::TENSOR_LAYOUT_HWC);

    if (mask != nullptr) {
        CHECK_TENSOR_COMPARISON(input.shape() == mask->shape());
    }

    const auto o_height = histogram.shape()[histogram.shape().layout().height_index()];
    const auto o_width = histogram.shape()[histogram.shape().layout().width_index()];
    const auto o_channels = histogram.shape()[histogram.shape().layout().channels_index()];

    const auto i_batch_i = input.shape().layout().batch_index();
    const auto i_batch = (i_batch_i >= 0) ? input.shape()[i_batch_i] : 1;
    const auto i_height = input.shape()[input.shape().layout().height_index()];
    const auto i_width = input.shape()[input.shape().layout().width_index()];
    const auto i_channels = input.shape()[input.shape().layout().channels_index()];

    if (o_width != 256) {
        throw Exception("Invalid width: output tensor must have width of 256.", eStatusType::INVALID_VALUE);
    }
    if (o_height != i_batch) {
        throw Exception(
            "Invalid height: output tensor must have height equal to "
            "input tensor batch size.",
            eStatusType::INVALID_COMBINATION);
    }
    if (i_channels != 1 || o_channels != 1) {
        throw Exception("Invalid channel size: tensors must have channel size of 1.", eStatusType::INVALID_VALUE);
    }

    auto batch_size = i_batch;
    auto input_height = i_height;
    auto input_width = i_width;

    auto input_data = input.exportData<roccv::TensorDataStrided>();
    auto histogram_data = histogram.exportData<roccv::TensorDataStrided>();

    if (device == eDeviceType::GPU) {
        HIP_VALIDATE_NO_ERRORS(hipMemset2DAsync(histogram_data.basePtr(),
                                                histogram_data.stride(histogram_data.shape().layout().height_index()),
                                                0, 256 * histogram.dtype().size(), batch_size, stream));

        const dim3 threads_block(256, 1, 1);

        const dim3 grid_size((input_height * input_width + threads_block.x - 1) / threads_block.x, 1, batch_size);
        const auto shared_mem_size = 256 * histogram.dtype().size();

        if (mask == nullptr) {
            Kernels::Device::histogram_kernel<uint8_t, int32_t>
                <<<grid_size, threads_block, shared_mem_size, stream>>>(
                    detail::get_sdwrapper<TENSOR_LAYOUT_NHWC>(input),
                    detail::get_sdwrapper<TENSOR_LAYOUT_NHWC>(histogram), batch_size, input_height, input_width);
        }
        else {
            Kernels::Device::histogram_kernel<uint8_t, int32_t>
                <<<grid_size, threads_block, shared_mem_size, stream>>>(
                    detail::get_sdwrapper<TENSOR_LAYOUT_NHWC>(input),
                    detail::get_sdwrapper<TENSOR_LAYOUT_NHWC>(histogram),
                    detail::get_sdwrapper<TENSOR_LAYOUT_NHWC>(*mask), batch_size, input_height, input_width);
        }
    } else if (device == eDeviceType::CPU) {
        auto memset_ptr = static_cast<uint8_t*>(histogram_data.basePtr());
        const auto memset_offset = histogram_data.stride(histogram_data.shape().layout().height_index());
        const auto memset_width = 256 * histogram.dtype().size();
        const auto memset_height = batch_size;
        const int memset_value = 0;
        for (int64_t i = 0; i < memset_height; i++) {
            auto curr_ptr = memset_ptr + memset_offset * i;
            memset(curr_ptr, memset_value, memset_width);
        }

        if (mask == nullptr) {
            Kernels::Host::histogram_kernel<uint8_t, int32_t>(
                detail::get_sdwrapper<TENSOR_LAYOUT_NHWC>(input),
                detail::get_sdwrapper<TENSOR_LAYOUT_NHWC>(histogram), batch_size, input_height, input_width);
        } else {
            Kernels::Host::histogram_kernel<uint8_t, int32_t>(
                detail::get_sdwrapper<TENSOR_LAYOUT_NHWC>(input),
                detail::get_sdwrapper<TENSOR_LAYOUT_NHWC>(histogram),
                detail::get_sdwrapper<TENSOR_LAYOUT_NHWC>(*mask), batch_size, input_height, input_width);
        }
    }
}
}  // namespace roccv