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
#include "op_cvt_color.hpp"

#include <hip/hip_runtime.h>

#include <iostream>

#include "common/conversion_helpers.hpp"
#include "common/strided_data_wrap.hpp"
#include "common/validation_helpers.hpp"
#include "kernels/device/cvt_color_device.hpp"
#include "kernels/host/cvt_color_host.hpp"

namespace roccv {
CvtColor::CvtColor() {}

CvtColor::~CvtColor() {}

void CvtColor::operator()(hipStream_t stream, const Tensor &input, Tensor &output, eColorConversionCode conversionCode,
                          eDeviceType device) {
    // Verify that the tensors are located on the right device (CPU or GPU).
    CHECK_TENSOR_DEVICE(input, device);
    CHECK_TENSOR_DEVICE(output, device);

    CHECK_TENSOR_COMPARISON(input.shape().layout() == output.shape().layout());

    // Ensure all tensors are using supported layouts.
    CHECK_TENSOR_LAYOUT(input, eTensorLayout::TENSOR_LAYOUT_NHWC, eTensorLayout::TENSOR_LAYOUT_HWC);
    CHECK_TENSOR_LAYOUT(output, eTensorLayout::TENSOR_LAYOUT_NHWC, eTensorLayout::TENSOR_LAYOUT_HWC);

    // Ensure all tensors are using supported datatypes
    CHECK_TENSOR_DATATYPES(input, eDataType::DATA_TYPE_U8);
    CHECK_TENSOR_DATATYPES(output, eDataType::DATA_TYPE_U8);

    const auto o_batch_i = output.shape().layout().batch_index();
    const auto o_batch = (o_batch_i >= 0) ? output.shape()[o_batch_i] : 1;
    const auto o_height = output.shape()[output.shape().layout().height_index()];
    const auto o_width = output.shape()[output.shape().layout().width_index()];
    const auto o_channels = output.shape()[output.shape().layout().channels_index()];

    const auto i_batch_i = input.shape().layout().batch_index();
    const auto i_batch = (i_batch_i >= 0) ? input.shape()[i_batch_i] : 1;
    const auto i_height = input.shape()[input.shape().layout().height_index()];
    const auto i_width = input.shape()[input.shape().layout().width_index()];
    const auto i_channels = input.shape()[input.shape().layout().channels_index()];

    if (o_batch != i_batch) {
        throw Exception("Invalid batch size: input != output", eStatusType::INVALID_COMBINATION);
    }

    eChannelType outputChannelType;
    if (conversionCode == COLOR_RGB2GRAY || conversionCode == COLOR_BGR2GRAY) {
        outputChannelType = eChannelType::Grayscale;
    }
    if (o_channels != i_channels) {
        if (outputChannelType != eChannelType::Grayscale)
            throw Exception("Invalid channel size: input != output", eStatusType::INVALID_COMBINATION);
    }
    if (o_batch <= 0) {
        throw Exception("Invalid batch size: must be greater than 0", eStatusType::OUT_OF_BOUNDS);
    }
    if (o_channels <= 0 || o_channels > 4) {
        throw Exception("Invalid channel size: must be greater than 0 and less than or equal to 4",
                        eStatusType::OUT_OF_BOUNDS);
    }
    if (o_height <= 0) {
        throw Exception("Invalid output height size: must be greater than 0", eStatusType::OUT_OF_BOUNDS);
    }
    if (o_width <= 0) {
        throw Exception("Invalid output width size: must be greater than 0", eStatusType::OUT_OF_BOUNDS);
    }
    if (i_height <= 0) {
        throw Exception("Invalid input height size: must be greater than 0", eStatusType::OUT_OF_BOUNDS);
    }
    if (i_width <= 0) {
        throw Exception("Invalid input width size: must be greater than 0", eStatusType::OUT_OF_BOUNDS);
    }

    auto batch_size = i_batch;
    auto height = i_height;
    auto width = i_width;

    if (device == eDeviceType::CPU) {
        eDataType inType = input.dtype().etype();

        switch (inType) {
            case eDataType::DATA_TYPE_U8: {
                if (conversionCode == COLOR_RGB2YUV) {
                    Kernels::Host::rgb_or_bgr_to_yuv<uint8_t>(detail::get_sdwrapper<TENSOR_LAYOUT_NHWC>(input),
                                                              detail::get_sdwrapper<TENSOR_LAYOUT_NHWC>(output), width,
                                                              height, batch_size, 0, 128);
                } else if (conversionCode == COLOR_BGR2YUV) {
                    Kernels::Host::rgb_or_bgr_to_yuv<uint8_t>(detail::get_sdwrapper<TENSOR_LAYOUT_NHWC>(input),
                                                              detail::get_sdwrapper<TENSOR_LAYOUT_NHWC>(output), width,
                                                              height, batch_size, 2, 128);
                } else if (conversionCode == COLOR_YUV2RGB) {
                    Kernels::Host::yuv_to_rgb_or_bgr<uint8_t>(detail::get_sdwrapper<TENSOR_LAYOUT_NHWC>(input),
                                                              detail::get_sdwrapper<TENSOR_LAYOUT_NHWC>(output), width,
                                                              height, batch_size, 0, 128);
                } else if (conversionCode == COLOR_YUV2BGR) {
                    Kernels::Host::yuv_to_rgb_or_bgr<uint8_t>(detail::get_sdwrapper<TENSOR_LAYOUT_NHWC>(input),
                                                              detail::get_sdwrapper<TENSOR_LAYOUT_NHWC>(output), width,
                                                              height, batch_size, 2, 128);
                } else if (conversionCode == COLOR_RGB2BGR) {
                    Kernels::Host::rgb_or_bgr_to_bgr_or_rgb<uint8_t>(detail::get_sdwrapper<TENSOR_LAYOUT_NHWC>(input),
                                                                     detail::get_sdwrapper<TENSOR_LAYOUT_NHWC>(output),
                                                                     width, height, batch_size, 0, 2);
                } else if (conversionCode == COLOR_BGR2RGB) {
                    Kernels::Host::rgb_or_bgr_to_bgr_or_rgb<uint8_t>(detail::get_sdwrapper<TENSOR_LAYOUT_NHWC>(input),
                                                                     detail::get_sdwrapper<TENSOR_LAYOUT_NHWC>(output),
                                                                     width, height, batch_size, 2, 0);
                } else if (conversionCode == COLOR_RGB2GRAY) {
                    Kernels::Host::rgb_or_bgr_to_grayscale<uint8_t>(detail::get_sdwrapper<TENSOR_LAYOUT_NHWC>(input),
                                                                    detail::get_sdwrapper<TENSOR_LAYOUT_NHWC>(output),
                                                                    width, height, batch_size, 0);
                } else if (conversionCode == COLOR_BGR2GRAY) {
                    Kernels::Host::rgb_or_bgr_to_grayscale<uint8_t>(detail::get_sdwrapper<TENSOR_LAYOUT_NHWC>(input),
                                                                    detail::get_sdwrapper<TENSOR_LAYOUT_NHWC>(output),
                                                                    width, height, batch_size, 2);
                } else {
                    throw Exception("Invalid input channel types", eStatusType::INVALID_COMBINATION);
                }
            } break;
            default:
                throw Exception("Invalid tensor data type", eStatusType::INVALID_VALUE);
        }
    } else if (device == eDeviceType::GPU) {
        dim3 blockSize(64, 16);
        dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y, batch_size);

        eDataType inType = input.dtype().etype();

        switch (inType) {
            case eDataType::DATA_TYPE_U8: {
                if (conversionCode == COLOR_RGB2YUV) {
                    Kernels::Device::rgb_or_bgr_to_yuv<uint8_t><<<gridSize, blockSize, 0, stream>>>(
                        detail::get_sdwrapper<TENSOR_LAYOUT_NHWC>(input),
                        detail::get_sdwrapper<TENSOR_LAYOUT_NHWC>(output), width, height, batch_size, 0, 128);
                } else if (conversionCode == COLOR_BGR2YUV) {
                    Kernels::Device::rgb_or_bgr_to_yuv<uint8_t><<<gridSize, blockSize, 0, stream>>>(
                        detail::get_sdwrapper<TENSOR_LAYOUT_NHWC>(input),
                        detail::get_sdwrapper<TENSOR_LAYOUT_NHWC>(output), width, height, batch_size, 2, 128);
                } else if (conversionCode == COLOR_YUV2RGB) {
                    Kernels::Device::yuv_to_rgb_or_bgr<uint8_t><<<gridSize, blockSize, 0, stream>>>(
                        detail::get_sdwrapper<TENSOR_LAYOUT_NHWC>(input),
                        detail::get_sdwrapper<TENSOR_LAYOUT_NHWC>(output), width, height, batch_size, 0, 128);
                } else if (conversionCode == COLOR_YUV2BGR) {
                    Kernels::Device::yuv_to_rgb_or_bgr<uint8_t><<<gridSize, blockSize, 0, stream>>>(
                        detail::get_sdwrapper<TENSOR_LAYOUT_NHWC>(input),
                        detail::get_sdwrapper<TENSOR_LAYOUT_NHWC>(output), width, height, batch_size, 2, 128);
                } else if (conversionCode == COLOR_RGB2BGR) {
                    Kernels::Device::rgb_or_bgr_to_bgr_or_rgb<uint8_t><<<gridSize, blockSize, 0, stream>>>(
                        detail::get_sdwrapper<TENSOR_LAYOUT_NHWC>(input),
                        detail::get_sdwrapper<TENSOR_LAYOUT_NHWC>(output), width, height, batch_size, 0, 2);
                } else if (conversionCode == COLOR_BGR2RGB) {
                    Kernels::Device::rgb_or_bgr_to_bgr_or_rgb<uint8_t><<<gridSize, blockSize, 0, stream>>>(
                        detail::get_sdwrapper<TENSOR_LAYOUT_NHWC>(input),
                        detail::get_sdwrapper<TENSOR_LAYOUT_NHWC>(output), width, height, batch_size, 2, 0);
                } else if (conversionCode == COLOR_RGB2GRAY) {
                    Kernels::Device::rgb_or_bgr_to_grayscale<uint8_t><<<gridSize, blockSize, 0, stream>>>(
                        detail::get_sdwrapper<TENSOR_LAYOUT_NHWC>(input),
                        detail::get_sdwrapper<TENSOR_LAYOUT_NHWC>(output), width, height, batch_size, 0);
                } else if (conversionCode == COLOR_BGR2GRAY) {
                    Kernels::Device::rgb_or_bgr_to_grayscale<uint8_t><<<gridSize, blockSize, 0, stream>>>(
                        detail::get_sdwrapper<TENSOR_LAYOUT_NHWC>(input),
                        detail::get_sdwrapper<TENSOR_LAYOUT_NHWC>(output), width, height, batch_size, 2);
                } else {
                    throw Exception("Invalid input channel types", eStatusType::INVALID_COMBINATION);
                }
            } break;
            default:
                throw Exception("Invalid tensors data type", eStatusType::INVALID_VALUE);
        }
    }
}
}  // namespace roccv