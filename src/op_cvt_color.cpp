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

#include "common/array_wrapper.hpp"
#include "common/conversion_helpers.hpp"
#include "common/math_vector.hpp"
#include "common/strided_data_wrap.hpp"
#include "common/validation_helpers.hpp"
#include "core/tensor.hpp"
#include "core/wrappers/image_wrapper.hpp"
#include "kernels/device/cvt_color_device.hpp"
#include "kernels/host/cvt_color_host.hpp"

namespace roccv {
CvtColor::CvtColor() {}

CvtColor::~CvtColor() {}

template <typename T>
void dispatch_cvt_color(hipStream_t stream, const Tensor &input, const Tensor &output, int64_t width, int64_t height,
                        int64_t batch_size, int index, float delta, const eColorConversionCode conversionCode,
                        const eDeviceType device) {
    ImageWrapper<T> inputWrapper(input);
    ImageWrapper<T> outputWrapper(output);

    if (device == eDeviceType::GPU) {
        dim3 blockSize(64, 16);
        dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y, batch_size);
        switch (conversionCode) {
            case COLOR_RGB2YUV:
            case COLOR_BGR2YUV:
                Kernels::Device::rgb_or_bgr_to_yuv<T>
                    <<<gridSize, blockSize, 0, stream>>>(inputWrapper, outputWrapper, index, delta);
                break;
            case COLOR_YUV2RGB:
            case COLOR_YUV2BGR:
                Kernels::Device::yuv_to_rgb_or_bgr<T>
                    <<<gridSize, blockSize, 0, stream>>>(inputWrapper, outputWrapper, index, delta);
                break;
            case COLOR_RGB2BGR:
            case COLOR_BGR2RGB:
                Kernels::Device::rgb_or_bgr_to_bgr_or_rgb<T>
                    <<<gridSize, blockSize, 0, stream>>>(inputWrapper, outputWrapper, index, delta);
                break;
            case COLOR_RGB2GRAY:
            case COLOR_BGR2GRAY: {
                ImageWrapper<uchar1> outputWrapperGrayscale(output);
                Kernels::Device::rgb_or_bgr_to_grayscale<T>
                    <<<gridSize, blockSize, 0, stream>>>(inputWrapper, outputWrapperGrayscale, index);
                break;
            }
            default:
                break;
        }
    } else if (device == eDeviceType::CPU) {
        switch (conversionCode) {
            case COLOR_RGB2YUV:
            case COLOR_BGR2YUV:
                Kernels::Host::rgb_or_bgr_to_yuv<T>(inputWrapper, outputWrapper, index, delta);
                break;
            case COLOR_YUV2RGB:
            case COLOR_YUV2BGR:
                Kernels::Host::yuv_to_rgb_or_bgr<T>(inputWrapper, outputWrapper, index, delta);
                break;
            case COLOR_RGB2BGR:
            case COLOR_BGR2RGB:
                Kernels::Host::rgb_or_bgr_to_bgr_or_rgb<T>(inputWrapper, outputWrapper, index, delta);
                break;
            case COLOR_RGB2GRAY:
            case COLOR_BGR2GRAY: {
                ImageWrapper<uchar1> outputWrapperGrayscale(output);
                Kernels::Host::rgb_or_bgr_to_grayscale<T>(inputWrapper, outputWrapperGrayscale, index);
                break;
            }
            default:
                break;
        }
    }
}

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
    // check valid
    CHECK_TENSOR_COMPARISON(input.shape().layout() == output.shape().layout());
    CHECK_TENSOR_COMPARISON(o_batch == i_batch);
    CHECK_TENSOR_COMPARISON(o_batch > 0);
    CHECK_TENSOR_COMPARISON(o_channels > 0 && o_channels < 5);
    CHECK_TENSOR_COMPARISON(o_height > 0);
    CHECK_TENSOR_COMPARISON(o_width > 0);
    CHECK_TENSOR_COMPARISON(i_height > 0);
    CHECK_TENSOR_COMPARISON(i_width > 0);
    if (o_channels != i_channels)  // allowed only in gray
        CHECK_TENSOR_COMPARISON((conversionCode == COLOR_RGB2GRAY) || (conversionCode == COLOR_BGR2GRAY));

    // Select kernel dispatcher based on conversionCode
    // clang-format off
    using ColorConvFn = void(*)(hipStream_t, const Tensor&, const Tensor&, int64_t, int64_t, int64_t, int, float, const eColorConversionCode, const eDeviceType);
    using FuncEntry = std::tuple<ColorConvFn, int, float>;
    static const std::unordered_map<eColorConversionCode, FuncEntry> funcs = {
        { COLOR_RGB2YUV , {dispatch_cvt_color<uchar3>, 0, 128.0f}},
        { COLOR_BGR2YUV , {dispatch_cvt_color<uchar3>, 2, 128.0f}},
        { COLOR_YUV2RGB , {dispatch_cvt_color<uchar3>, 0, 128.0f}},
        { COLOR_YUV2BGR , {dispatch_cvt_color<uchar3>, 2, 128.0f}},
        { COLOR_RGB2BGR , {dispatch_cvt_color<uchar3>, 0,   2.0f}},
        { COLOR_BGR2RGB , {dispatch_cvt_color<uchar3>, 2,   0.0f}},
        { COLOR_RGB2GRAY, {dispatch_cvt_color<uchar3>, 0,   0.0f}},
        { COLOR_BGR2GRAY, {dispatch_cvt_color<uchar3>, 2,   0.0f}}
    };
    // clang-format on
    auto [func, orderIdx, delta] = funcs.at(conversionCode);
    func(stream, input, output, width, height, batch_size, orderIdx, delta, conversionCode, device);
}

}  // namespace roccv