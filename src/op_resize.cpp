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
#include "op_resize.hpp"

#include <functional>
#include <unordered_map>

#include "common/validation_helpers.hpp"
#include "core/detail/casting.hpp"
#include "core/exception.hpp"
#include "core/status_type.h"
#include "core/wrappers/interpolation_wrapper.hpp"
#include "kernels/device/resize_device.hpp"
#include "kernels/host/resize_host.hpp"

namespace roccv {

template <typename T, eInterpolationType I>
void dispatch_resize_interp(hipStream_t stream, const Tensor& input, const Tensor& output, const eDeviceType device) {
    ImageWrapper<T> outputWrapper(output);
    // Interpolation wrapper uses a constant border mode with all black as the constant value.
    T borderValue = detail::RangeCast<T>(make_float4(0, 0, 0, 1.0f));
    InterpolationWrapper<T, eBorderType::BORDER_TYPE_CONSTANT, I> inputWrapper(input, borderValue);

    float scaleX = inputWrapper.width() / static_cast<float>(outputWrapper.width());
    float scaleY = inputWrapper.height() / static_cast<float>(outputWrapper.height());

    switch (device) {
        case eDeviceType::GPU: {
            dim3 block(64, 16);
            dim3 grid((outputWrapper.width() + block.x - 1) / block.x, (outputWrapper.height() + block.y - 1) / block.y,
                      outputWrapper.batches());
            Kernels::Device::resize<<<grid, block, 0, stream>>>(inputWrapper, outputWrapper, scaleX, scaleY);
            break;
        }

        case eDeviceType::CPU: {
            Kernels::Host::resize(inputWrapper, outputWrapper, scaleX, scaleY);
            break;
        }
    }
}

template <typename T>
void dispatch_resize_dtype(hipStream_t stream, const Tensor& input, const Tensor& output,
                           const eInterpolationType interpolation, const eDeviceType device) {
    static const std::unordered_map<
        eInterpolationType,
        std::function<void(hipStream_t stream, const Tensor& input, const Tensor& output, const eDeviceType device)>>
        funcs = {{eInterpolationType::INTERP_TYPE_NEAREST,
                  dispatch_resize_interp<T, eInterpolationType::INTERP_TYPE_NEAREST>},
                 {eInterpolationType::INTERP_TYPE_LINEAR,
                  dispatch_resize_interp<T, eInterpolationType::INTERP_TYPE_LINEAR>}};

    auto func = funcs.at(interpolation);
    if (func == 0)
        throw Exception("Not mapped to a defined function.", eStatusType::INVALID_OPERATION);
    func(stream, input, output, device);
}

void Resize::operator()(hipStream_t stream, const Tensor& input, const Tensor& output,
                        const eInterpolationType interpolation, const eDeviceType device) const {
    CHECK_TENSOR_DEVICE(input, device);
    CHECK_TENSOR_DEVICE(output, device);

    CHECK_TENSOR_CHANNELS(input, 1, 3, 4);
    CHECK_TENSOR_DATATYPES(input, DATA_TYPE_U8, DATA_TYPE_F32);
    CHECK_TENSOR_LAYOUT(input, TENSOR_LAYOUT_HWC, TENSOR_LAYOUT_NHWC);

    CHECK_TENSOR_COMPARISON(input.layout() == output.layout());
    CHECK_TENSOR_COMPARISON(input.dtype() == output.dtype());
    CHECK_TENSOR_COMPARISON(input.shape(input.layout().channels_index()) ==
                             output.shape(output.layout().channels_index()));
    if (input.layout().batch_index() != -1) {
        CHECK_TENSOR_COMPARISON(input.shape(input.layout().batch_index()) ==
                                 output.shape(output.layout().batch_index()));
    }

    // clang-format off
    static const std::unordered_map<eDataType, std::array<std::function<void(hipStream_t stream, const Tensor& input, const Tensor& output,
                       const eInterpolationType interpolation, const eDeviceType device)>, 4>>
        funcs = {
            {eDataType::DATA_TYPE_U8, {dispatch_resize_dtype<uchar1>, 0, dispatch_resize_dtype<uchar3>, dispatch_resize_dtype<uchar4>}},
            {eDataType::DATA_TYPE_F32, {dispatch_resize_dtype<float1>, 0, dispatch_resize_dtype<float3>, dispatch_resize_dtype<float4>}}
        };
    // clang-format on

    auto func = funcs.at(input.dtype().etype())[input.shape(input.layout().channels_index()) - 1];
    if (func == 0)
        throw Exception("Not mapped to a defined function.", eStatusType::INVALID_OPERATION);
    func(stream, input, output, interpolation, device);
}
}  // namespace roccv
