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
#include "op_flip.hpp"

#include <hip/hip_runtime.h>

#include <functional>
#include <unordered_map>

#include "common/validation_helpers.hpp"
#include "core/exception.hpp"
#include "core/status_type.h"
#include "core/wrappers/image_wrapper.hpp"
#include "kernels/device/flip_device.hpp"
#include "kernels/host/flip_host.hpp"

namespace roccv {

template <typename T, eAxis FlipType>
void dispatch_flip_axis(hipStream_t stream, const Tensor& input, const Tensor& output, const eDeviceType device) {
    ImageWrapper<T> inputWrapper(input);
    ImageWrapper<T> outputWrapper(output);

    switch (device) {
        case eDeviceType::GPU: {
            dim3 block(64, 16);
            dim3 grid((outputWrapper.width() + block.x - 1) / block.x, (outputWrapper.height() + block.y - 1) / block.y,
                      outputWrapper.batches());
            Kernels::Device::flip<FlipType><<<grid, block, 0, stream>>>(inputWrapper, outputWrapper);
            break;
        }

        case eDeviceType::CPU: {
            Kernels::Host::flip<FlipType>(inputWrapper, outputWrapper);
            break;
        }
    }
}

template <typename T>
void dispatch_flip_dtype(hipStream_t stream, const Tensor& input, const Tensor& output, int32_t flipCode,
                         const eDeviceType device) {
    eAxis flipType;
    if (flipCode == 0) {
        flipType = eAxis::X;
    } else if (flipCode > 0) {
        flipType = eAxis::Y;
    } else {
        flipType = eAxis::BOTH;
    }

    // Dispatch proper kernel based on provided flip type.
    std::unordered_map<eAxis, std::function<void(hipStream_t stream, const Tensor& input, const Tensor& output,
                                                 const eDeviceType device)>>
        funcs = {{eAxis::X, dispatch_flip_axis<T, eAxis::X>},
                 {eAxis::Y, dispatch_flip_axis<T, eAxis::Y>},
                 {eAxis::BOTH, dispatch_flip_axis<T, eAxis::BOTH>}};

    auto func = funcs[flipType];
    if (func == 0)
        throw Exception("Not mapped to a defined function.", eStatusType::INVALID_OPERATION);
    func(stream, input, output, device);
}

void Flip::operator()(hipStream_t stream, const Tensor& input, const Tensor& output, int32_t flipCode,
                      const eDeviceType device) const {
    // Tensor validation
    CHECK_TENSOR_DEVICE(input, device);
    CHECK_TENSOR_DATATYPES(input, DATA_TYPE_U8, DATA_TYPE_S32, DATA_TYPE_F32);
    CHECK_TENSOR_CHANNELS(input, 1, 3, 4);
    CHECK_TENSOR_LAYOUT(input, TENSOR_LAYOUT_HWC, TENSOR_LAYOUT_NHWC);

    CHECK_TENSOR_DEVICE(output, device);
    CHECK_TENSOR_COMPARISON(input.layout() == output.layout());
    CHECK_TENSOR_COMPARISON(input.dtype() == output.dtype());
    CHECK_TENSOR_COMPARISON(input.shape() == output.shape());

    // clang-format off
    std::function<void(hipStream_t stream, const Tensor& input, const Tensor& output, int32_t flipCode,
        const eDeviceType device)> funcs[5][4] = {
            {dispatch_flip_dtype<uchar1>, 0, dispatch_flip_dtype<uchar3>, dispatch_flip_dtype<uchar4>},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {dispatch_flip_dtype<int1>, 0, dispatch_flip_dtype<int3>, dispatch_flip_dtype<int4>},
            {dispatch_flip_dtype<float1>, 0, dispatch_flip_dtype<float3>, dispatch_flip_dtype<float4>}
        };
    // clang-format on

    auto func = funcs[input.dtype().etype()][input.shape(input.layout().channels_index()) - 1];
    if (func == 0)
        throw Exception("Not mapped to a defined function.", eStatusType::INVALID_OPERATION);
    func(stream, input, output, flipCode, device);
}
}  // namespace roccv
