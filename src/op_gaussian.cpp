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
#include <hip/hip_runtime.h>

#include <functional>

#include "common/validation_helpers.hpp"
#include "core/detail/casting.hpp"
#include "core/wrappers/image_wrapper.hpp"
#include "op_thresholding.hpp"
// #include "kernels/device/gaussian_device.hpp"
// #include "kernels/host/gaussian_host.hpp"

namespace roccv {

// TODO: figure out if should group pair params

// skeleton for final dispatch layer for kernel launch
template <typename T>
void dispatch_gaussian(hipStream_t stream, const Tensor& input, const Tensor& output, eDeviceType device) {
    ImageWrapper<T> inputWrapper(input);
    ImageWrapper<T> outputWrapper(output);

    switch (device) {
        case eDeviceType::GPU: {
            dim3 block(64, 16);
            dim3 grid((outputWrapper.width() + block.x - 1) / block.x, (outputWrapper.height() + block.y - 1) / block.y,
                      outputWrapper.batches());
            Kernels::Device::gaussian<<<grid, block, 0, stream>>>(inputWrapper, outputWrapper);
            break;
        }

        case eDeviceType::CPU: {
            Kernels::Host::gaussian(inputWrapper, outputWrapper);
            break;
        }
    }
}

// skeleton
void Gaussian::operator()(hipStream_t stream, const Tensor& input, const Tensor& output, int kernelWidth,
                          int kernelHeight, double sigmaX, double sigmaY, eBorderType borderMode,
                          eDeviceType device) const {

    // Validate input tensor
    CHECK_TENSOR_DEVICE(input, device);
    CHECK_TENSOR_DATATYPES(input, DATA_TYPE_U8, DATA_TYPE_U16, DATA_TYPE_S16, DATA_TYPE_S32, DATA_TYPE_F32);
    CHECK_TENSOR_LAYOUT(input, TENSOR_LAYOUT_HWC, TENSOR_LAYOUT_NHWC);
    CHECK_TENSOR_CHANNELS(input, 1, 3, 4);

    // Validate output tensor
    CHECK_TENSOR_COMPARISON(input.dtype() == output.dtype());
    CHECK_TENSOR_COMPARISON(input.device() == output.device());
    CHECK_TENSOR_COMPARISON(input.shape() == output.shape());

    // clang-format off
    static const std::unordered_map<
    eDataType, std::array<std::function<void(hipStream_t stream, const Tensor& input, const Tensor& output,
        eDeviceType device)>, 4>>
        funcs = {
//{{DISPATCH_TABLE}}
        };
    // clang-format on
    auto func = funcs.at(input.dtype().etype())[input.shape(input.layout().channels_index()) - 1];
    if (func == 0) throw Exception("Not mapped to a defined function.", eStatusType::INVALID_OPERATION);
    func(stream, input, output, device);
}
}  // namespace roccv