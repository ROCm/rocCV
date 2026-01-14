/**
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

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

#include "op_reformat.hpp"

#include <functional>
#include <unordered_map>

#include "common/validation_helpers.hpp"
#include "core/wrappers/image_wrapper.hpp"
#include "kernels/device/reformat_device.hpp"
#include "kernels/host/reformat_host.hpp"

namespace roccv {

namespace {
template <int Channels, typename T>
void DispatchReformatChannels(hipStream_t stream, const Tensor& input, const Tensor& output, const eDeviceType device) {
    ImageWrapper<T> inputWrap(input);
    ImageWrapper<T> outputWrap(output);

    switch (device) {
        case eDeviceType::GPU: {
            dim3 block(64, 16);
            dim3 grid((outputWrap.width() + block.x - 1) / block.x, (outputWrap.height() + block.y - 1) / block.y,
                      outputWrap.batches());
            Kernels::Device::reformat<Channels, T><<<grid, block, 0, stream>>>(inputWrap, outputWrap);
            break;
        }

        case eDeviceType::CPU: {
            Kernels::Host::reformat<Channels, T>(inputWrap, outputWrap);
            break;
        }
    }
}

template <typename T>
void DispatchReformatType(hipStream_t stream, const Tensor& input, const Tensor& output, const eDeviceType device) {
    // clang-format off
    static const std::array<std::function<void(hipStream_t stream, const Tensor& input, const Tensor& output, const eDeviceType device)>, 4> funcs = {
        DispatchReformatChannels<1, T>,
        nullptr,    // Not supported
        DispatchReformatChannels<3, T>,
        DispatchReformatChannels<4, T>,
    };
    // clang-format on

    auto func = funcs.at(input.shape(input.layout().channels_index()) - 1);
    if (func == nullptr) {
        throw Exception("Not mapped to a defined function.", eStatusType::INVALID_OPERATION);
    }

    func(stream, input, output, device);
}
}  // namespace

void Reformat::operator()(hipStream_t stream, const Tensor& input, const Tensor& output,
                          const eDeviceType device) const {
    // Validate the input and output tensors
    CHECK_TENSOR_DEVICE(input, device);
    CHECK_TENSOR_DEVICE(output, device);
    CHECK_TENSOR_DATATYPES(input, eDataType::DATA_TYPE_U8, eDataType::DATA_TYPE_S8, eDataType::DATA_TYPE_U16,
                           eDataType::DATA_TYPE_S16, eDataType::DATA_TYPE_U32, eDataType::DATA_TYPE_S32,
                           eDataType::DATA_TYPE_F32, eDataType::DATA_TYPE_F64);
    CHECK_TENSOR_DATATYPES(output, eDataType::DATA_TYPE_U8, eDataType::DATA_TYPE_S8, eDataType::DATA_TYPE_U16,
                           eDataType::DATA_TYPE_S16, eDataType::DATA_TYPE_U32, eDataType::DATA_TYPE_S32,
                           eDataType::DATA_TYPE_F32, eDataType::DATA_TYPE_F64);
    CHECK_TENSOR_LAYOUT(input, eTensorLayout::TENSOR_LAYOUT_NHWC, eTensorLayout::TENSOR_LAYOUT_NCHW,
                        eTensorLayout::TENSOR_LAYOUT_HWC);
    CHECK_TENSOR_LAYOUT(output, eTensorLayout::TENSOR_LAYOUT_NHWC, eTensorLayout::TENSOR_LAYOUT_NCHW,
                        eTensorLayout::TENSOR_LAYOUT_HWC);
    CHECK_TENSOR_CHANNELS(input, 1, 3, 4);
    CHECK_TENSOR_CHANNELS(output, 1, 3, 4);

    CHECK_TENSOR_COMPARISON(input.dtype() == output.dtype());
    CHECK_TENSOR_COMPARISON(input.layout() != output.layout());

    if (input.layout().batch_index() != -1 && output.layout().batch_index() != -1) {
        CHECK_TENSOR_COMPARISON(input.shape(input.layout().batch_index()) ==
                                output.shape(output.layout().batch_index()));
    }

    CHECK_TENSOR_COMPARISON(input.shape(input.layout().channels_index()) ==
                            output.shape(output.layout().channels_index()));
    CHECK_TENSOR_COMPARISON(input.shape(input.layout().width_index()) == output.shape(output.layout().width_index()));
    CHECK_TENSOR_COMPARISON(input.shape(input.layout().height_index()) == output.shape(output.layout().height_index()));

    // Select kernel dispatcher based on the input and output datatypes.
    // clang-format off
    static const std::unordered_map<eDataType, std::function<void(hipStream_t stream, const Tensor& input, const Tensor& output,
                                                     const eDeviceType device)>>
        funcs = {
            {eDataType::DATA_TYPE_U8, DispatchReformatType<unsigned char>},
            {eDataType::DATA_TYPE_S8, DispatchReformatType<signed char>},
            {eDataType::DATA_TYPE_U16, DispatchReformatType<unsigned short>},
            {eDataType::DATA_TYPE_S16, DispatchReformatType<signed short>},
            {eDataType::DATA_TYPE_U32, DispatchReformatType<unsigned int>},
            {eDataType::DATA_TYPE_S32, DispatchReformatType<signed int>},
            {eDataType::DATA_TYPE_F32, DispatchReformatType<float>},
            {eDataType::DATA_TYPE_F64, DispatchReformatType<double>},
        };
    // clang-format on

    auto func = funcs.at(input.dtype().etype());

    if (func == nullptr) {
        throw Exception("Not mapped to a defined function.", eStatusType::INVALID_OPERATION);
    }

    func(stream, input, output, device);
}
}  // namespace roccv