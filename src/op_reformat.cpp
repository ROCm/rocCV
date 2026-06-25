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
#include "operator_types.h"

namespace roccv {

namespace {
template <int Channels, typename T>
void DispatchReformatChannels(hipStream_t stream, const Tensor& input, const Tensor& output, eDeviceType device) {
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

        default: {
            throw Exception("Unsupported device type for Reformat operation.", eStatusType::INVALID_OPERATION);
            break;
        }
    }
}

template <typename T>
void DispatchReformatType(hipStream_t stream, const Tensor& input, const Tensor& output, eDeviceType device) {
    // clang-format off
    static const std::array<std::function<void(hipStream_t stream, const Tensor& input, const Tensor& output, eDeviceType device)>, 4> funcs = {
        DispatchReformatChannels<1, T>,
        nullptr,    // Not supported
        DispatchReformatChannels<3, T>,
        DispatchReformatChannels<4, T>,
    };
    // clang-format on

    auto func = funcs.at(input.shape("C") - 1);
    if (func == nullptr) {
        throw Exception("Unsupported channel count for Reformat operation.", eStatusType::INVALID_OPERATION);
    }

    func(stream, input, output, device);
}
}  // namespace

void Reformat::operator()(hipStream_t stream, const Tensor& input, const Tensor& output,
                          eDeviceType device) const {
    // clang-format off

    // Validate the input and output tensors
    CHECK_TENSOR_DEVICE(input, device);
    CHECK_TENSOR_DEVICE(output, device);
    CHECK_TENSOR_DATATYPES(input, eDataType::DATA_TYPE_U8, eDataType::DATA_TYPE_S8, eDataType::DATA_TYPE_U16, eDataType::DATA_TYPE_S16, eDataType::DATA_TYPE_U32, eDataType::DATA_TYPE_S32, eDataType::DATA_TYPE_F32, eDataType::DATA_TYPE_F64);
    CHECK_TENSOR_DATATYPES(output, eDataType::DATA_TYPE_U8, eDataType::DATA_TYPE_S8, eDataType::DATA_TYPE_U16, eDataType::DATA_TYPE_S16, eDataType::DATA_TYPE_U32, eDataType::DATA_TYPE_S32, eDataType::DATA_TYPE_F32, eDataType::DATA_TYPE_F64);
    CHECK_TENSOR_LAYOUT(input, eTensorLayout::TENSOR_LAYOUT_NHWC, eTensorLayout::TENSOR_LAYOUT_NCHW, eTensorLayout::TENSOR_LAYOUT_HWC, eTensorLayout::TENSOR_LAYOUT_CHW);
    CHECK_TENSOR_LAYOUT(output, eTensorLayout::TENSOR_LAYOUT_NHWC, eTensorLayout::TENSOR_LAYOUT_NCHW, eTensorLayout::TENSOR_LAYOUT_HWC, eTensorLayout::TENSOR_LAYOUT_CHW);
    CHECK_TENSOR_CHANNELS(input, 1, 3, 4);
    CHECK_TENSOR_CHANNELS(output, 1, 3, 4);

    CHECK_TENSOR_COMPARISON(input.dtype() == output.dtype());

    const TensorShape inputShape = input.shape();
    const TensorShape outputShape = output.shape();

    // Validate the input and output shapes
    const int inputBatchSize = inputShape.containsDim("N") ? inputShape["N"] : 1;
    const int outputBatchSize = outputShape.containsDim("N") ? outputShape["N"] : 1;
    
    CHECK_TENSOR_COMPARISON(inputBatchSize == outputBatchSize);
    CHECK_TENSOR_COMPARISON(inputShape["C"] == outputShape["C"]);
    CHECK_TENSOR_COMPARISON(inputShape["W"] == outputShape["W"]);
    CHECK_TENSOR_COMPARISON(inputShape["H"] == outputShape["H"]);

    if (needsInt64Wrapper(input) || needsInt64Wrapper(output)) {
        throw Exception("Input or output tensor is too large for int32 indexing", eStatusType::INVALID_OPERATION);
    }

    // Select kernel dispatcher based on the input and output datatypes.
    static const std::unordered_map<eDataType, std::function<void(hipStream_t stream, const Tensor& input, const Tensor& output,
                                                     eDeviceType device)>>
        funcs = {
            {eDataType::DATA_TYPE_U8,  DispatchReformatType<unsigned char>},
            {eDataType::DATA_TYPE_S8,  DispatchReformatType<signed char>},
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
        throw Exception("Unsupported datatype for Reformat operation.", eStatusType::INVALID_OPERATION);
    }

    func(stream, input, output, device);
}
}  // namespace roccv