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
#include "op_convert_to.hpp"

#include <functional>

#include <hip/hip_runtime.h>
#include "core/wrappers/image_wrapper.hpp"
#include "common/validation_helpers.hpp"
#include "core/detail/casting.hpp"
#include "core/detail/type_traits.hpp"
#include "kernels/device/convert_to_device.hpp"
#include "kernels/host/convert_to_host.hpp"

namespace roccv {

template <typename SRC_DT, typename DST_DT, int NC>
void dispatch_convert_to_channels(hipStream_t stream, const Tensor &input, const Tensor &output,
                                       const double alpha, const double beta, const eDeviceType device) {
    
    using SRC_DT_NC = detail::MakeType<SRC_DT, NC>;
    using DST_DT_NC = detail::MakeType<DST_DT, NC>;

    ImageWrapper<SRC_DT_NC> inputWrapper(input);
    ImageWrapper<DST_DT_NC> outputWrapper(output);

    using SRC_BT = detail::BaseType<SRC_DT>;
    using DST_BT = detail::BaseType<DST_DT>;

    using DT_AB = decltype(float() * SRC_BT() * DST_BT());
    
    DT_AB alpha_ab = detail::SaturateCast<DT_AB>(alpha);
    DT_AB beta_ab = detail::SaturateCast<DT_AB>(beta);

    // Launch CPU/GPU kernel depending on requested device type.
    switch (device) {
        case eDeviceType::GPU: {
            dim3 block(64, 16);
            dim3 grid((outputWrapper.width() + block.x - 1) / block.x, (outputWrapper.height() + block.y - 1) / block.y,
                      outputWrapper.batches());
            Kernels::Device::convert_to<<<grid, block, 0, stream>>>(inputWrapper, outputWrapper, alpha_ab, beta_ab);
            break;
        }
        case eDeviceType::CPU: {
            Kernels::Host::convert_to(inputWrapper, outputWrapper, alpha_ab, beta_ab);
            break;
        }
    }
}

template <typename SRC_DT, typename DST_DT>
void dispatch_convert_to_output_dtype(hipStream_t stream, const Tensor &input, const Tensor &output,
                                       const double alpha, const double beta, const eDeviceType device) {

    int64_t channels = output.shape(output.layout().channels_index());
    // Select kernel dispatcher based on number of channels.
    // clang-format off
    static const std::array<std::function<void(hipStream_t, const Tensor &, const Tensor &, const double, const double, const eDeviceType)>, 4>
        funcs = {dispatch_convert_to_channels<SRC_DT, DST_DT, 1>, dispatch_convert_to_channels<SRC_DT, DST_DT, 2>, dispatch_convert_to_channels<SRC_DT, DST_DT, 3>, dispatch_convert_to_channels<SRC_DT, DST_DT, 4>};
        
            
    // clang-format on

    auto func = funcs.at(channels - 1);
    if (func == 0) throw Exception("Not mapped to a defined function.", eStatusType::INVALID_OPERATION);
    func(stream, input, output, alpha, beta, device);
}

template <typename SRC_DT>
void dispatch_convert_to_input_dtype(hipStream_t stream, const Tensor &input, const Tensor &output,
                                       const double alpha, const double beta, const eDeviceType device) {
    
    eDataType output_dtype = output.dtype().etype();
    
    // Select kernel dispatcher based on a base input datatype.
    // clang-format off
    static const std::unordered_map<eDataType, std::function<void(hipStream_t, const Tensor &, const Tensor &, const double, const double, const eDeviceType)>>
        funcs = {
            {eDataType::DATA_TYPE_U8, dispatch_convert_to_output_dtype<SRC_DT, uchar>},
            {eDataType::DATA_TYPE_S8,  dispatch_convert_to_output_dtype<SRC_DT, signed char>},
            {eDataType::DATA_TYPE_U16,  dispatch_convert_to_output_dtype<SRC_DT, ushort>},
            {eDataType::DATA_TYPE_S16,  dispatch_convert_to_output_dtype<SRC_DT, short>},
            {eDataType::DATA_TYPE_S32,  dispatch_convert_to_output_dtype<SRC_DT, int>},
            {eDataType::DATA_TYPE_F32, dispatch_convert_to_output_dtype<SRC_DT, float>},
            {eDataType::DATA_TYPE_F64, dispatch_convert_to_output_dtype<SRC_DT, double>}
        };
    // clang-format on
    // std make pair possibly needed
    auto func = funcs.at(output_dtype);
    if (func == 0) throw Exception("Not mapped to a defined function.", eStatusType::INVALID_OPERATION);
    func(stream, input, output, alpha, beta, device);

}

void ConvertTo::operator()(hipStream_t stream, const Tensor &input, const Tensor &output,
                                const double alpha, const double beta, const eDeviceType device) const {
    
    // Validate input tensor
    CHECK_TENSOR_DEVICE(input, device);
    CHECK_TENSOR_DATATYPES(input, DATA_TYPE_S8, DATA_TYPE_U8, DATA_TYPE_U16, DATA_TYPE_S16,
                           DATA_TYPE_S32, DATA_TYPE_F32, DATA_TYPE_F64);
    CHECK_TENSOR_DATATYPES(output, DATA_TYPE_S8, DATA_TYPE_U8, DATA_TYPE_U16, DATA_TYPE_S16,
                           DATA_TYPE_S32, DATA_TYPE_F32, DATA_TYPE_F64);
    CHECK_TENSOR_LAYOUT(input, TENSOR_LAYOUT_HWC, TENSOR_LAYOUT_NHWC);
    CHECK_TENSOR_CHANNELS(input, 1, 2, 3, 4);

    eDataType input_dtype = input.dtype().etype();
    int64_t channels = input.shape(input.layout().channels_index());

    // Validate output tensor
    CHECK_TENSOR_COMPARISON(input.device() == output.device());
    CHECK_TENSOR_COMPARISON(output.shape(output.layout().channels_index()) == channels);
    CHECK_TENSOR_COMPARISON(output.layout() == input.layout());
    if (output.layout().batch_index() != -1) {
        CHECK_TENSOR_COMPARISON(output.shape(output.layout().batch_index()) ==
                                input.shape(input.layout().batch_index()));
    }

    // Select kernel dispatcher based on a base input datatype.
    // clang-format off
    static const std::unordered_map<eDataType, std::function<void(hipStream_t, const Tensor &, const Tensor &, const double, const double, const eDeviceType)>>
        funcs = {
            {eDataType::DATA_TYPE_U8, dispatch_convert_to_input_dtype<uchar>},
            {eDataType::DATA_TYPE_S8,  dispatch_convert_to_input_dtype<signed char>},
            {eDataType::DATA_TYPE_U16,  dispatch_convert_to_input_dtype<ushort>},
            {eDataType::DATA_TYPE_S16,  dispatch_convert_to_input_dtype<short>},
            {eDataType::DATA_TYPE_S32,  dispatch_convert_to_input_dtype<int>},
            {eDataType::DATA_TYPE_F32, dispatch_convert_to_input_dtype<float>},
            {eDataType::DATA_TYPE_F64, dispatch_convert_to_input_dtype<double>}
        };
    // clang-format on
    // std make pair possibly needed
    auto func = funcs.at(input_dtype);
    if (func == 0) throw Exception("Not mapped to a defined function.", eStatusType::INVALID_OPERATION);
    func(stream, input, output, alpha, beta, device);
}
}   // namespace roccv