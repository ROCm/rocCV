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

#include "op_gamma_contrast.hpp"

#include <hip/hip_runtime.h>

#include <functional>

#include <algorithm>
#include <cstring>
#include <iostream>
#include <vector>

#include "common/array_wrapper.hpp"
#include "common/math_vector.hpp"
#include "common/strided_data_wrap.hpp"
#include "core/wrappers/image_wrapper.hpp"
#include "common/validation_helpers.hpp"
#include "core/tensor.hpp"
#include "kernels/device/gamma_contrast_device.hpp"
#include "kernels/host/gamma_contrast_host.hpp"

namespace roccv {

template <typename T>
void dispatch_gamma_contrast_dtype(hipStream_t stream, const Tensor &input, const Tensor &output,
                                    float gamma, eDeviceType device) {
    ImageWrapper<T> inputWrapper(input);
    ImageWrapper<T> outputWrapper(output);

    if (device == eDeviceType::GPU) {
        dim3 block(64, 16);
        dim3 grid((outputWrapper.width() + block.x - 1) / block.x, (outputWrapper.height() + block.y - 1) / block.y, outputWrapper.batches());
       
        Kernels::Device::gamma_contrast<<<grid, block, 0, stream>>>(inputWrapper, outputWrapper, gamma);
    }
    else if (device == eDeviceType::CPU) {
        Kernels::Host::gamma_contrast(inputWrapper, outputWrapper, gamma);
    }
}

void GammaContrast::operator()(hipStream_t stream, const Tensor &input, const Tensor &output,
                               float gamma, eDeviceType device) {
    CHECK_TENSOR_DEVICE(input, device);
    CHECK_TENSOR_DEVICE(output, device);
    
    CHECK_TENSOR_DATATYPES(input, eDataType::DATA_TYPE_U8, eDataType::DATA_TYPE_U16, eDataType::DATA_TYPE_U32, eDataType::DATA_TYPE_F32);
    CHECK_TENSOR_DATATYPES(output, eDataType::DATA_TYPE_U8, eDataType::DATA_TYPE_U16, eDataType::DATA_TYPE_U32, eDataType::DATA_TYPE_F32);
    
    CHECK_TENSOR_LAYOUT(input, eTensorLayout::TENSOR_LAYOUT_NHWC, eTensorLayout::TENSOR_LAYOUT_HWC);
    CHECK_TENSOR_LAYOUT(output, eTensorLayout::TENSOR_LAYOUT_NHWC, eTensorLayout::TENSOR_LAYOUT_HWC);
    
    CHECK_TENSOR_CHANNELS(input, 1, 3, 4);

    CHECK_TENSOR_COMPARISON(input.layout() == output.layout());
    CHECK_TENSOR_COMPARISON(input.shape() == output.shape());
    CHECK_TENSOR_COMPARISON(input.dtype() == output.dtype());
    CHECK_TENSOR_COMPARISON(output.shape(output.layout().channels_index()) == input.shape(input.layout().channels_index()));
    CHECK_TENSOR_COMPARISON(output.shape(output.layout().width_index()) == input.shape(input.layout().width_index()));
    CHECK_TENSOR_COMPARISON(output.shape(output.layout().height_index()) == input.shape(input.layout().height_index()));
    CHECK_TENSOR_COMPARISON(output.shape(output.layout().batch_index()) == input.shape(input.layout().batch_index()));

    // Select kernel dispatcher based on number of channels and a base datatype.
    // clang-format off
    static const std::unordered_map<
    eDataType, std::array<std::function<void(hipStream_t, const Tensor &, const Tensor &, float, const eDeviceType)>, 4>>
        funcs = 
        {
            {eDataType::DATA_TYPE_U8, {dispatch_gamma_contrast_dtype<uchar1>, 0, dispatch_gamma_contrast_dtype<uchar3>, dispatch_gamma_contrast_dtype<uchar4>}},
            {eDataType::DATA_TYPE_U16, {dispatch_gamma_contrast_dtype<ushort1>, 0, dispatch_gamma_contrast_dtype<ushort3>, dispatch_gamma_contrast_dtype<ushort4>}},
            {eDataType::DATA_TYPE_U32, {dispatch_gamma_contrast_dtype<uint1>, 0, dispatch_gamma_contrast_dtype<uint3>, dispatch_gamma_contrast_dtype<uint4>}},
            {eDataType::DATA_TYPE_F32, {dispatch_gamma_contrast_dtype<float1>, 0, dispatch_gamma_contrast_dtype<float3>, dispatch_gamma_contrast_dtype<float4>}}
        };
    // clang-format on

    auto func = funcs.at(input.dtype().etype())[input.shape(input.layout().channels_index()) - 1];
    func(stream, input, output, gamma, device);

}
}  // namespace roccv