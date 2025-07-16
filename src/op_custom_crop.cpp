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
#include "op_custom_crop.hpp"

#include <hip/hip_runtime.h>

#include <functional>

#include "common/strided_data_wrap.hpp"
#include "common/validation_helpers.hpp"
#include "core/wrappers/image_wrapper.hpp"
#include "kernels/device/custom_crop_device.hpp"
#include "kernels/host/custom_crop_host.hpp"

namespace roccv {

template <typename T>
void dispatch_custom_crop_dtype(hipStream_t stream, const Tensor& input, const Tensor& output, const Box_t cropRect, const eDeviceType device) {
    ImageWrapper<T> inputWrapper(input);
    ImageWrapper<T> outputWrapper(output);

    switch (device) {
        case eDeviceType::GPU: {
            dim3 block(64, 16);
            dim3 grid((outputWrapper.width() + block.x - 1) / block.x, (outputWrapper.height() + block.y - 1) / block.y,
                      outputWrapper.batches());
            Kernels::Device::custom_crop<<<grid, block, 0, stream>>>(inputWrapper, outputWrapper, cropRect);
            break;
        }

        case eDeviceType::CPU: {
            Kernels::Host::custom_crop(inputWrapper, outputWrapper, cropRect);
            break;
        }
    }
}

void CustomCrop::operator()(hipStream_t stream, const Tensor& input, const Tensor& output, const Box_t cropRect,
                            const eDeviceType device) const {
    CHECK_TENSOR_DEVICE(input, device);
    CHECK_TENSOR_LAYOUT(input, TENSOR_LAYOUT_HWC, TENSOR_LAYOUT_NHWC);
    CHECK_TENSOR_DATATYPES(input, DATA_TYPE_U8, DATA_TYPE_S8, DATA_TYPE_U16, DATA_TYPE_S16, DATA_TYPE_U32, DATA_TYPE_S32, DATA_TYPE_F32, DATA_TYPE_F64);
    CHECK_TENSOR_CHANNELS(input, 1, 3, 4);

    size_t batchSize = input.shape(input.layout().batch_index());
    size_t channels = input.shape(input.layout().channels_index());

    CHECK_TENSOR_DEVICE(output, device);
    CHECK_TENSOR_COMPARISON(input.layout() == output.layout());
    CHECK_TENSOR_COMPARISON(input.dtype() == output.dtype());
    CHECK_TENSOR_COMPARISON(output.shape(output.layout().channels_index()) == channels);
    CHECK_TENSOR_COMPARISON(output.shape(output.layout().batch_index()) == batchSize);
    CHECK_TENSOR_COMPARISON(output.shape(output.layout().width_index()) <= input.shape(input.layout().width_index()));
    CHECK_TENSOR_COMPARISON(output.shape(output.layout().height_index()) <= input.shape(input.layout().height_index()));
    CHECK_TENSOR_COMPARISON(output.shape(output.layout().width_index()) == cropRect.width);
    CHECK_TENSOR_COMPARISON(output.shape(output.layout().height_index()) == cropRect.height);
    CHECK_TENSOR_COMPARISON(input.shape(input.layout().width_index()) >= (cropRect.x + cropRect.width));
    CHECK_TENSOR_COMPARISON(input.shape(input.layout().height_index()) >= (cropRect.y + cropRect.height));

    // Select kernel dispatcher based on number of channels and a base datatype.
    // clang-format off
    static const std::unordered_map<
    eDataType, std::array<std::function<void(hipStream_t, const Tensor &, const Tensor &, const Box_t, const eDeviceType)>, 4>>
        funcs = 
        {
            {eDataType::DATA_TYPE_U8, {dispatch_custom_crop_dtype<uchar1>, 0, dispatch_custom_crop_dtype<uchar3>, dispatch_custom_crop_dtype<uchar4>}},
            {eDataType::DATA_TYPE_S8, {dispatch_custom_crop_dtype<char1>, 0, dispatch_custom_crop_dtype<char3>, dispatch_custom_crop_dtype<char4>}},
            {eDataType::DATA_TYPE_U16, {dispatch_custom_crop_dtype<ushort1>, 0, dispatch_custom_crop_dtype<ushort3>, dispatch_custom_crop_dtype<ushort4>}},
            {eDataType::DATA_TYPE_S16, {dispatch_custom_crop_dtype<short1>, 0, dispatch_custom_crop_dtype<short3>, dispatch_custom_crop_dtype<short4>}},
            {eDataType::DATA_TYPE_U32, {dispatch_custom_crop_dtype<uint1>, 0, dispatch_custom_crop_dtype<uint3>, dispatch_custom_crop_dtype<uint4>}},
            {eDataType::DATA_TYPE_S32, {dispatch_custom_crop_dtype<int1>, 0, dispatch_custom_crop_dtype<int3>, dispatch_custom_crop_dtype<int4>}},
            {eDataType::DATA_TYPE_F32, {dispatch_custom_crop_dtype<float1>, 0, dispatch_custom_crop_dtype<float3>, dispatch_custom_crop_dtype<float4>}},
            {eDataType::DATA_TYPE_F64, {dispatch_custom_crop_dtype<double1>, 0, dispatch_custom_crop_dtype<double3>, dispatch_custom_crop_dtype<double4>}}
        };
    // clang-format on

    auto func = funcs.at(input.dtype().etype())[input.shape(input.layout().channels_index()) - 1];
    if (func == 0)
        throw Exception("Not mapped to a defined function.", eStatusType::INVALID_OPERATION);
    func(stream, input, output, cropRect, device);
}
}  // namespace roccv