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
#include "op_remap.hpp"

#include <functional>

#include "common/array_wrapper.hpp"
#include "core/wrappers/image_wrapper.hpp"
#include "core/wrappers/interpolation_wrapper.hpp"
#include "common/validation_helpers.hpp"
#include "core/detail/casting.hpp"
#include "core/detail/math/math.hpp"
#include "core/detail/type_traits.hpp"
#include "kernels/host/remap_host.hpp"
#include "kernels/device/remap_device.hpp"

namespace roccv {
Remap::Remap() {}

Remap::~Remap() {}

template <typename T, eBorderType B, eInterpolationType I, eInterpolationType M>
void dispatch_remap_mapInterp(hipStream_t stream, const Tensor &input, const Tensor &output, const Tensor &map,
                                        const eRemapType mapValueType, const bool alignCorners, 
                                        const T borderValue, const eDeviceType device) {
    ImageWrapper<T> outputWrapper(output);
    InterpolationWrapper<float2, B, M> wrappedMapTensor(map, make_float2(0,0));
    InterpolationWrapper<T, B, I> inputWrapper(input, borderValue);
    

    // Launch CPU/GPU kernel depending on requested device type.
    switch (device) {
        case eDeviceType::GPU: {
            dim3 block(64, 16);
            dim3 grid((outputWrapper.width() + block.x - 1) / block.x, (outputWrapper.height() + block.y - 1) / block.y,
                      outputWrapper.batches());
            Kernels::Device::remap<T><<<grid, block, 0, stream>>>(inputWrapper, outputWrapper, wrappedMapTensor);
            break;
        }

        case eDeviceType::CPU: {
            Kernels::Host::remap<T>(inputWrapper, outputWrapper, wrappedMapTensor);
            break;
        }
    }

}

template <typename T, eBorderType B, eInterpolationType I>
void dispatch_remap_interp(hipStream_t stream, const Tensor &input, const Tensor &output, const Tensor &map,
                                        const eInterpolationType mapInterpolation, const eRemapType mapValueType, const bool alignCorners, 
                                        const T borderValue, const eDeviceType device) {
    // Select kernel dispatcher based on selected interpolation mode.
    // clang-format off
    const std::function<void(hipStream_t stream, const Tensor&, const Tensor&, const Tensor&, const eRemapType, const bool, const T,
                             const eDeviceType)>
        funcs[3] = {
            dispatch_remap_mapInterp<T, B, I, eInterpolationType::INTERP_TYPE_NEAREST>,
            dispatch_remap_mapInterp<T, B, I, eInterpolationType::INTERP_TYPE_LINEAR>,
            0
        };  // clang-format on

    auto func = funcs[mapInterpolation];
    if (func == 0)
        throw Exception("Not mapped to a defined function.", eStatusType::INVALID_OPERATION);
    func(stream, input, output, map, mapValueType, alignCorners, borderValue, device);
}

template <typename T, eBorderType B>
void dispatch_remap_border_mode(hipStream_t stream, const Tensor &input, const Tensor &output, const Tensor &map,
                            const eInterpolationType inInterpolation, const eInterpolationType mapInterpolation, const eRemapType mapValueType,
                            const bool alignCorners, const T borderValue, const eDeviceType device) {
    // Select kernel dispatcher based on selected interpolation mode.
    // clang-format off
    const std::function<void(hipStream_t stream, const Tensor&, const Tensor&, const Tensor&, const eInterpolationType, const eRemapType, const bool, const T,
                             const eDeviceType)>
        funcs[3] = {
            dispatch_remap_interp<T, B, eInterpolationType::INTERP_TYPE_NEAREST>,
            dispatch_remap_interp<T, B, eInterpolationType::INTERP_TYPE_LINEAR>,
            0
        };  // clang-format on

    auto func = funcs[inInterpolation];
    if (func == 0)
        throw Exception("Not mapped to a defined function.", eStatusType::INVALID_OPERATION);
    func(stream, input, output, map, mapInterpolation, mapValueType, alignCorners, borderValue, device);
}

template <typename T>
void dispatch_remap_dtype(hipStream_t stream, const Tensor &input, const Tensor &output, const Tensor &map,
                            const eInterpolationType inInterpolation, const eInterpolationType mapInterpolation, const eRemapType mapValueType,
                            const bool alignCorners, const eBorderType borderType, const float4 borderValue, const eDeviceType device) {
    // Select kernel dispatcher based on requested border mode.
    // clang-format off
    const std::function<void(hipStream_t, const Tensor&, const Tensor&, const Tensor&, const eInterpolationType, const eInterpolationType, const eRemapType, const bool, T, const eDeviceType)>
        funcs[4] = {
            dispatch_remap_border_mode<T, eBorderType::BORDER_TYPE_CONSTANT>,
            dispatch_remap_border_mode<T, eBorderType::BORDER_TYPE_REPLICATE>,
            dispatch_remap_border_mode<T, eBorderType::BORDER_TYPE_REFLECT>,
            dispatch_remap_border_mode<T, eBorderType::BORDER_TYPE_WRAP>
        };
    // clang-format on

    auto func = funcs[borderType];
    if (func == 0)
        throw Exception("Not mapped to a defined function.", eStatusType::INVALID_OPERATION);
    func(stream, input, output, map, inInterpolation, mapInterpolation, mapValueType, alignCorners, detail::RangeCast<T>(borderValue), device);
}

void Remap::operator()(hipStream_t stream, const Tensor &input, const Tensor &output, const Tensor &map, 
                        const eInterpolationType inInterpolation, const eInterpolationType mapInterpolation, const eRemapType mapValueType, 
                        const bool alignCorners, const eBorderType borderType, const float4 borderValue, eDeviceType device) {

    // Verify that the tensors are located on the right device (CPU or GPU).
    CHECK_TENSOR_DEVICE(input, device);
    CHECK_TENSOR_DEVICE(output, device);
    CHECK_TENSOR_DEVICE(map, device);

    // Ensure all tensors are using supported datatypes
    CHECK_TENSOR_DATATYPES(input, eDataType::DATA_TYPE_U8);
    CHECK_TENSOR_DATATYPES(output, eDataType::DATA_TYPE_U8);
    CHECK_TENSOR_DATATYPES(map, eDataType::DATA_TYPE_F32);

    // Ensure all tensors are using supported layouts.
    CHECK_TENSOR_LAYOUT(input, eTensorLayout::TENSOR_LAYOUT_NHWC, eTensorLayout::TENSOR_LAYOUT_HWC);
    CHECK_TENSOR_LAYOUT(output, eTensorLayout::TENSOR_LAYOUT_NHWC, eTensorLayout::TENSOR_LAYOUT_HWC);
    CHECK_TENSOR_LAYOUT(map, eTensorLayout::TENSOR_LAYOUT_NHWC, eTensorLayout::TENSOR_LAYOUT_HWC);                         

    // Ensure the layout and shapes for the input/output tensors match
    CHECK_TENSOR_COMPARISON(input.layout() == output.layout());
    CHECK_TENSOR_COMPARISON(input.shape() == output.shape());
    CHECK_TENSOR_CHANNELS(input, 1, 3, 4);

    eDataType dtype = input.dtype().etype();
    int64_t channels = input.shape(input.layout().channels_index());
    
    // Select kernel dispatcher based on number of channels and a base datatype.
    // clang-format off
    const std::function<void(hipStream_t, const Tensor &, const Tensor &, const Tensor &,
                             const eInterpolationType, const eInterpolationType, const eRemapType, 
                             const bool, const eBorderType, const float4, const eDeviceType)>
        funcs[1][4] = {
            {dispatch_remap_dtype<uchar1>, 0, dispatch_remap_dtype<uchar3>, dispatch_remap_dtype<uchar4>},
        };
    // clang-format on

    auto func = funcs[dtype][channels - 1];
    if (func == 0)
        throw Exception("Not mapped to a defined function.", eStatusType::INVALID_OPERATION);
    func(stream, input, output, map, inInterpolation, mapInterpolation, mapValueType, alignCorners, borderType, borderValue, device);
}
}