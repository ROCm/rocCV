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
#include "op_warp_perspective.hpp"

#include <functional>

#include "common/array_wrapper.hpp"
#include "common/validation_helpers.hpp"
#include "core/detail/casting.hpp"
#include "core/detail/math/math.hpp"
#include "core/detail/type_traits.hpp"
#include "kernels/device/warp_perspective_device.hpp"
#include "kernels/host/warp_perspective_host.hpp"

namespace roccv {
template <typename T, eBorderType B, eInterpolationType I>
void dispatch_warp_perspective_interp(hipStream_t stream, const Tensor &input, const Tensor &output,
                                      const PerspectiveTransform transMatrix, const T borderValue,
                                      const eDeviceType device) {
    ArrayWrapper<float, 9> transform(transMatrix);
    ImageWrapper<T> outputWrapper(output);
    InterpolationWrapper<T, B, I> inputWrapper(input, borderValue);

    // Launch CPU/GPU kernel depending on requested device type.
    switch (device) {
        case eDeviceType::GPU: {
            dim3 block(64, 16);
            dim3 grid((outputWrapper.width() + block.x - 1) / block.x, (outputWrapper.height() + block.y - 1) / block.y,
                      outputWrapper.batches());
            Kernels::Device::warp_perspective<<<grid, block, 0, stream>>>(inputWrapper, outputWrapper, transform);
            break;
        }

        case eDeviceType::CPU: {
            Kernels::Host::warp_perspective(inputWrapper, outputWrapper, transform);
            break;
        }
    }
}

template <typename T, eBorderType B>
void dispatch_warp_perspective_border_mode(hipStream_t stream, const Tensor &input, const Tensor &output,
                                           const PerspectiveTransform transMatrix,
                                           const eInterpolationType interpolation, const T borderValue,
                                           const eDeviceType device) {
    // Select kernel dispatcher based on selected interpolation mode.
    // clang-format off
    static const std::unordered_map<eInterpolationType, std::function<void(hipStream_t stream, const Tensor &, const Tensor &, const PerspectiveTransform, const T, const eDeviceType)>>
        funcs = {
            {eInterpolationType::INTERP_TYPE_NEAREST,   dispatch_warp_perspective_interp<T, B, eInterpolationType::INTERP_TYPE_NEAREST>},
            {eInterpolationType::INTERP_TYPE_LINEAR,    dispatch_warp_perspective_interp<T, B, eInterpolationType::INTERP_TYPE_LINEAR>}
        };  // clang-format on

    auto func = funcs.at(interpolation);
    if (func == 0) throw Exception("Not mapped to a defined function.", eStatusType::INVALID_OPERATION);
    func(stream, input, output, transMatrix, borderValue, device);
}

template <typename T>
void dispatch_warp_perspective_dtype(hipStream_t stream, const Tensor &input, const Tensor &output,
                                     const PerspectiveTransform transMatrix, const eInterpolationType interpolation,
                                     const eBorderType borderType, const float4 borderValue, const eDeviceType device) {
    // Select kernel dispatcher based on requested border mode.
    // clang-format off
    static const std::unordered_map<eBorderType, std::function<void(hipStream_t, const Tensor&, const Tensor&, const PerspectiveTransform, const eInterpolationType, T, const eDeviceType)>>
        funcs = {
            {eBorderType::BORDER_TYPE_CONSTANT,     dispatch_warp_perspective_border_mode<T, eBorderType::BORDER_TYPE_CONSTANT>},
            {eBorderType::BORDER_TYPE_REPLICATE,    dispatch_warp_perspective_border_mode<T, eBorderType::BORDER_TYPE_REPLICATE>},
            {eBorderType::BORDER_TYPE_REFLECT,      dispatch_warp_perspective_border_mode<T, eBorderType::BORDER_TYPE_REFLECT>},
            {eBorderType::BORDER_TYPE_WRAP,         dispatch_warp_perspective_border_mode<T, eBorderType::BORDER_TYPE_WRAP>}
        };
    // clang-format on

    auto func = funcs.at(borderType);
    if (func == 0) throw Exception("Not mapped to a defined function.", eStatusType::INVALID_OPERATION);
    func(stream, input, output, transMatrix, interpolation, detail::RangeCast<T>(borderValue), device);
}

void WarpPerspective::operator()(hipStream_t stream, const Tensor &input, const Tensor &output,
                                 const PerspectiveTransform transMatrix, bool isInverted,
                                 const eInterpolationType interpolation, const eBorderType borderType,
                                 const float4 borderValue, const eDeviceType device) const {
    // Validate input tensor
    CHECK_TENSOR_DEVICE(input, device);
    CHECK_TENSOR_DATATYPES(input, DATA_TYPE_U8, DATA_TYPE_F32);
    CHECK_TENSOR_LAYOUT(input, TENSOR_LAYOUT_HWC, TENSOR_LAYOUT_NHWC);
    CHECK_TENSOR_CHANNELS(input, 1, 3, 4);

    eDataType dtype = input.dtype().etype();
    int64_t channels = input.shape(input.layout().channels_index());

    // Validate output tensor
    CHECK_TENSOR_COMPARISON(input.device() == output.device());
    CHECK_TENSOR_COMPARISON(output.shape(output.layout().channels_index()) == channels);
    CHECK_TENSOR_COMPARISON(output.dtype() == input.dtype());
    CHECK_TENSOR_COMPARISON(output.layout() == input.layout());
    if (output.layout().batch_index() != -1) {
        CHECK_TENSOR_COMPARISON(output.shape(output.layout().batch_index()) ==
                                input.shape(input.layout().batch_index()));
    }

    PerspectiveTransform invertedTransform;

    // Ensure the input perspective transform matrix is inverted before passing into the kernel.
    detail::math::Matrix<float, 3, 3> mat;
    mat.load(transMatrix);
    if (!isInverted) {
        detail::math::inv_inplace(mat);
    }
    mat.store(invertedTransform);

    // Select kernel dispatcher based on number of channels and a base datatype.
    // clang-format off
    static const std::unordered_map<eDataType, std::array<std::function<void(hipStream_t, const Tensor &, const Tensor &, const PerspectiveTransform, const eInterpolationType, const eBorderType, const float4, const eDeviceType)>, 4>>
        funcs = {
            {eDataType::DATA_TYPE_U8,  {dispatch_warp_perspective_dtype<uchar1>, 0, dispatch_warp_perspective_dtype<uchar3>, dispatch_warp_perspective_dtype<uchar4>}},
            {eDataType::DATA_TYPE_F32, {dispatch_warp_perspective_dtype<float1>, 0, dispatch_warp_perspective_dtype<float3>, dispatch_warp_perspective_dtype<float4>}}
        };
    // clang-format on

    auto func = funcs.at(dtype)[channels - 1];
    if (func == 0) throw Exception("Not mapped to a defined function.", eStatusType::INVALID_OPERATION);
    func(stream, input, output, invertedTransform, interpolation, borderType, borderValue, device);
}
}  // namespace roccv