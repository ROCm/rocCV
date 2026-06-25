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
#include "op_warp_affine.hpp"

#include <functional>

#include "common/array_wrapper.hpp"
#include "common/validation_helpers.hpp"
#include "core/detail/casting.hpp"
#include "core/detail/math/math.hpp"
#include "kernels/device/warp_affine_device.hpp"
#include "kernels/host/warp_affine_host.hpp"
#include "operator_types.h"

namespace roccv {
template <typename T, eBorderType B, eInterpolationType I>
void dispatch_warp_affine_interp(hipStream_t stream, const Tensor &input, const Tensor &output,
                                 const AffineTransform affineInv, T borderValue, eDeviceType device) {
    ArrayWrapper<float, 6> transform(affineInv);
    ImageWrapper<T> outputWrapper(output);
    InterpolationWrapper<T, B, I> inputWrapper(input, borderValue);

    switch (device) {
        case eDeviceType::GPU: {
            dim3 block(64, 16);
            dim3 grid((outputWrapper.width() + block.x - 1) / block.x, (outputWrapper.height() + block.y - 1) / block.y,
                      outputWrapper.batches());
            Kernels::Device::warp_affine<<<grid, block, 0, stream>>>(inputWrapper, outputWrapper, transform);
            break;
        }

        case eDeviceType::CPU: {
            Kernels::Host::warp_affine(inputWrapper, outputWrapper, transform);
            break;
        }
    }
}

template <typename T, eBorderType B>
void dispatch_warp_affine_border_mode(hipStream_t stream, const Tensor &input, const Tensor &output,
                                      const AffineTransform affineInv, eInterpolationType interpolation, T borderValue,
                                      eDeviceType device) {
    // clang-format off
    static const std::unordered_map<eInterpolationType, std::function<void(hipStream_t stream, const Tensor &, const Tensor &, const AffineTransform, T, eDeviceType)>>
        funcs = {
            {eInterpolationType::INTERP_TYPE_NEAREST, dispatch_warp_affine_interp<T, B, eInterpolationType::INTERP_TYPE_NEAREST>},
            {eInterpolationType::INTERP_TYPE_LINEAR,  dispatch_warp_affine_interp<T, B, eInterpolationType::INTERP_TYPE_LINEAR>},
            {eInterpolationType::INTERP_TYPE_CUBIC,   dispatch_warp_affine_interp<T, B, eInterpolationType::INTERP_TYPE_CUBIC>}
        };
    // clang-format on

    if (!funcs.contains(interpolation)) {
        throw Exception("Operation does not support the given interpolation mode.", eStatusType::NOT_IMPLEMENTED);
    }

    auto func = funcs.at(interpolation);
    func(stream, input, output, affineInv, borderValue, device);
}

template <typename T>
void dispatch_warp_affine_dtype(hipStream_t stream, const Tensor &input, const Tensor &output,
                                const AffineTransform affineInv, eInterpolationType interpolation,
                                eBorderType borderType, float4 borderValue, eDeviceType device) {
    // clang-format off
    static const std::unordered_map<eBorderType, std::function<void(hipStream_t, const Tensor&, const Tensor&, const AffineTransform, eInterpolationType, T, eDeviceType)>>
        funcs = {
            {eBorderType::BORDER_TYPE_CONSTANT,   dispatch_warp_affine_border_mode<T, eBorderType::BORDER_TYPE_CONSTANT>},
            {eBorderType::BORDER_TYPE_REPLICATE,  dispatch_warp_affine_border_mode<T, eBorderType::BORDER_TYPE_REPLICATE>},
            {eBorderType::BORDER_TYPE_REFLECT,    dispatch_warp_affine_border_mode<T, eBorderType::BORDER_TYPE_REFLECT>},
            {eBorderType::BORDER_TYPE_REFLECT101, dispatch_warp_affine_border_mode<T, eBorderType::BORDER_TYPE_REFLECT101>},
            {eBorderType::BORDER_TYPE_WRAP,       dispatch_warp_affine_border_mode<T, eBorderType::BORDER_TYPE_WRAP>}
        };
    // clang-format on

    if (!funcs.contains(borderType)) {
        throw Exception("Operator does not support the given border mode.", eStatusType::NOT_IMPLEMENTED);
    }

    auto func = funcs.at(borderType);
    func(stream, input, output, affineInv, interpolation, detail::SaturateCast<T>(borderValue), device);
}

void WarpAffine::operator()(hipStream_t stream, const Tensor &input, const Tensor &output, const AffineTransform xform,
                            bool isInverted, eInterpolationType interp, eBorderType borderMode, float4 borderValue,
                            eDeviceType device) const {
    CHECK_TENSOR_DEVICE(input, device);
    CHECK_TENSOR_DATATYPES(input, DATA_TYPE_S8, DATA_TYPE_U8, DATA_TYPE_U16, DATA_TYPE_S16, DATA_TYPE_U32,
                           DATA_TYPE_S32, DATA_TYPE_F32, DATA_TYPE_F64);
    CHECK_TENSOR_LAYOUT(input, TENSOR_LAYOUT_HWC, TENSOR_LAYOUT_NHWC);
    CHECK_TENSOR_CHANNELS(input, 1, 3, 4);

    eDataType dtype = input.dtype().etype();
    int64_t channels = input.shape(input.layout().channels_index());

    CHECK_TENSOR_COMPARISON(input.device() == output.device());
    CHECK_TENSOR_COMPARISON(output.shape(output.layout().channels_index()) == channels);
    CHECK_TENSOR_COMPARISON(output.dtype() == input.dtype());
    CHECK_TENSOR_COMPARISON(output.layout() == input.layout());
    if (output.layout().batch_index() != -1) {
        CHECK_TENSOR_COMPARISON(output.shape(output.layout().batch_index()) ==
                                input.shape(input.layout().batch_index()));
    }

    if (needsInt64Wrapper(input) || needsInt64Wrapper(output)) {
        throw Exception("Input or output tensor is too large for int32 indexing", eStatusType::INVALID_OPERATION);
    }

    PerspectiveTransform full{};
#pragma unroll
    for (int i = 0; i < 6; i++) {
        full[i] = xform[i];
    }
    full[6] = 0.0f;
    full[7] = 0.0f;
    full[8] = 1.0f;

    detail::math::Matrix<float, 3, 3> mat;
    mat.load(full);
    if (!isInverted) {
        detail::math::inv_inplace(mat);
    }
    mat.store(full);

    AffineTransform affineInv{};
#pragma unroll
    for (int i = 0; i < 6; i++) {
        affineInv[i] = full[i];
    }

    // clang-format off
    static const std::unordered_map<eDataType, std::array<std::function<void(hipStream_t, const Tensor &, const Tensor &, const AffineTransform, eInterpolationType, eBorderType, float4, eDeviceType)>, 4>>
        funcs = {
            {eDataType::DATA_TYPE_U8,  {dispatch_warp_affine_dtype<uchar1>, 0, dispatch_warp_affine_dtype<uchar3>, dispatch_warp_affine_dtype<uchar4>}},
            {eDataType::DATA_TYPE_S8,  {dispatch_warp_affine_dtype<char1>, 0, dispatch_warp_affine_dtype<char3>, dispatch_warp_affine_dtype<char4>}},
            {eDataType::DATA_TYPE_U16,  {dispatch_warp_affine_dtype<ushort1>, 0, dispatch_warp_affine_dtype<ushort3>, dispatch_warp_affine_dtype<ushort4>}},
            {eDataType::DATA_TYPE_S16,  {dispatch_warp_affine_dtype<short1>, 0, dispatch_warp_affine_dtype<short3>, dispatch_warp_affine_dtype<short4>}},
            {eDataType::DATA_TYPE_U32,  {dispatch_warp_affine_dtype<uint1>, 0, dispatch_warp_affine_dtype<uint3>, dispatch_warp_affine_dtype<uint4>}},
            {eDataType::DATA_TYPE_S32,  {dispatch_warp_affine_dtype<int1>, 0, dispatch_warp_affine_dtype<int3>, dispatch_warp_affine_dtype<int4>}},
            {eDataType::DATA_TYPE_F32, {dispatch_warp_affine_dtype<float1>, 0, dispatch_warp_affine_dtype<float3>, dispatch_warp_affine_dtype<float4>}},
            {eDataType::DATA_TYPE_F64, {dispatch_warp_affine_dtype<double1>, 0, dispatch_warp_affine_dtype<double3>, dispatch_warp_affine_dtype<double4>}}
        };
    // clang-format on

    auto func = funcs.at(dtype)[channels - 1];
    if (func == 0) throw Exception("Not mapped to a defined function.", eStatusType::INVALID_OPERATION);
    func(stream, input, output, affineInv, interp, borderMode, borderValue, device);
}
}  // namespace roccv
