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
#include "common/validation_helpers.hpp"
#include "core/detail/casting.hpp"
#include "core/detail/internal_structs.hpp"
#include "core/detail/math/math.hpp"
#include "core/detail/type_traits.hpp"
#include "core/wrappers/image_wrapper.hpp"
#include "core/wrappers/interpolation_wrapper.hpp"
#include "kernels/device/remap_device.hpp"
#include "kernels/host/remap_host.hpp"
#include "operator_types.h"

using namespace roccv::detail;
namespace roccv {
Remap::Remap() {}

Remap::~Remap() {}

RemapParams GetRemapParams(const int2 &srcSize, const int2 &dstSize, const int2 &mapSize, bool alignCorners,
                           eRemapType mapValueType) {
    RemapParams params{};

    switch (mapValueType) {
        case REMAP_ABSOLUTE:
            params.srcScale = make_float2(0.f, 0.f);
            params.mapScale = StaticCast<float2>(mapSize) / StaticCast<float2>(dstSize);
            params.valScale = make_float2(1.f, 1.f);
            params.srcOffset = make_float2(0.f, 0.f);
            params.dstOffset = 0.f;
            break;
        case REMAP_ABSOLUTE_NORMALIZED:
            params.srcScale = make_float2(0.f, 0.f);
            params.mapScale = StaticCast<float2>(mapSize) / StaticCast<float2>(dstSize);
            params.valScale = (StaticCast<float2>(srcSize) - (alignCorners ? 1.f : 0.f)) / 2.f;
            params.srcOffset = params.valScale - (alignCorners ? 0.f : .5f);
            params.dstOffset = 0.f;
            break;
        case REMAP_RELATIVE_NORMALIZED:
            params.srcScale = StaticCast<float2>(srcSize) / StaticCast<float2>(dstSize);
            params.mapScale = (StaticCast<float2>(mapSize) - 1.f) / StaticCast<float2>(dstSize);
            params.valScale = StaticCast<float2>(srcSize) - 1.f;
            params.dstOffset = alignCorners ? 0.f : .5f;
            params.srcOffset = params.srcScale * params.dstOffset - params.dstOffset;
            break;
        default:
            throw Exception("Unsupported mapValueType passed to GetRemapParams", eStatusType::NOT_IMPLEMENTED);
    }
    return params;
}

template <typename T, eBorderType B, eInterpolationType I, eInterpolationType M>
void dispatch_remap_mapInterp(hipStream_t stream, const Tensor &input, const Tensor &output, const Tensor &map,
                              const eRemapType mapValueType, const bool alignCorners, const T borderValue,
                              const eDeviceType device) {
    ImageWrapper<T> outputWrapper(output);
    InterpolationWrapper<float2, B, M> wrappedMapTensor(map, make_float2(0, 0));
    InterpolationWrapper<T, B, I> inputWrapper(input, borderValue);

    int mapBatchSize = wrappedMapTensor.batches();

    int2 srcSize = make_int2(inputWrapper.width(), inputWrapper.height());
    int2 dstSize = make_int2(outputWrapper.width(), outputWrapper.height());
    int2 mapSize = make_int2(wrappedMapTensor.width(), wrappedMapTensor.height());

    RemapParams params = GetRemapParams(srcSize, dstSize, mapSize, alignCorners, mapValueType);

    // Launch CPU/GPU kernel depending on requested device type.
    switch (device) {
        case eDeviceType::GPU: {
            dim3 block(64, 16);
            dim3 grid((outputWrapper.width() + block.x - 1) / block.x, (outputWrapper.height() + block.y - 1) / block.y,
                      outputWrapper.batches());
            Kernels::Device::remap<<<grid, block, 0, stream>>>(inputWrapper, outputWrapper, wrappedMapTensor,
                                                               mapBatchSize, params);
            break;
        }

        case eDeviceType::CPU: {
            Kernels::Host::remap(inputWrapper, outputWrapper, wrappedMapTensor, mapBatchSize, params);
            break;
        }
    }
}

template <typename T, eBorderType B, eInterpolationType I>
void dispatch_remap_interp(hipStream_t stream, const Tensor &input, const Tensor &output, const Tensor &map,
                           const eInterpolationType mapInterpolation, const eRemapType mapValueType,
                           const bool alignCorners, const T borderValue, const eDeviceType device) {
    // Select kernel dispatcher based on selected interpolation mode.
    // clang-format off
    static const std::unordered_map<eInterpolationType, std::function<void(hipStream_t stream, const Tensor&, const Tensor&, const Tensor&, const eRemapType, const bool, const T, const eDeviceType)>>
        funcs = {
            {eInterpolationType::INTERP_TYPE_NEAREST, dispatch_remap_mapInterp<T, B, I, eInterpolationType::INTERP_TYPE_NEAREST>},
            {eInterpolationType::INTERP_TYPE_LINEAR,  dispatch_remap_mapInterp<T, B, I, eInterpolationType::INTERP_TYPE_LINEAR>},
            {eInterpolationType::INTERP_TYPE_CUBIC,  dispatch_remap_mapInterp<T, B, I, eInterpolationType::INTERP_TYPE_CUBIC>}
        };  // clang-format on

    if (!funcs.contains(mapInterpolation)) {
        throw Exception("Operation does not support the given interpolation mode for mapInterpolation.",
                        eStatusType::NOT_IMPLEMENTED);
    }

    auto func = funcs.at(mapInterpolation);
    func(stream, input, output, map, mapValueType, alignCorners, borderValue, device);
}

template <typename T, eBorderType B>
void dispatch_remap_border_mode(hipStream_t stream, const Tensor &input, const Tensor &output, const Tensor &map,
                                const eInterpolationType inInterpolation, const eInterpolationType mapInterpolation,
                                const eRemapType mapValueType, const bool alignCorners, const T borderValue,
                                const eDeviceType device) {
    // Select kernel dispatcher based on selected interpolation mode.
    // clang-format off
    static const std::unordered_map<eInterpolationType, std::function<void(hipStream_t stream, const Tensor&, const Tensor&, const Tensor&, const eInterpolationType, const eRemapType, const bool, const T, const eDeviceType)>>
        funcs = {
            {eInterpolationType::INTERP_TYPE_NEAREST, dispatch_remap_interp<T, B, eInterpolationType::INTERP_TYPE_NEAREST>},
            {eInterpolationType::INTERP_TYPE_LINEAR,  dispatch_remap_interp<T, B, eInterpolationType::INTERP_TYPE_LINEAR>},
            {eInterpolationType::INTERP_TYPE_CUBIC,  dispatch_remap_interp<T, B, eInterpolationType::INTERP_TYPE_CUBIC>}
        };  // clang-format on

    if (!funcs.contains(inInterpolation)) {
        throw Exception("Remap does not support the given interpolation mode for inInterpolation.",
                        eStatusType::NOT_IMPLEMENTED);
    }

    auto func = funcs.at(inInterpolation);
    func(stream, input, output, map, mapInterpolation, mapValueType, alignCorners, borderValue, device);
}

template <typename T>
void dispatch_remap_dtype(hipStream_t stream, const Tensor &input, const Tensor &output, const Tensor &map,
                          const eInterpolationType inInterpolation, const eInterpolationType mapInterpolation,
                          const eRemapType mapValueType, const bool alignCorners, const eBorderType borderType,
                          const float4 borderValue, const eDeviceType device) {
    // Select kernel dispatcher based on requested border mode.
    // clang-format off
    static const std::unordered_map<eBorderType, std::function<void(hipStream_t, const Tensor&, const Tensor&, const Tensor&, const eInterpolationType, const eInterpolationType, const eRemapType, const bool, T, const eDeviceType)>>
        funcs = {
            {eBorderType::BORDER_TYPE_CONSTANT,     dispatch_remap_border_mode<T, eBorderType::BORDER_TYPE_CONSTANT>},
            {eBorderType::BORDER_TYPE_REPLICATE,    dispatch_remap_border_mode<T, eBorderType::BORDER_TYPE_REPLICATE>},
            {eBorderType::BORDER_TYPE_REFLECT,      dispatch_remap_border_mode<T, eBorderType::BORDER_TYPE_REFLECT>},
            {eBorderType::BORDER_TYPE_REFLECT101,   dispatch_remap_border_mode<T, eBorderType::BORDER_TYPE_REFLECT101>},
            {eBorderType::BORDER_TYPE_WRAP,         dispatch_remap_border_mode<T, eBorderType::BORDER_TYPE_WRAP>}
        };
    // clang-format on

    if (!funcs.contains(borderType)) {
        throw Exception("Remap does not support the given border mode.", eStatusType::NOT_IMPLEMENTED);
    }

    auto func = funcs.at(borderType);
    func(stream, input, output, map, inInterpolation, mapInterpolation, mapValueType, alignCorners,
         detail::SaturateCast<T>(borderValue), device);
}

void Remap::operator()(hipStream_t stream, const Tensor &input, const Tensor &output, const Tensor &map,
                       const eInterpolationType inInterpolation, const eInterpolationType mapInterpolation,
                       const eRemapType mapValueType, const bool alignCorners, const eBorderType borderType,
                       const float4 borderValue, eDeviceType device) {
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
    CHECK_TENSOR_COMPARISON(map.layout() == output.layout());
    CHECK_TENSOR_COMPARISON((map.shape(map.layout().batch_index()) == input.shape(input.layout().batch_index())) ||
                            (map.shape(map.layout().batch_index()) == 1));

    CHECK_TENSOR_CHANNELS(input, 1, 3, 4);
    CHECK_TENSOR_CHANNELS(map, 2);

    if (needsInt64Wrapper(input) || needsInt64Wrapper(output)) {
        throw Exception("Input or output tensor is too large for int32 indexing", eStatusType::INVALID_OPERATION);
    }

    eDataType dtype = input.dtype().etype();
    int64_t channels = input.shape(input.layout().channels_index());

    // Select kernel dispatcher based on number of channels and a base datatype.
    // clang-format off
    static const std::unordered_map<eDataType, std::array<std::function<void(hipStream_t, const Tensor &, const Tensor &, const Tensor &, const eInterpolationType, const eInterpolationType, const eRemapType,  const bool, const eBorderType, const float4, const eDeviceType)>, 4>>
        funcs = {
            {eDataType::DATA_TYPE_U8, {dispatch_remap_dtype<uchar1>, 0, dispatch_remap_dtype<uchar3>, dispatch_remap_dtype<uchar4>}},
        };
    // clang-format on

    auto func = funcs.at(dtype)[channels - 1];
    if (func == 0) throw Exception("Not mapped to a defined function.", eStatusType::INVALID_OPERATION);
    func(stream, input, output, map, inInterpolation, mapInterpolation, mapValueType, alignCorners, borderType,
         borderValue, device);
}
}  // namespace roccv