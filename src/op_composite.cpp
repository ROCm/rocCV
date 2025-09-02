/*
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include "op_composite.hpp"

#include <functional>

#include "common/validation_helpers.hpp"
#include "core/wrappers/image_wrapper.hpp"
#include "kernels/device/composite_device.hpp"
#include "kernels/host/composite_host.hpp"

namespace roccv {

template <typename SrcType, typename DstType, typename MaskType>
void dispatch_composite_masktype(hipStream_t stream, const Tensor& foreground, const Tensor& background,
                                 const Tensor& mask, const Tensor& output, const eDeviceType device) {
    ImageWrapper<SrcType> fgWrapper(foreground);
    ImageWrapper<SrcType> bgWrapper(background);
    ImageWrapper<MaskType> maskWrapper(mask);
    ImageWrapper<DstType> outputWrapper(output);

    switch (device) {
        case eDeviceType::GPU: {
            dim3 block(64, 16);
            dim3 grid((outputWrapper.width() + block.x - 1) / block.x, (outputWrapper.height() + block.y - 1) / block.y,
                      outputWrapper.batches());
            Kernels::Device::composite<<<grid, block, 0, stream>>>(fgWrapper, bgWrapper, maskWrapper, outputWrapper);
            break;
        }

        case eDeviceType::CPU: {
            Kernels::Host::composite(fgWrapper, bgWrapper, maskWrapper, outputWrapper);
            break;
        }
    }
}

template <typename SrcType, typename DstType>
void dispatch_composite_dsttype(hipStream_t stream, const Tensor& foreground, const Tensor& background,
                                const Tensor& mask, const Tensor& output, const eDeviceType device) {
    // clang-format off
    static const std::unordered_map<eDataType, std::array<std::function<void(hipStream_t, const Tensor&, const Tensor&, const Tensor&, const Tensor&, const eDeviceType)>, 4>>
        funcs = {
            {eDataType::DATA_TYPE_U8, {dispatch_composite_masktype<SrcType, DstType, uchar1>, 0, 0, 0}},
            {eDataType::DATA_TYPE_F32, {dispatch_composite_masktype<SrcType, DstType, float1>, 0, 0, 0}}
        };
    // clang-format on

    if (!funcs.contains(mask.dtype().etype())) {
        throw Exception("Operator does not support the given datatype for the mask tensor.",
                        eStatusType::NOT_IMPLEMENTED);
    }

    auto func = funcs.at(mask.dtype().etype())[mask.shape(mask.layout().channels_index()) - 1];
    if (func == 0) {
        throw Exception("Operator does not support the given channel count for the mask tensor.",
                        eStatusType::NOT_IMPLEMENTED);
    }

    func(stream, foreground, background, mask, output, device);
}

template <typename SrcType>
void dispatch_composite_srctype(hipStream_t stream, const Tensor& foreground, const Tensor& background,
                                const Tensor& mask, const Tensor& output, const eDeviceType device) {
    // clang-format off
    static const std::unordered_map<eDataType, std::array<std::function<void(hipStream_t, const Tensor&, const Tensor&, const Tensor&, const Tensor&, const eDeviceType)>, 4>>
        funcs = {
            {eDataType::DATA_TYPE_U8,  {0, 0, dispatch_composite_dsttype<SrcType, uchar3>, dispatch_composite_dsttype<SrcType, uchar4>}},
            {eDataType::DATA_TYPE_F32, {0, 0, dispatch_composite_dsttype<SrcType, float3>, dispatch_composite_dsttype<SrcType, float4>}}
        };
    // clang-format on

    auto func = funcs.at(output.dtype().etype())[output.shape(output.layout().channels_index()) - 1];
    if (func == 0) throw Exception("Not mapped to a defined function.", eStatusType::INVALID_OPERATION);
    func(stream, foreground, background, mask, output, device);
}

void Composite::operator()(hipStream_t stream, const Tensor& foreground, const Tensor& background, const Tensor& mask,
                           const Tensor& output, const eDeviceType device) const {
    // Validate foreground tensor
    CHECK_TENSOR_DEVICE(foreground, device);
    CHECK_TENSOR_LAYOUT(foreground, eTensorLayout::TENSOR_LAYOUT_NHWC, eTensorLayout::TENSOR_LAYOUT_HWC);
    CHECK_TENSOR_DATATYPES(foreground, eDataType::DATA_TYPE_U8, eDataType::DATA_TYPE_F32);
    CHECK_TENSOR_CHANNELS(foreground, 3);

    // Validate background tensor
    CHECK_TENSOR_DEVICE(background, device);
    CHECK_TENSOR_LAYOUT(background, eTensorLayout::TENSOR_LAYOUT_NHWC, eTensorLayout::TENSOR_LAYOUT_HWC);
    CHECK_TENSOR_DATATYPES(background, eDataType::DATA_TYPE_U8, eDataType::DATA_TYPE_F32);
    CHECK_TENSOR_CHANNELS(background, 3);
    CHECK_TENSOR_COMPARISON(foreground.shape() == background.shape());

    // Validate mask tensor
    CHECK_TENSOR_DEVICE(mask, device);
    CHECK_TENSOR_LAYOUT(mask, eTensorLayout::TENSOR_LAYOUT_NHWC, eTensorLayout::TENSOR_LAYOUT_HWC);
    CHECK_TENSOR_DATATYPES(mask, eDataType::DATA_TYPE_U8, eDataType::DATA_TYPE_F32);
    CHECK_TENSOR_COMPARISON(mask.layout() == foreground.layout());

    // If the mask contains a batch index, ensure it contains the same number of images as the foreground and background
    // tensors.
    if (mask.layout().batch_index() != -1) {
        CHECK_TENSOR_COMPARISON(mask.shape(mask.layout().batch_index()) ==
                                foreground.shape(foreground.layout().batch_index()));
    }
    CHECK_TENSOR_CHANNELS(mask, 1);

    // Validate output tensor
    CHECK_TENSOR_DEVICE(output, device);
    CHECK_TENSOR_LAYOUT(output, eTensorLayout::TENSOR_LAYOUT_NHWC, eTensorLayout::TENSOR_LAYOUT_HWC);
    CHECK_TENSOR_DATATYPES(output, eDataType::DATA_TYPE_U8, eDataType::DATA_TYPE_F32);
    CHECK_TENSOR_COMPARISON(output.shape(output.layout().width_index()) ==
                            foreground.shape(foreground.layout().width_index()));
    CHECK_TENSOR_COMPARISON(output.shape(output.layout().height_index()) ==
                            foreground.shape(foreground.layout().height_index()));
    CHECK_TENSOR_COMPARISON(output.layout() == foreground.layout());

    // If the output has a layout with a batch index, ensure it contains the same number of images as the foreground and
    // background tensors.
    if (output.layout().batch_index() != -1) {
        CHECK_TENSOR_COMPARISON(output.shape(output.layout().batch_index()) ==
                                foreground.shape(foreground.layout().batch_index()));
    }

    CHECK_TENSOR_CHANNELS(output, 3, 4);

    // clang-format off
    static const std::unordered_map<eDataType, std::array<std::function<void(hipStream_t, const Tensor&, const Tensor&, const Tensor&, const Tensor&, const eDeviceType)>, 4>>
        funcs = {
            {eDataType::DATA_TYPE_U8,  {0, 0, dispatch_composite_srctype<uchar3>, 0}},
            {eDataType::DATA_TYPE_F32, {0, 0, dispatch_composite_srctype<float3>, 0}}
        };
    // clang-format on

    auto func = funcs.at(foreground.dtype().etype())[foreground.shape(output.layout().channels_index()) - 1];
    if (func == 0) throw Exception("Not mapped to a defined function.", eStatusType::INVALID_OPERATION);
    func(stream, foreground, background, mask, output, device);
}
};  // namespace roccv