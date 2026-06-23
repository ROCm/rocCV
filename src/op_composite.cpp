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

#include "common/validation_helpers.hpp"
#include "core/wrappers/image_wrapper.hpp"
#include "kernels/device/composite_device.hpp"
#include "kernels/host/composite_host.hpp"

namespace roccv {

template <typename SrcType, typename DstType, typename MaskType>
void dispatch_composite_masktype(hipStream_t stream, const Tensor& foreground, const Tensor& background,
                                 const Tensor& mask, const Tensor& output, eDeviceType device) {
    ImageWrapper<SrcType> fgWrapper(foreground);
    ImageWrapper<SrcType> bgWrapper(background);
    ImageWrapper<MaskType> maskWrapper(mask);
    ImageWrapper<DstType> outputWrapper(output);

    switch (device) {
        case eDeviceType::GPU: {
            dim3 block = Kernels::Device::PackedBlock();
            dim3 grid = Kernels::Device::PackedGrid<DstType>(outputWrapper.width(), outputWrapper.height(),
                                                             outputWrapper.batches(), block);
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
                                const Tensor& mask, const Tensor& output, eDeviceType device) {
    // Mask is validated upstream to be single-channel U8 or F32.
    switch (mask.dtype().etype()) {
        case eDataType::DATA_TYPE_U8:
            dispatch_composite_masktype<SrcType, DstType, uchar1>(stream, foreground, background, mask, output, device);
            break;
        case eDataType::DATA_TYPE_F32:
            dispatch_composite_masktype<SrcType, DstType, float1>(stream, foreground, background, mask, output, device);
            break;
        default:
            throw Exception("Operator does not support the given datatype for the mask tensor.",
                            eStatusType::NOT_IMPLEMENTED);
    }
}

template <typename SrcType>
void dispatch_composite_srctype(hipStream_t stream, const Tensor& foreground, const Tensor& background,
                                const Tensor& mask, const Tensor& output, eDeviceType device) {
    // Output is validated upstream to be U8 or F32 with 3 or 4 channels.
    const eDataType dtype = output.dtype().etype();
    const int64_t channels = output.shape(output.layout().channels_index());

    if (dtype == eDataType::DATA_TYPE_U8 && channels == 3)
        dispatch_composite_dsttype<SrcType, uchar3>(stream, foreground, background, mask, output, device);
    else if (dtype == eDataType::DATA_TYPE_U8 && channels == 4)
        dispatch_composite_dsttype<SrcType, uchar4>(stream, foreground, background, mask, output, device);
    else if (dtype == eDataType::DATA_TYPE_F32 && channels == 3)
        dispatch_composite_dsttype<SrcType, float3>(stream, foreground, background, mask, output, device);
    else if (dtype == eDataType::DATA_TYPE_F32 && channels == 4)
        dispatch_composite_dsttype<SrcType, float4>(stream, foreground, background, mask, output, device);
    else
        throw Exception("Not mapped to a defined function.", eStatusType::INVALID_OPERATION);
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

    // Foreground/background are validated upstream to be 3-channel U8 or F32.
    switch (foreground.dtype().etype()) {
        case eDataType::DATA_TYPE_U8:
            dispatch_composite_srctype<uchar3>(stream, foreground, background, mask, output, device);
            break;
        case eDataType::DATA_TYPE_F32:
            dispatch_composite_srctype<float3>(stream, foreground, background, mask, output, device);
            break;
        default:
            throw Exception("Not mapped to a defined function.", eStatusType::INVALID_OPERATION);
    }
}
};  // namespace roccv