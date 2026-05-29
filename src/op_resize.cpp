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
#include "op_resize.hpp"

#include <functional>
#include <unordered_map>

#include "common/validation_helpers.hpp"
#include "core/detail/casting.hpp"
#include "core/exception.hpp"
#include "core/image_batch_var_shape.hpp"
#include "core/status_type.h"
#include "core/wrappers/image_batch_var_shape_wrapper.hpp"
#include "core/wrappers/interpolation_wrapper.hpp"
#include "kernels/device/resize_device.hpp"
#include "kernels/host/resize_host.hpp"

namespace roccv {

// Common launch tail shared by the uniform (Tensor) and variable-shape (ImageBatchVarShape) input paths. The
// source base wrapper (TensorWrapper<T> or ImageBatchVarShapeWrapper<T>) is built by the caller; everything
// downstream — border/interpolation composition, output wrapper, grid, kernel — is identical. The kernel derives
// the per-image scale internally, so no scale is computed here.
template <typename T, eInterpolationType I, typename SrcBaseWrapper>
void dispatch_resize_launch(hipStream_t stream, SrcBaseWrapper srcBase, const Tensor& output, eDeviceType device) {
    TensorWrapper<T> outputWrapper(output);
    // Resize operation should clamp values at the border (REPLICATE border mode)
    auto inputWrapper =
        MakeInterpolationWrapper<I>(MakeBorderWrapper<eBorderType::BORDER_TYPE_REPLICATE>(srcBase, T{}));

    switch (device) {
        case eDeviceType::GPU: {
            dim3 block(64, 16);
            dim3 grid((outputWrapper.width() + block.x - 1) / block.x, (outputWrapper.height() + block.y - 1) / block.y,
                      outputWrapper.batches());
            Kernels::Device::resize<<<grid, block, 0, stream>>>(inputWrapper, outputWrapper);
            break;
        }

        case eDeviceType::CPU: {
            Kernels::Host::resize(inputWrapper, outputWrapper);
            break;
        }
    }
}

template <typename T, eInterpolationType I>
void dispatch_resize_interp(hipStream_t stream, const Tensor& input, const Tensor& output, eDeviceType device) {
    dispatch_resize_launch<T, I>(stream, TensorWrapper<T>(input), output, device);
}

template <typename T, eInterpolationType I>
void dispatch_resize_interp_var(hipStream_t stream, ImageBatchVarShape& input, const Tensor& output,
                                eDeviceType device) {
    // The exported snapshot is a non-owning view into the batch's descriptor table; the wrapper copies out the
    // device/host pointer it needs, and the batch outlives this call.
    auto data = input.exportData(stream);
    dispatch_resize_launch<T, I>(stream, ImageBatchVarShapeWrapper<T>(data), output, device);
}

template <typename T>
void dispatch_resize_dtype(hipStream_t stream, const Tensor& input, const Tensor& output,
                           eInterpolationType interpolation, eDeviceType device) {
    static const std::unordered_map<eInterpolationType, std::function<void(hipStream_t stream, const Tensor& input,
                                                                           const Tensor& output, eDeviceType device)>>
        funcs = {
            {eInterpolationType::INTERP_TYPE_NEAREST,
             dispatch_resize_interp<T, eInterpolationType::INTERP_TYPE_NEAREST>},
            {eInterpolationType::INTERP_TYPE_LINEAR, dispatch_resize_interp<T, eInterpolationType::INTERP_TYPE_LINEAR>},
            {eInterpolationType::INTERP_TYPE_CUBIC, dispatch_resize_interp<T, eInterpolationType::INTERP_TYPE_CUBIC>}};

    if (!funcs.contains(interpolation)) {
        throw Exception("Operation does not support the given interpolation mode.", eStatusType::NOT_IMPLEMENTED);
    }

    auto func = funcs.at(interpolation);
    func(stream, input, output, device);
}

template <typename T>
void dispatch_resize_dtype_var(hipStream_t stream, ImageBatchVarShape& input, const Tensor& output,
                               eInterpolationType interpolation, eDeviceType device) {
    static const std::unordered_map<
        eInterpolationType,
        std::function<void(hipStream_t stream, ImageBatchVarShape & input, const Tensor& output, eDeviceType device)>>
        funcs = {{eInterpolationType::INTERP_TYPE_NEAREST,
                  dispatch_resize_interp_var<T, eInterpolationType::INTERP_TYPE_NEAREST>},
                 {eInterpolationType::INTERP_TYPE_LINEAR,
                  dispatch_resize_interp_var<T, eInterpolationType::INTERP_TYPE_LINEAR>},
                 {eInterpolationType::INTERP_TYPE_CUBIC,
                  dispatch_resize_interp_var<T, eInterpolationType::INTERP_TYPE_CUBIC>}};

    if (!funcs.contains(interpolation)) {
        throw Exception("Operation does not support the given interpolation mode.", eStatusType::NOT_IMPLEMENTED);
    }

    auto func = funcs.at(interpolation);
    func(stream, input, output, device);
}

void Resize::operator()(hipStream_t stream, const Tensor& input, const Tensor& output, eInterpolationType interpolation,
                        eDeviceType device) const {
    CHECK_TENSOR_DEVICE(input, device);
    CHECK_TENSOR_DEVICE(output, device);

    CHECK_TENSOR_CHANNELS(input, 1, 3, 4);
    CHECK_TENSOR_DATATYPES(input, DATA_TYPE_U8, DATA_TYPE_F32);
    CHECK_TENSOR_LAYOUT(input, TENSOR_LAYOUT_HWC, TENSOR_LAYOUT_NHWC);

    CHECK_TENSOR_COMPARISON(input.layout() == output.layout());
    CHECK_TENSOR_COMPARISON(input.dtype() == output.dtype());
    CHECK_TENSOR_COMPARISON(input.shape(input.layout().channels_index()) ==
                            output.shape(output.layout().channels_index()));
    if (input.layout().batch_index() != -1) {
        CHECK_TENSOR_COMPARISON(input.shape(input.layout().batch_index()) ==
                                output.shape(output.layout().batch_index()));
    }

    // clang-format off
    static const std::unordered_map<eDataType, std::array<std::function<void(hipStream_t stream, const Tensor& input, const Tensor& output,
                       eInterpolationType interpolation, eDeviceType device)>, 4>>
        funcs = {
            {eDataType::DATA_TYPE_U8, {dispatch_resize_dtype<uchar1>, 0, dispatch_resize_dtype<uchar3>, dispatch_resize_dtype<uchar4>}},
            {eDataType::DATA_TYPE_F32, {dispatch_resize_dtype<float1>, 0, dispatch_resize_dtype<float3>, dispatch_resize_dtype<float4>}}
        };
    // clang-format on

    auto func = funcs.at(input.dtype().etype())[input.shape(input.layout().channels_index()) - 1];
    if (func == 0) throw Exception("Not mapped to a defined function.", eStatusType::INVALID_OPERATION);
    func(stream, input, output, interpolation, device);
}

void Resize::operator()(hipStream_t stream, ImageBatchVarShape& input, const Tensor& output,
                        eInterpolationType interpolation, eDeviceType device) const {
    // The variable-shape batch resolves dtype/channels from its single shared format; a heterogeneous (or empty)
    // batch can't be expressed by the T-templated wrapper and is rejected.
    CHECK_IMAGE_BATCH_DEVICE(input, device);
    CHECK_TENSOR_DEVICE(output, device);

    CHECK_IMAGE_BATCH_UNIFORM_FORMAT(input);
    CHECK_IMAGE_BATCH_DATATYPES(input, DATA_TYPE_U8, DATA_TYPE_F32);
    CHECK_IMAGE_BATCH_CHANNELS(input, 1, 3, 4);

    // The output holds the resized batch as a uniform tensor, so it must carry an explicit batch dimension.
    CHECK_TENSOR_LAYOUT(output, TENSOR_LAYOUT_NHWC);
    CHECK_TENSOR_DATATYPES(output, DATA_TYPE_U8, DATA_TYPE_F32);
    CHECK_TENSOR_CHANNELS(output, 1, 3, 4);

    // Output dtype/channels/batch must agree with the input batch.
    ImageFormat format = input.uniqueFormat();
    CHECK_TENSOR_COMPARISON(output.dtype().etype() == format.dtype());
    CHECK_TENSOR_COMPARISON(output.shape(output.layout().channels_index()) == format.channels());
    CHECK_TENSOR_COMPARISON(output.shape(output.layout().batch_index()) == input.numImages());

    // clang-format off
    static const std::unordered_map<eDataType, std::array<std::function<void(hipStream_t stream, ImageBatchVarShape& input, const Tensor& output,
                       eInterpolationType interpolation, eDeviceType device)>, 4>>
        funcs = {
            {eDataType::DATA_TYPE_U8, {dispatch_resize_dtype_var<uchar1>, 0, dispatch_resize_dtype_var<uchar3>, dispatch_resize_dtype_var<uchar4>}},
            {eDataType::DATA_TYPE_F32, {dispatch_resize_dtype_var<float1>, 0, dispatch_resize_dtype_var<float3>, dispatch_resize_dtype_var<float4>}}
        };
    // clang-format on

    auto func = funcs.at(format.dtype())[format.channels() - 1];
    if (func == 0) throw Exception("Not mapped to a defined function.", eStatusType::INVALID_OPERATION);
    func(stream, input, output, interpolation, device);
}
}  // namespace roccv
