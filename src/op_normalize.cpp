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

#include "op_normalize.hpp"

#include <functional>
#include <unordered_map>

#include "common/validation_helpers.hpp"
#include "core/detail/type_traits.hpp"
#include "core/tensor.hpp"
#include "core/wrappers/image_wrapper.hpp"
#include "kernels/device/normalize_device.hpp"
#include "kernels/host/normalize_host.hpp"

namespace roccv {

template <typename T, bool ScaleStddev>
void dispatch_normalize_stddev(hipStream_t stream, const Tensor& input, const Tensor& base, const Tensor& scale,
                               const Tensor& output, float global_scale, float shift, float epsilon,
                               const eDeviceType device) {
    // Work type for base/stddev tensors, these must be floats with the same number of channels as the input/output
    // tensors.
    using work_type = detail::MakeType<float, detail::NumComponents<T>>;

    ImageWrapper<T> inputWrap(input);
    ImageWrapper<T> outputWrap(output);
    ImageWrapper<work_type> scaleWrap(scale);
    ImageWrapper<work_type> baseWrap(base);

    switch (device) {
        case eDeviceType::GPU: {
            dim3 block(32, 8);
            dim3 grid((outputWrap.width() + block.x - 1) / block.x, (outputWrap.height() + block.y - 1) / block.y,
                      outputWrap.batches());
            Kernels::Device::normalize<ScaleStddev>
                <<<grid, block, 0, stream>>>(inputWrap, baseWrap, scaleWrap, outputWrap, global_scale, shift, epsilon);
            break;
        }
        case eDeviceType::CPU: {
            Kernels::Host::normalize<ScaleStddev>(inputWrap, baseWrap, scaleWrap, outputWrap, global_scale, shift,
                                                  epsilon);
            break;
        }
    }
}

template <typename T>
void dispatch_normalize_dtype(hipStream_t stream, const Tensor& input, const Tensor& base, const Tensor& scale,
                              const Tensor& output, float global_scale, float shift, float epsilon, uint32_t flags,
                              const eDeviceType device) {
    // Create kernel dispatching table based on whether or not scale is interpreted as standard deviation or not.
    std::function<void(hipStream_t stream, const Tensor& input, const Tensor& base, const Tensor& scale,
                       const Tensor& output, float global_scale, float shift, float epsilon, const eDeviceType device)>
        funcs[2] = {dispatch_normalize_stddev<T, false>, dispatch_normalize_stddev<T, true>};

    auto func = funcs[(flags & ROCCV_NORMALIZE_SCALE_IS_STDDEV) != 0];
    func(stream, input, base, scale, output, global_scale, shift, epsilon, device);
}

void Normalize::operator()(hipStream_t stream, const Tensor& input, const Tensor& base, const Tensor& scale,
                           const Tensor& output, float global_scale, float shift, float epsilon, uint32_t flags,
                           const eDeviceType device) const {
    // Check all tensors are on the proper device
    CHECK_TENSOR_DEVICE(input, device);
    CHECK_TENSOR_DEVICE(base, device);
    CHECK_TENSOR_DEVICE(scale, device);
    CHECK_TENSOR_DEVICE(output, device);

    // Check tensor metadata to ensure supported types, layout, and channel count
    CHECK_TENSOR_DATATYPES(input, DATA_TYPE_U8, DATA_TYPE_S8, DATA_TYPE_U32, DATA_TYPE_S32, DATA_TYPE_F32,
                           DATA_TYPE_S16);
    CHECK_TENSOR_LAYOUT(input, TENSOR_LAYOUT_HWC, TENSOR_LAYOUT_NHWC);
    CHECK_TENSOR_CHANNELS(input, 1, 3, 4);
    CHECK_TENSOR_DATATYPES(scale, DATA_TYPE_F32);
    CHECK_TENSOR_DATATYPES(base, DATA_TYPE_F32);

    // Compare tensors with each other to ensure they are valid combinations
    CHECK_TENSOR_COMPARISON(input.dtype() == output.dtype());
    CHECK_TENSOR_COMPARISON(input.layout() == output.layout());
    CHECK_TENSOR_COMPARISON(input.layout() == scale.layout());
    CHECK_TENSOR_COMPARISON(input.layout() == base.layout());
    CHECK_TENSOR_COMPARISON(input.shape() == output.shape());

    // Check that input/output shape matches with base/scale tensors
    for (int i = 0; i < input.rank(); i++) {
        CHECK_TENSOR_COMPARISON(base.shape(i) == 1 || base.shape(i) == input.shape(i));
        CHECK_TENSOR_COMPARISON(scale.shape(i) == 1 || scale.shape(i) == input.shape(i));
    }

    // Check that base/scale tensors have the same number of channels as the input/output tensors.
    // TODO: Need to support scalar base/scale tensors at some point. Will require some extra handling on the kernel
    // level. Once in place, this check can be removed.
    CHECK_TENSOR_COMPARISON(base.shape(base.layout().channels_index()) == input.shape(input.layout().channels_index()));
    CHECK_TENSOR_COMPARISON(scale.shape(scale.layout().channels_index()) ==
                            input.shape(input.layout().channels_index()));

    // Create kernel dispatching table based on input/output datatype and number of channels.
    // clang-format off
    static const std::unordered_map<
        eDataType, std::array<std::function<void(hipStream_t, const Tensor&, const Tensor&, const Tensor&,
                                      const Tensor&, float, float, float,
                                      uint32_t, const eDeviceType)>, 4>>
        funcs =
    {
        {eDataType::DATA_TYPE_U8, {dispatch_normalize_dtype<uchar1>, nullptr, dispatch_normalize_dtype<uchar3>, dispatch_normalize_dtype<uchar4>}},
        {eDataType::DATA_TYPE_S8, {dispatch_normalize_dtype<char1>, nullptr, dispatch_normalize_dtype<char3>, dispatch_normalize_dtype<char4>}},
        {eDataType::DATA_TYPE_U32, {dispatch_normalize_dtype<uint1>, nullptr, dispatch_normalize_dtype<uint3>, dispatch_normalize_dtype<uint4>}},
        {eDataType::DATA_TYPE_S32, {dispatch_normalize_dtype<int1>, nullptr, dispatch_normalize_dtype<int3>, dispatch_normalize_dtype<int4>}},
        {eDataType::DATA_TYPE_F32, {dispatch_normalize_dtype<float1>, nullptr, dispatch_normalize_dtype<float3>, dispatch_normalize_dtype<float4>}},
        {eDataType::DATA_TYPE_S16, {dispatch_normalize_dtype<short1>, nullptr, dispatch_normalize_dtype<short3>, dispatch_normalize_dtype<short4>}}
    };
    // clang-format on

    auto func = funcs.at(input.dtype().etype())[input.shape(input.layout().channels_index()) - 1];
    func(stream, input, base, scale, output, global_scale, shift, epsilon, flags, device);
}
}  // namespace roccv
