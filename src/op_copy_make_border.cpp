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

#include "op_copy_make_border.hpp"

#include <functional>

#include "common/conversion_helpers.hpp"
#include "common/validation_helpers.hpp"
#include "core/wrappers/border_wrapper.hpp"
#include "core/wrappers/image_wrapper.hpp"
#include "core/wrappers/interpolation_wrapper.hpp"
#include "kernels/device/copy_make_border_device.hpp"
#include "kernels/host/copy_make_border_host.hpp"

namespace roccv {

/**
 * @brief Dispatches the copy_make_border kernel given a BorderType.
 */
template <typename T, eBorderType BorderMode>
void dispatch_copy_make_border_border_mode(hipStream_t stream, const Tensor& input, const Tensor& output, int32_t top,
                                           int32_t left, T border_value, const eDeviceType device) {
    BorderWrapper<T, BorderMode> in_desc(input, border_value);
    ImageWrapper<T> out_desc(output);

    switch (device) {
        case eDeviceType::GPU: {
            dim3 block_dim(64, 16);
            dim3 grid_dim((out_desc.width() + block_dim.x - 1) / block_dim.x,
                          (out_desc.height() + block_dim.y - 1) / block_dim.y, out_desc.batches());
            Kernels::Device::copy_make_border<<<grid_dim, block_dim, 0, stream>>>(in_desc, out_desc, top, left);
            break;
        }
        case eDeviceType::CPU: {
            Kernels::Host::copy_make_border(in_desc, out_desc, top, left);
            break;
        }
    }
}

/**
 * @brief Dispatches the copy_make_border kernel given a vectorized type.
 */
template <typename T>
void dispatch_copy_make_border(hipStream_t stream, const Tensor& input, const Tensor& output, int32_t top, int32_t left,
                               eBorderType border_mode, float4 border_value, const eDeviceType device) {
    // clang-format off
    // Maps copy_make_border dispatchers to a given border type.
    static const std::unordered_map<eBorderType, std::function<void(hipStream_t, const Tensor&, const Tensor&, int32_t, int32_t, T, const eDeviceType)>>
    funcs = {
        {eBorderType::BORDER_TYPE_CONSTANT,     dispatch_copy_make_border_border_mode<T, eBorderType::BORDER_TYPE_CONSTANT>},
        {eBorderType::BORDER_TYPE_REPLICATE,    dispatch_copy_make_border_border_mode<T, eBorderType::BORDER_TYPE_REPLICATE>},
        {eBorderType::BORDER_TYPE_REFLECT,      dispatch_copy_make_border_border_mode<T, eBorderType::BORDER_TYPE_REFLECT>},
        {eBorderType::BORDER_TYPE_WRAP,         dispatch_copy_make_border_border_mode<T, eBorderType::BORDER_TYPE_WRAP>}
    };
    // clang-format on

    if (!funcs.contains(border_mode)) {
        throw Exception("Operation does not support the given border mode.", eStatusType::NOT_IMPLEMENTED);
    }

    auto func = funcs.at(border_mode);
    func(stream, input, output, top, left, detail::RangeCast<T>(border_value), device);
}

void CopyMakeBorder::operator()(hipStream_t stream, const Tensor& input, const Tensor& output, int32_t top,
                                int32_t left, eBorderType border_mode, float4 border_value,
                                const eDeviceType device) const {
    CHECK_TENSOR_DEVICE(input, device);
    CHECK_TENSOR_LAYOUT(input, eTensorLayout::TENSOR_LAYOUT_NHWC, eTensorLayout::TENSOR_LAYOUT_HWC);
    CHECK_TENSOR_DATATYPES(input, eDataType::DATA_TYPE_U8, eDataType::DATA_TYPE_S8, eDataType::DATA_TYPE_U32,
                           eDataType::DATA_TYPE_S32, eDataType::DATA_TYPE_F32);

    CHECK_TENSOR_DEVICE(output, device);
    CHECK_TENSOR_COMPARISON(output.dtype() == input.dtype());
    CHECK_TENSOR_COMPARISON(output.layout() == input.layout());
    CHECK_TENSOR_COMPARISON(output.shape(output.layout().batch_index()) == input.shape(input.layout().batch_index()));
    CHECK_TENSOR_COMPARISON(output.shape(output.layout().channels_index()) ==
                            input.shape(input.layout().channels_index()));

    // clang-format off
    // Maps kernel dispatchers according to the underlying data type and number of channels.
    static const std::unordered_map<eDataType, std::array<std::function<void(hipStream_t, const Tensor&, const Tensor&, int32_t, int32_t, eBorderType, float4, const eDeviceType)>, 4>>
        funcs = {
            {eDataType::DATA_TYPE_U8,  {dispatch_copy_make_border<uchar1>, 0,  dispatch_copy_make_border<uchar3>,  dispatch_copy_make_border<uchar4>}},
            {eDataType::DATA_TYPE_S8,  {dispatch_copy_make_border<char1>,  0,  dispatch_copy_make_border<char3>,   dispatch_copy_make_border<char4>}},
            {eDataType::DATA_TYPE_U32, {dispatch_copy_make_border<uint1>,  0,  dispatch_copy_make_border<uint3>,   dispatch_copy_make_border<uint4>}},
            {eDataType::DATA_TYPE_S32, {dispatch_copy_make_border<int1>,   0,  dispatch_copy_make_border<int3>,    dispatch_copy_make_border<int4>}},
            {eDataType::DATA_TYPE_F32, {dispatch_copy_make_border<float1>, 0,  dispatch_copy_make_border<float3>,  dispatch_copy_make_border<float4>}}
        };
    // clang-format on

    eDataType dtype = output.dtype().etype();
    int64_t channels = output.shape(output.layout().channels_index());

    auto func = funcs.at(dtype)[channels - 1];
    if (func == 0) throw Exception("Not mapped to a defined function.", eStatusType::INVALID_OPERATION);
    func(stream, input, output, top, left, border_mode, border_value, device);
}
}  // namespace roccv