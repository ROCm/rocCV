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
#include "op_rotate.hpp"

#include <functional>
#include <unordered_map>

#include "common/array_wrapper.hpp"
#include "common/validation_helpers.hpp"
#include "core/wrappers/interpolation_wrapper.hpp"
#include "kernels/device/rotate_device.hpp"
#include "kernels/host/rotate_host.hpp"
#include "operator_types.h"

namespace roccv {
void GetRotationMatrix(const double angleDeg, const double2 shift, double *mat) {
    double angleRad = angleDeg * (M_PI / 180.0);
    mat[0] = cos(angleRad);
    mat[1] = sin(angleRad);
    mat[2] = shift.x;
    mat[3] = -sin(angleRad);
    mat[4] = cos(angleRad);
    mat[5] = shift.y;
}

template <typename T, eInterpolationType InterpType>
void dispatch_rotate_interp(hipStream_t stream, const Tensor &input, const Tensor &output, const double angleDeg,
                            const double2 shift, const eDeviceType device) {
    // Get inverted affine matrix for rotation
    double mat[6];
    GetRotationMatrix(angleDeg, shift, mat);

    // Wrap in a kernel-friendly fixed size array to ensure the affine matrix gets transferred to the device properly.
    ArrayWrapper<double, 6> matWrap(mat);

    T borderVal = detail::RangeCast<T>(make_float4(0.0f, 0.0f, 0.0f, 0.0f));

    ImageWrapper<T> outputWrap(output);
    InterpolationWrapper<T, eBorderType::BORDER_TYPE_CONSTANT, InterpType> inputWrap(input, borderVal);

    switch (device) {
        case eDeviceType::GPU: {
            dim3 block(32, 16);
            dim3 grid((outputWrap.width() + block.x - 1) / block.x, (outputWrap.height() + block.y - 1) / block.y,
                      outputWrap.batches());
            Kernels::Device::rotate<<<grid, block, 0, stream>>>(inputWrap, outputWrap, matWrap);
            break;
        }

        case eDeviceType::CPU: {
            Kernels::Host::rotate(inputWrap, outputWrap, matWrap);
            break;
        }
    }
}

template <typename T>
void dispatch_rotate_type(hipStream_t stream, const Tensor &input, const Tensor &output, const double angleDeg,
                          const double2 shift, const eInterpolationType interpolation, const eDeviceType device) {
    // clang-format off
    static const std::unordered_map<eInterpolationType,
                                    std::function<void(hipStream_t, const Tensor &, const Tensor &, const double,
                                                       const double2, const eDeviceType)>>
        funcs = {
            {eInterpolationType::INTERP_TYPE_NEAREST, dispatch_rotate_interp<T, eInterpolationType::INTERP_TYPE_NEAREST>},
            {eInterpolationType::INTERP_TYPE_LINEAR, dispatch_rotate_interp<T, eInterpolationType::INTERP_TYPE_LINEAR>}
        };
    // clang-format on

    if (!funcs.contains(interpolation)) {
        throw Exception("Not mapped to a defined function", eStatusType::INVALID_OPERATION);
    }
    auto func = funcs.at(interpolation);
    func(stream, input, output, angleDeg, shift, device);
}

void Rotate::operator()(hipStream_t stream, const Tensor &input, const Tensor &output, const double angleDeg,
                        const double2 shift, const eInterpolationType interpolation, const eDeviceType device) const {
    CHECK_TENSOR_DEVICE(input, device);
    CHECK_TENSOR_CHANNELS(input, 1, 3, 4);
    CHECK_TENSOR_DATATYPES(input, eDataType::DATA_TYPE_U8, eDataType::DATA_TYPE_S8, eDataType::DATA_TYPE_U16,
                           eDataType::DATA_TYPE_S16, eDataType::DATA_TYPE_U32, eDataType::DATA_TYPE_S32,
                           eDataType::DATA_TYPE_F32, eDataType::DATA_TYPE_F64);
    CHECK_TENSOR_LAYOUT(input, eTensorLayout::TENSOR_LAYOUT_HWC, eTensorLayout::TENSOR_LAYOUT_NHWC);

    CHECK_TENSOR_COMPARISON(input.layout() == output.layout());
    CHECK_TENSOR_COMPARISON(input.device() == output.device());
    CHECK_TENSOR_COMPARISON(input.dtype() == output.dtype());
    CHECK_TENSOR_COMPARISON(input.shape() == output.shape());

    // clang-format off
    static const std::unordered_map<
        eDataType, std::array<std::function<void(hipStream_t, const Tensor &, const Tensor &, const double,
                                                 const double2, const eInterpolationType, const eDeviceType)>,
                              4>>
        funcs = {
            {eDataType::DATA_TYPE_U8,  {dispatch_rotate_type<uchar1>, nullptr, dispatch_rotate_type<uchar3>, dispatch_rotate_type<uchar4>}},
            {eDataType::DATA_TYPE_S8,  {dispatch_rotate_type<char1>, nullptr, dispatch_rotate_type<char3>, dispatch_rotate_type<char4>}},
            {eDataType::DATA_TYPE_U16, {dispatch_rotate_type<ushort1>, nullptr, dispatch_rotate_type<ushort3>, dispatch_rotate_type<ushort4>}},
            {eDataType::DATA_TYPE_S16, {dispatch_rotate_type<short1>, nullptr, dispatch_rotate_type<short3>, dispatch_rotate_type<short4>}},
            {eDataType::DATA_TYPE_U32, {dispatch_rotate_type<uint1>, nullptr, dispatch_rotate_type<uint3>, dispatch_rotate_type<uint4>}},
            {eDataType::DATA_TYPE_S32, {dispatch_rotate_type<int1>, nullptr, dispatch_rotate_type<int3>, dispatch_rotate_type<int4>}},
            {eDataType::DATA_TYPE_F32, {dispatch_rotate_type<float1>, nullptr, dispatch_rotate_type<float3>, dispatch_rotate_type<float4>}},
            {eDataType::DATA_TYPE_F64, {dispatch_rotate_type<double1>, nullptr, dispatch_rotate_type<double3>, dispatch_rotate_type<double4>}}
        };
    // clang-format on

    eDataType dtype = input.dtype().etype();
    int channels = input.shape(input.layout().channels_index());
    if (!funcs.contains(dtype)) {
        throw Exception("Not mapped to a defined function", eStatusType::INVALID_OPERATION);
    }
    auto func = funcs.at(dtype).at(channels - 1);
    if (func == nullptr) {
        throw Exception("Not mapped to a defined function", eStatusType::INVALID_OPERATION);
    }

    func(stream, input, output, angleDeg, shift, interpolation, device);
}
}  // namespace roccv