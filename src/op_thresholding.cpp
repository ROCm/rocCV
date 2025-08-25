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
#include "op_thresholding.hpp"

#include <hip/hip_runtime.h>

#include <functional>
#include "common/strided_data_wrap.hpp"
#include "common/validation_helpers.hpp"
#include "core/wrappers/image_wrapper.hpp"
#include "core/wrappers/generic_tensor_wrapper.hpp"
#include "core/exception.hpp"
#include "core/status_type.h"
#include "kernels/device/thresholding_device.hpp"
#include "kernels/host/thresholding_host.hpp"

namespace roccv {
Threshold::Threshold(eThresholdType threshType, int32_t maxBatchSize)
    : m_threshType(threshType), m_maxBatchSize(maxBatchSize) {}

Threshold::~Threshold() {}

template <typename T>
void dispatch_threshold_dtype(hipStream_t stream, const Tensor &input, const Tensor &output,
                                    const Tensor &thresh, const Tensor &maxVal, eThresholdType m_threshType, int32_t m_maxBatchSize, eDeviceType device) {
    ImageWrapper<T> inputWrapper(input);
    ImageWrapper<T> outputWrapper(output);

    const auto height = input.shape()[input.shape().layout().height_index()];
    const auto width = input.shape()[input.shape().layout().width_index()];
    
    if (device == eDeviceType::GPU) {
        dim3 block(64, 16);
        dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y, m_maxBatchSize);
        
        switch (m_threshType) {
            case THRESH_BINARY:
                Kernels::Device::binary_generic<<<grid, block, 0, stream>>>(inputWrapper, outputWrapper, 
                                    GenericTensorWrapper<double>(thresh), GenericTensorWrapper<double>(maxVal), m_maxBatchSize);
                break;
            case THRESH_BINARY_INV:
                Kernels::Device::binary_inv_generic<<<grid, block, 0, stream>>>(inputWrapper, outputWrapper, 
                                    GenericTensorWrapper<double>(thresh), GenericTensorWrapper<double>(maxVal), m_maxBatchSize);
                break;
            case THRESH_TRUNC:
                Kernels::Device::trunc_generic<<<grid, block, 0, stream>>>(inputWrapper, outputWrapper, 
                                    GenericTensorWrapper<double>(thresh), m_maxBatchSize);
                break;
            case THRESH_TOZERO:
                Kernels::Device::tozero_generic<<<grid, block, 0, stream>>>(inputWrapper, outputWrapper, 
                                    GenericTensorWrapper<double>(thresh), m_maxBatchSize);
                break;
            case THRESH_TOZERO_INV:
                Kernels::Device::tozeroinv_generic<<<grid, block, 0, stream>>>(inputWrapper, outputWrapper, 
                                    GenericTensorWrapper<double>(thresh), m_maxBatchSize);
                break;
        }
        
    } else if (device == eDeviceType::CPU) {
        
        switch (m_threshType) {
            case THRESH_BINARY:
                Kernels::Host::binary_generic(inputWrapper, outputWrapper, GenericTensorWrapper<double>(thresh), 
                                GenericTensorWrapper<double>(maxVal), m_maxBatchSize);
                break;
            case THRESH_BINARY_INV:
                Kernels::Host::binary_inv_generic(inputWrapper, outputWrapper, GenericTensorWrapper<double>(thresh), 
                                GenericTensorWrapper<double>(maxVal), m_maxBatchSize);
                break;       
            case THRESH_TRUNC:
                Kernels::Host::trunc_generic(inputWrapper, outputWrapper, GenericTensorWrapper<double>(thresh), m_maxBatchSize);
                break;
            case THRESH_TOZERO:
                Kernels::Host::tozero_generic(inputWrapper, outputWrapper, GenericTensorWrapper<double>(thresh), m_maxBatchSize);
                break;
            case THRESH_TOZERO_INV:
                Kernels::Host::tozeroinv_generic(inputWrapper, outputWrapper, GenericTensorWrapper<double>(thresh), m_maxBatchSize);
                break;
        }
    }
}

void Threshold::operator()(hipStream_t stream, const Tensor &input, const Tensor &output,
                           const Tensor &thresh, const Tensor &maxVal, eDeviceType device) {
    // Verify that the tensors are located on the right device (CPU or GPU).
    CHECK_TENSOR_DEVICE(input, device);
    CHECK_TENSOR_DEVICE(output, device);

    // Ensure all tensors are using supported datatypes
    CHECK_TENSOR_DATATYPES(input, eDataType::DATA_TYPE_U8, eDataType::DATA_TYPE_U16, eDataType::DATA_TYPE_S16, eDataType::DATA_TYPE_F32, eDataType::DATA_TYPE_F64);
    CHECK_TENSOR_DATATYPES(output, eDataType::DATA_TYPE_U8, eDataType::DATA_TYPE_U16, eDataType::DATA_TYPE_S16, eDataType::DATA_TYPE_F32, eDataType::DATA_TYPE_F64);
    CHECK_TENSOR_DATATYPES(thresh, eDataType::DATA_TYPE_F64);
    CHECK_TENSOR_DATATYPES(maxVal, eDataType::DATA_TYPE_F64);

    // Ensure all tensors are using supported layouts.
    CHECK_TENSOR_LAYOUT(input, eTensorLayout::TENSOR_LAYOUT_NHWC, eTensorLayout::TENSOR_LAYOUT_HWC);
    CHECK_TENSOR_LAYOUT(output, eTensorLayout::TENSOR_LAYOUT_NHWC, eTensorLayout::TENSOR_LAYOUT_HWC);
    CHECK_TENSOR_LAYOUT(thresh, eTensorLayout::TENSOR_LAYOUT_N);
    CHECK_TENSOR_LAYOUT(maxVal, eTensorLayout::TENSOR_LAYOUT_N);

    CHECK_TENSOR_CHANNELS(input, 1, 3, 4);

    // Ensure the layout and shapes for the input/output tensor match
    CHECK_TENSOR_COMPARISON(input.layout() == output.layout());
    CHECK_TENSOR_COMPARISON(input.shape() == output.shape());
    CHECK_TENSOR_COMPARISON(input.dtype() == output.dtype());
    CHECK_TENSOR_COMPARISON(output.shape(output.layout().batch_index()) == input.shape(input.layout().batch_index()));
    CHECK_TENSOR_COMPARISON(output.shape(output.layout().channels_index()) == input.shape(input.layout().channels_index()));
    CHECK_TENSOR_COMPARISON(output.shape(output.layout().width_index()) == input.shape(input.layout().width_index()));
    CHECK_TENSOR_COMPARISON(output.shape(output.layout().height_index()) == input.shape(input.layout().height_index()));
    CHECK_TENSOR_COMPARISON(m_maxBatchSize == input.shape(input.layout().batch_index()));

    // Select kernel dispatcher based on number of channels and a base datatype.
    // clang-format off
    static const std::unordered_map<
    eDataType, std::array<std::function<void(hipStream_t, const Tensor &, const Tensor &, const Tensor &, const Tensor &, eThresholdType, int32_t, const eDeviceType)>, 4>>
        funcs = 
        {
            {eDataType::DATA_TYPE_U8, {dispatch_threshold_dtype<uchar1>, 0, dispatch_threshold_dtype<uchar3>, dispatch_threshold_dtype<uchar4>}},
            {eDataType::DATA_TYPE_U16, {dispatch_threshold_dtype<ushort1>, 0, dispatch_threshold_dtype<ushort3>, dispatch_threshold_dtype<ushort4>}},
            {eDataType::DATA_TYPE_S16, {dispatch_threshold_dtype<short1>, 0, dispatch_threshold_dtype<short3>, dispatch_threshold_dtype<short4>}},
            {eDataType::DATA_TYPE_F32, {dispatch_threshold_dtype<float1>, 0, dispatch_threshold_dtype<float3>, dispatch_threshold_dtype<float4>}},
            {eDataType::DATA_TYPE_F64, {dispatch_threshold_dtype<double1>, 0, dispatch_threshold_dtype<double3>, dispatch_threshold_dtype<double4>}}
        };
    // clang-format on

    auto func = funcs.at(input.dtype().etype())[input.shape(input.layout().channels_index()) - 1];
    func(stream, input, output, thresh, maxVal, m_threshType, m_maxBatchSize, device);
}
}  // namespace roccv