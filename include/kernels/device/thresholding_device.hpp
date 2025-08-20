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

#pragma once

#include <hip/hip_runtime.h>
#include <core/wrappers/generic_tensor_wrapper.hpp>
#include "core/detail/type_traits.hpp"
#include "core/detail/casting.hpp"
#include "operator_types.h"

namespace Kernels {
namespace Device {
template <typename SrcWrapper, typename DstWrapper>
__global__ void binary_generic(SrcWrapper input, DstWrapper output,
                                roccv::GenericTensorWrapper<double> thresh,
                                roccv::GenericTensorWrapper<double> maxVal,
                                const int32_t maxBatchSize) {
    using namespace roccv::detail;
    const auto x_idx = threadIdx.x + blockIdx.x * blockDim.x;
    const auto y_idx = threadIdx.y + blockIdx.y * blockDim.y;
    const auto z_idx = threadIdx.z + blockIdx.z * blockDim.z;
    
    using src_type = typename SrcWrapper::ValueType;
    using dst_type = typename DstWrapper::ValueType;
    using base_type = BaseType<dst_type>;

    if (x_idx < output.width() && y_idx < output.height() && z_idx < maxBatchSize) {
        double th = thresh.at(z_idx);
        double mv = maxVal.at(z_idx);
        src_type inputVal = input.at(z_idx, y_idx, x_idx, 0);
        dst_type outputVal;
#pragma unroll
        for (int i = 0; i < output.channels(); i++) {
            double ip = StaticCast<double>(GetElement(inputVal, i));
            double outVal = ip > th ? mv : 0;
            GetElement(outputVal, i) = StaticCast<base_type>(outVal);
        }
        output.at(z_idx, y_idx, x_idx, 0) = outputVal;
    }
}

template <typename SrcWrapper, typename DstWrapper>
__global__ void binary_inv_generic(SrcWrapper input, DstWrapper output,
                                    roccv::GenericTensorWrapper<double> thresh,
                                    roccv::GenericTensorWrapper<double> maxVal,
                                    const int32_t maxBatchSize) {
    using namespace roccv::detail;
    const auto x_idx = threadIdx.x + blockIdx.x * blockDim.x;
    const auto y_idx = threadIdx.y + blockIdx.y * blockDim.y;
    const auto z_idx = threadIdx.z + blockIdx.z * blockDim.z;
    
    using src_type = typename SrcWrapper::ValueType;
    using dst_type = typename DstWrapper::ValueType;
    using base_type = BaseType<dst_type>;

    if (x_idx < output.width() && y_idx < output.height() && z_idx < maxBatchSize) {
        double th = thresh.at(z_idx);
        double mv = maxVal.at(z_idx);
        src_type inputVal = input.at(z_idx, y_idx, x_idx, 0);
        dst_type outputVal;
#pragma unroll
        for (int i = 0; i < output.channels(); i++) {
            double ip = StaticCast<double>(GetElement(inputVal, i));
            double outVal = ip > th ? 0 : mv;
            GetElement(outputVal, i) = StaticCast<base_type>(outVal);
        }
        output.at(z_idx, y_idx, x_idx, 0) = outputVal;
    }
}

template <typename SrcWrapper, typename DstWrapper>
__global__ void trunc_generic(SrcWrapper input, DstWrapper output,
                                roccv::GenericTensorWrapper<double> thresh,
                                const int32_t maxBatchSize) {
    using namespace roccv::detail;
    const auto x_idx = threadIdx.x + blockIdx.x * blockDim.x;
    const auto y_idx = threadIdx.y + blockIdx.y * blockDim.y;
    const auto z_idx = threadIdx.z + blockIdx.z * blockDim.z;
    
    using src_type = typename SrcWrapper::ValueType;
    using dst_type = typename DstWrapper::ValueType;
    using base_type = BaseType<dst_type>;

    if (x_idx < output.width() && y_idx < output.height() && z_idx < maxBatchSize) {
        double th = thresh.at(z_idx);
        src_type inputVal = input.at(z_idx, y_idx, x_idx, 0);
        dst_type outputVal;
#pragma unroll
        for (int i = 0; i < output.channels(); i++) {
            double ip = StaticCast<double>(GetElement(inputVal, i));
            double outVal = ip > th ? th : ip;
            GetElement(outputVal, i) = StaticCast<base_type>(outVal);
        }
        output.at(z_idx, y_idx, x_idx, 0) = outputVal;
    }
}

template <typename SrcWrapper, typename DstWrapper>
__global__ void tozero_generic(SrcWrapper input, DstWrapper output,
                                roccv::GenericTensorWrapper<double> thresh,
                                const int32_t maxBatchSize) {
    using namespace roccv::detail;
    const auto x_idx = threadIdx.x + blockIdx.x * blockDim.x;
    const auto y_idx = threadIdx.y + blockIdx.y * blockDim.y;
    const auto z_idx = threadIdx.z + blockIdx.z * blockDim.z;
    
    using src_type = typename SrcWrapper::ValueType;
    using dst_type = typename DstWrapper::ValueType;
    using base_type = BaseType<dst_type>;

    if (x_idx < output.width() && y_idx < output.height() && z_idx < maxBatchSize) {
        double th = thresh.at(z_idx);
        src_type inputVal = input.at(z_idx, y_idx, x_idx, 0);
        dst_type outputVal;
#pragma unroll
        for (int i = 0; i < output.channels(); i++) {
            double ip = StaticCast<double>(GetElement(inputVal, i));
            double outVal = ip > th ? ip : 0;
            GetElement(outputVal, i) = StaticCast<base_type>(outVal);
        }
        output.at(z_idx, y_idx, x_idx, 0) = outputVal;
    }
}

template <typename SrcWrapper, typename DstWrapper>
__global__ void tozeroinv_generic(SrcWrapper input, DstWrapper output,
                                    roccv::GenericTensorWrapper<double> thresh,
                                    const int32_t maxBatchSize) {
    using namespace roccv::detail;
    const auto x_idx = threadIdx.x + blockIdx.x * blockDim.x;
    const auto y_idx = threadIdx.y + blockIdx.y * blockDim.y;
    const auto z_idx = threadIdx.z + blockIdx.z * blockDim.z;
    
    using src_type = typename SrcWrapper::ValueType;
    using dst_type = typename DstWrapper::ValueType;
    using base_type = BaseType<dst_type>;

    if (x_idx >= output.width() || y_idx >= output.height()) return;
    
    double th = thresh.at(z_idx);
    src_type inputVal = input.at(z_idx, y_idx, x_idx, 0);
    dst_type outputVal;
#pragma unroll
    for (int i = 0; i < output.channels(); i++) {
        double ip = StaticCast<double>(GetElement(inputVal, i));
        double outVal = ip > th ? 0 : ip;
        GetElement(outputVal, i) = StaticCast<base_type>(outVal);
    }
    output.at(z_idx, y_idx, x_idx, 0) = outputVal;
}
}   // namespace Device
}   // namespace Kernels