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

#pragma once

#include <hip/hip_runtime.h>

#include "core/detail/casting.hpp"

namespace Kernels {
namespace Device {
template <typename SrcWrapper, typename MaskWrapper, typename DstWrapper>
__global__ void composite(SrcWrapper foreground, SrcWrapper background, MaskWrapper mask, DstWrapper output) {
    using namespace roccv::detail;  // For RangeCast, NumElements, etc.
    using src_type = typename SrcWrapper::ValueType;
    using dst_type = typename DstWrapper::ValueType;
    using work_type = MakeType<float, NumElements<src_type>>;

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch = blockIdx.z;

    if (x >= foreground.width() || y >= foreground.height()) return;

    // Range cast all input values to float to avoid overflowing values and keep them in the same range.
    auto maskFactor = RangeCast<float1>(mask.at(batch, y, x, 0));
    auto fgVal = RangeCast<work_type>(foreground.at(batch, y, x, 0));
    auto bgVal = RangeCast<work_type>(background.at(batch, y, x, 0));

    work_type result = bgVal + maskFactor.x * (fgVal - bgVal);

    // If number of channels in output is 4, ensure that the last channel (alpha in this case) is always fully on.
    if constexpr (NumElements<dst_type> == 4) {
        output.at(batch, y, x, 0) = RangeCast<dst_type>((MakeType<float, 4>){result.x, result.y, result.z, 1.0f});
    } else {
        output.at(batch, y, x, 0) = RangeCast<dst_type>(result);
    }
}
}  // namespace Device
}  // namespace Kernels