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
#include "kernels/device/packed_apply.hpp"

namespace Kernels {
namespace Device {
template <typename SrcWrapper, typename MaskWrapper, typename DstWrapper>
__global__ void composite(SrcWrapper foreground, SrcWrapper background, MaskWrapper mask, DstWrapper output) {
    using namespace roccv::detail;  // For RangeCast, NumElements, etc.
    using src_type = typename SrcWrapper::ValueType;
    using dst_type = typename DstWrapper::ValueType;
    using work_type = MakeType<float, NumElements<src_type>>;

    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch = blockIdx.z;
    if (y >= foreground.height() || batch >= output.batches()) return;

    ApplyPackedGather(output, batch, y, [=] __device__(int n, int yy, int x) -> dst_type {
        // The blend is linear, so only the mask needs normalizing; fg/bg stay at native scale (no scaling multiply).
        auto maskFactor = RangeCast<float1>(mask.at(n, yy, x, 0));
        auto fgVal = StaticCast<work_type>(foreground.at(n, yy, x, 0));
        auto bgVal = StaticCast<work_type>(background.at(n, yy, x, 0));

        work_type result = bgVal + maskFactor.x * (fgVal - bgVal);

        // For 4-channel output, force the alpha channel fully on.
        if constexpr (NumElements<dst_type> == 4) {
            return SaturateCast<dst_type>((MakeType<float, 4>){result.x, result.y, result.z, RangeMax<dst_type>()});
        } else {
            return SaturateCast<dst_type>(result);
        }
    });
}
}  // namespace Device
}  // namespace Kernels