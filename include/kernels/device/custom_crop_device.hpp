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

#include "kernels/device/packed_apply.hpp"
#include "operator_types.h"

namespace Kernels {
namespace Device {
template <typename SrcWrapper, typename DstWrapper>
__global__ void custom_crop(SrcWrapper input, DstWrapper output, roccv::Box_t cropRect) {
    using dst_type = typename DstWrapper::ValueType;

    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int b = blockIdx.z;
    if (y >= cropRect.height || b >= output.batches()) return;

    ApplyPackedRow(output, b, y, [=] __device__(int n, int yy, int x) -> dst_type {
        return input.at(n, yy + cropRect.y, x + cropRect.x, 0);
    });
}
}  // namespace Device
}  // namespace Kernels