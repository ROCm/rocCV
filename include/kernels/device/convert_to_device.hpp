/**
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

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
#include "core/detail/casting.hpp"
#include "core/detail/type_traits.hpp"
#include "core/wrappers/image_wrapper.hpp"

namespace Kernels {
namespace Device {
template <typename SrcWrapper, typename DstWrapper, typename DT_AB>
__global__ void convert_to(SrcWrapper input, DstWrapper output, DT_AB alpha, DT_AB beta) {
    using namespace roccv::detail;  // For RangeCast, NumElements, etc.
    using dst_type = typename DstWrapper::ValueType;

    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int batch = blockIdx.z;

    if (x >= output.width() || y >= output.height() || batch >= output.batches()) return;

    output.at(batch, y, x, 0) = SaturateCast<dst_type>(alpha * (input.at(batch, y, x, 0)) + beta);

}
}   // namespace Device
}   // namespace Kernels