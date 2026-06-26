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

#include "core/wrappers/interpolation_wrapper.hpp"

namespace Kernels {
namespace Device {

template <typename SrcWrapper, typename DstWrapper, typename Mat>
__global__ void warp_affine(SrcWrapper input, DstWrapper output, Mat mat) {
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    const int b = blockIdx.z;

    if (x >= output.width() || y >= output.height()) return;

    const float ox = mat[0] * static_cast<float>(x) + mat[1] * static_cast<float>(y) + mat[2];
    const float oy = mat[3] * static_cast<float>(x) + mat[4] * static_cast<float>(y) + mat[5];
    output.at(b, y, x, 0) = input.at(b, oy, ox, 0);
}
}  // namespace Device
}  // namespace Kernels
