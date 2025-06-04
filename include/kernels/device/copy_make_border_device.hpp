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

namespace Kernels {
namespace Device {
/**
 * @brief GPU kernel for CopyMakeBorder operator.
 *
 * @tparam SrcDesc Must be a BorderWrapper.
 * @tparam DstDesc Must be a ImageWrapper.
 * @param src A BorderWrapper containing information for the input tensor.
 * @param dst A ImageWrapper containing information for the output tensor.
 * @param top The top pixel coordinate on the y-axis where the border should start.
 * @param left The left-most pixel coordinate on the x-axis where the border should start.
 * @return __global__
 */
template <typename SrcDesc, typename DstDesc>
__global__ void copy_make_border(SrcDesc src, DstDesc dst, int32_t top, int32_t left) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int b = blockIdx.z;

    if (x >= dst.width() || y >= dst.height() || b >= dst.batches()) return;

    dst.at(b, y, x, 0) = src.at(b, y - top, x - left, 0);
}
}  // namespace Device
}  // namespace Kernels