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

#include "kernels/device/packed_apply.hpp"

namespace Kernels::Device {

/**
 * @brief Resizes an image using the interpolation/border logic baked into the source wrapper.
 *
 * Each thread emits a contiguous run of PackWidth output pixels along x via the shared packed-write helper. Reads and
 * interpolation are unchanged from the scalar path, so results are bit-identical.
 */
template <typename SrcWrapper, typename DstWrapper>
__global__ void resize(SrcWrapper input, DstWrapper output, float scaleX, float scaleY) {
    using T = typename DstWrapper::ValueType;

    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch = blockIdx.z;
    if (y >= output.height() || batch >= output.batches()) return;

    const float srcY = fmaf(y + 0.5f, scaleY, -0.5f);

    ApplyPackedGather(output, batch, y, [=] __device__(int n, int /*y*/, int x) -> T {
        const float srcX = fmaf(x + 0.5f, scaleX, -0.5f);
        return input.at(n, srcY, srcX, 0);
    });
}
}  // namespace Kernels::Device