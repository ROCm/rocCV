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

namespace Kernels::Host {
template <typename SrcWrapper, typename DstWrapper>
void resize(SrcWrapper input, DstWrapper output, float scaleX, float scaleY) {
#pragma omp parallel for
    for (int batch = 0; batch < output.batches(); batch++) {
        for (int y = 0; y < output.height(); y++) {
            for (int x = 0; x < output.width(); x++) {
                float srcX = fminf(fmaf(x + 0.5f, scaleX, -0.5f), input.width() - 1);
                float srcY = fminf(fmaf(y + 0.5f, scaleY, -0.5f), input.height() - 1);
                output.at(batch, y, x, 0) = input.at(batch, srcY, srcX, 0);
            }
        }
    }
}
}  // namespace Kernels::Host