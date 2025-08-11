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

namespace Kernels::Host {
template <typename SrcWrapper, typename DstWrapper, typename MatWrapper>
void rotate(SrcWrapper src, DstWrapper dst, MatWrapper affineMat) {
#pragma omp parallel for
    for (int b = 0; b < dst.batches(); b++) {
        for (int y = 0; y < dst.height(); y++) {
            for (int x = 0; x < dst.width(); x++) {
                const auto xShift = x - affineMat[2];
                const auto yShift = y - affineMat[5];

                const float srcX = static_cast<float>(xShift * affineMat[0] + yShift * -affineMat[1]);
                const float srcY = static_cast<float>(xShift * -affineMat[3] + yShift * affineMat[4]);

                dst.at(b, y, x, 0) = src.at(b, srcY, srcX, 0);
            }
        }
    }
}
}  // namespace Kernels::Host