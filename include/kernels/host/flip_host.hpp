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

#include <cstring>

#include "operator_types.h"

namespace Kernels::Host {
template <eAxis FlipType, typename SrcWrapper, typename DstWrapper>
void flip(SrcWrapper input, DstWrapper output) {
    using T = typename DstWrapper::ValueType;
    const int width = static_cast<int>(output.width());
    const int height = static_cast<int>(output.height());

#pragma omp parallel for collapse(2) schedule(static)
    for (int b = 0; b < output.batches(); b++) {
        for (int y = 0; y < height; y++) {
            // For an X/BOTH flip the source row is mirrored vertically; otherwise it matches the output row.
            const int srcY = (FlipType == eAxis::X || FlipType == eAxis::BOTH) ? height - y - 1 : y;
            const T* __restrict__ inRow = &input.at(b, srcY, 0, 0);
            T* __restrict__ outRow = &output.at(b, y, 0, 0);

            if constexpr (FlipType == eAxis::X) {
                // No horizontal change: copy the contiguous row wholesale.
                std::memcpy(outRow, inRow, static_cast<size_t>(width) * sizeof(T));
            } else {
                // Y or BOTH: mirror the row horizontally.
                for (int x = 0; x < width; x++) {
                    outRow[x] = inRow[width - 1 - x];
                }
            }
        }
    }
}
}  // namespace Kernels::Host