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

#include "operator_types.h"

namespace Kernels::Host {
template <eAxis FlipType, typename SrcWrapper, typename DstWrapper>
void flip(SrcWrapper input, DstWrapper output) {
    for (int b = 0; b < output.batches(); b++) {
        for (int y = 0; y < output.height(); y++) {
            for (int x = 0; x < output.width(); x++) {
                int srcX = x;
                int srcY = y;
                if constexpr (FlipType == eAxis::Y || FlipType == eAxis::BOTH) {
                    // Flip along y-axis (horizontally)
                    srcX = output.width() - x - 1;
                }

                if constexpr (FlipType == eAxis::X || FlipType == eAxis::BOTH) {
                    // Flip along x-axis (vertically)
                    srcY = output.height() - y - 1;
                }

                output.at(b, y, x, 0) = input.at(b, srcY, srcX, 0);
            }
        }
    }
}
}  // namespace Kernels::Host