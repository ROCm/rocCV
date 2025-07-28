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
#include "kernels/kernel_helpers.hpp"
#include "operator_types.h"

namespace Kernels {
namespace Host {
template <bool has_alpha, typename T, typename SRC, typename DST>
void bndbox_kernel(SRC input, DST output, Rect_t *rects,
                   size_t n_rects, int64_t batch, int64_t height,
                   int64_t width) {
    for (int64_t b_idx = 0; b_idx < batch; b_idx++) {
        for (int64_t y_idx = 0; y_idx < height; y_idx++) {
            for (int64_t x_idx = 0; x_idx < width; x_idx++) {
                uchar4 shaded_pixel{0, 0, 0, 0}; // Todo use template

                for (size_t i = 0; i < n_rects; i++) {
                    Rect_t curr_rect = rects[i];
                    if (curr_rect.batch <= b_idx)
                        shade_rectangle(curr_rect, x_idx, y_idx, &shaded_pixel);
                }

                uchar4 out_color = MathVector::fill(
                    input.at(b_idx, y_idx, x_idx, 0));
                out_color.w = has_alpha ? out_color.w : 255;

                if (shaded_pixel.w != 0)
                    blend_single_color(out_color, shaded_pixel);

                MathVector::trunc(
                    out_color, &output.at(b_idx, y_idx, x_idx, 0));
            }
        }
    }
}
};  // namespace Host
};  // namespace Kernels