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

#include "operator_types.h"

namespace Kernels {
namespace Host {
template <typename T, typename OUT, typename SRC, typename DST>
void histogram_kernel(SRC input, DST output, int64_t batch, int64_t height,
                      int64_t width) {
    for (int64_t b_idx = 0; b_idx < batch; b_idx++) {
        for (int64_t y_idx = 0; y_idx < height; y_idx++) {
            for (int64_t x_idx = 0; x_idx < width; x_idx++) {
                auto hist_idx = input.template at<T>(b_idx, y_idx, x_idx, 0);
                output.template at<OUT>(0, b_idx, hist_idx, 0) += 1;
            }
        }
    }
}

template <typename T, typename OUT, typename SRC, typename DST, typename MASK>
void histogram_kernel(SRC input, DST output, MASK mask, int64_t batch,
                      int64_t height, int64_t width) {
    for (int64_t b_idx = 0; b_idx < batch; b_idx++) {
        for (int64_t y_idx = 0; y_idx < height; y_idx++) {
            for (int64_t x_idx = 0; x_idx < width; x_idx++) {
                if (mask.template at<T>(b_idx, y_idx, x_idx, 0)) {
                    auto hist_idx =
                        input.template at<T>(b_idx, y_idx, x_idx, 0);
                    output.template at<OUT>(0, b_idx, hist_idx, 0) += 1;
                }
            }
        }
    }
}
}  // namespace Host
}  // namespace Kernels