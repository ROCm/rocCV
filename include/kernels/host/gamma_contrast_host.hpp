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
template <typename T, typename SRC, typename DST>
void gamma_contrast_wrapped_u8(SRC input, DST output, int batch, int width,
                              int height, float *gamma) {
    for (int b = 0; b < batch; b++) {
        float gamma_val = gamma[b];
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                output.template at<T>(b, i, j, 0) =
                    MathVector::convert_base<uchar>(
                        MathVector::pow(MathVector::convert_base<double>(input.template at<T>(b, i, j, 0)) / 255.0, gamma_val) * 255.0);
            }
        }
    }
}
}  // namespace Host
}  // namespace Kernels