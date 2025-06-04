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
template <typename T, typename SourceWrapper, typename DestWrapper>
void custom_crop(SourceWrapper input, DestWrapper output, Box_t cropRect,
                 size_t channels, size_t batches) {
    for (int b = 0; b < batches; b++) {
        for (int i = 0; i < cropRect.width; i++) {
            for (int j = 0; j < cropRect.height; j++) {
                for (int k = 0; k < channels; k++) {
                    int sourceX = i + cropRect.x;
                    int sourceY = j + cropRect.y;
                    int sourceZ = k;
                    output.template at<T>(b, j, i, k) =
                        input.template at<T>(b, sourceY, sourceX, sourceZ);
                }
            }
        }
    }
}
}  // namespace Host
}  // namespace Kernels