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

#include "core/wrappers/image_wrapper.hpp"
#include "operator_types.h"

namespace Kernels::Host {

template <typename SrcWrapper, typename DstWrapper>
void custom_crop_kernel(SrcWrapper input, DstWrapper output, const int roi_x, const int roi_y, const int roi_w,
                        const int roi_h) {
    using work_type_t = typename DstWrapper::value_type;

#pragma omp parallel for
    for (int64_t b = 0; b < output.batches(); b++) {
        for (int64_t y = 0; y < output.height(); y++) {
            for (int64_t x = 0; x < output.width(); x++) {
                int src_y = roi_y + static_cast<int>(y);
                int src_x = roi_x + static_cast<int>(x);

                if (src_x >= 0 && src_x < roi_w && src_y >= 0 && src_y < roi_h) {
                    work_type_t in_val = input.at(b, src_y, src_x, 0);
                    output.at(b, y, x, 0) = in_val;
                }
            }
        }
    }
}

}  // namespace Kernels::Host
