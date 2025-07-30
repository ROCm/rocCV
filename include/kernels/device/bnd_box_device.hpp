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
#include <core/detail/type_traits.hpp>
#include "kernels/kernel_helpers.hpp"
#include "operator_types.h"

using namespace roccv;

namespace Kernels {
namespace Device {
template <bool has_alpha, typename T, typename BT = detail::BaseType<T>, typename SRC, typename DST>
__global__ void bndbox_kernel(SRC input, DST output, Rect_t *rects,
                              size_t n_rects, int64_t batch, int64_t height,
                              int64_t width) {
    // Working type for internal pixel format, which has 4 channels.
    using WorkType = detail::MakeType<BT, 4>;

                                const auto x_idx = threadIdx.x + blockIdx.x * blockDim.x;
    const auto y_idx = threadIdx.y + blockIdx.y * blockDim.y;
    const auto b_idx = threadIdx.z + blockIdx.z * blockDim.z;

    if (x_idx >= width || y_idx >= height || b_idx >= batch) {
        return;
    }

    WorkType shaded_pixel{0, 0, 0, 0};

    for (size_t i = 0; i < n_rects; i++) {
        Rect_t curr_rect = rects[i];
        if (curr_rect.batch <= b_idx)
            shade_rectangle<WorkType>(curr_rect, x_idx, y_idx, &shaded_pixel);
    }

    WorkType out_color =
        MathVector::fill(input.at(b_idx, y_idx, x_idx, 0));
    out_color.w = has_alpha ? out_color.w : (std::numeric_limits<BT>::max());

    if (shaded_pixel.w != 0) blend_single_color<WorkType>(out_color, shaded_pixel);

    MathVector::trunc(out_color,
                      &output.at(b_idx, y_idx, x_idx, 0));
}
};  // namespace Device
};  // namespace Kernels