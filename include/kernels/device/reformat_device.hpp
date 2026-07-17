/*
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include <hip/hip_runtime.h>

#include "core/wrappers/tensor_wrapper.hpp"

namespace Kernels::Device {

/**
 * @brief Device kernel for reformatting a tensor from one layout to another.
 *
 * @tparam T The datatype of the tensor.
 * @tparam Channels The number of channels in the tensor.
 * @param[in] input The input tensor.
 * @param[out] output The output tensor.
 */
template <int Channels, typename T>
__global__ void reformat(roccv::TensorWrapper<T> input, roccv::TensorWrapper<T> output) {
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    const int b = blockIdx.z;

    if (x >= output.width() || y >= output.height()) return;

#pragma unroll
    for (int c = 0; c < Channels; c++) {
        output.at(b, y, x, c) = input.at(b, y, x, c);
    }
}
}  // namespace Kernels::Device