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
namespace Device {
template <typename T, typename SRC, typename DST>
__global__ void binary_generic_kernel(SRC input, DST output, int64_t height,
                                      int64_t width, int64_t channels,
                                      const uint8_t *thresh,
                                      const uint8_t *maxVal,
                                      const int64_t batch_size) {
    const auto x_idx = threadIdx.x + blockIdx.x * blockDim.x;
    const auto y_idx = threadIdx.y + blockIdx.y * blockDim.y;
    const auto z_idx = threadIdx.z + blockIdx.z * blockDim.z;
    if (x_idx < width && y_idx < height && z_idx < batch_size) {
        uint8_t th = thresh[z_idx];
        uint8_t mv = maxVal[z_idx];
#pragma unroll
        for (int i = 0; i < channels; i++) {
            auto ip = input.template at<T>(z_idx, y_idx, x_idx, i);
            output.template at<T>(z_idx, y_idx, x_idx, i) = ip > th ? mv : 0;
        }
    }
}

template <typename T, typename SRC, typename DST>
__global__ void binary_inv_generic_kernel(SRC input, DST output, int64_t height,
                                          int64_t width, int64_t channels,
                                          const uint8_t *thresh,
                                          const uint8_t *maxVal,
                                          const int64_t batch_size) {
    const auto x_idx = threadIdx.x + blockIdx.x * blockDim.x;
    const auto y_idx = threadIdx.y + blockIdx.y * blockDim.y;
    const auto z_idx = threadIdx.z + blockIdx.z * blockDim.z;
    if (x_idx < width && y_idx < height && z_idx < batch_size) {
        uint8_t th = thresh[z_idx];
        uint8_t mv = maxVal[z_idx];
#pragma unroll
        for (int i = 0; i < channels; i++) {
            auto ip = input.template at<T>(z_idx, y_idx, x_idx, i);
            output.template at<T>(z_idx, y_idx, x_idx, i) = ip > th ? 0 : mv;
        }
    }
}

template <typename T, typename SRC, typename DST>
__global__ void trunc_generic_kernel(SRC input, DST output, int64_t height,
                                     int64_t width, int64_t channels,
                                     const uint8_t *thresh,
                                     const uint8_t *maxVal,
                                     const int64_t batch_size) {
    const auto x_idx = threadIdx.x + blockIdx.x * blockDim.x;
    const auto y_idx = threadIdx.y + blockIdx.y * blockDim.y;
    const auto z_idx = threadIdx.z + blockIdx.z * blockDim.z;
    if (x_idx < width && y_idx < height && z_idx < batch_size) {
        uint8_t th = thresh[z_idx];
#pragma unroll
        for (int i = 0; i < channels; i++) {
            auto ip = input.template at<T>(z_idx, y_idx, x_idx, i);
            output.template at<T>(z_idx, y_idx, x_idx, i) = ip > th ? th : ip;
        }
    }
}

template <typename T, typename SRC, typename DST>
__global__ void tozero_generic_kernel(SRC input, DST output, int64_t height,
                                      int64_t width, int64_t channels,
                                      const uint8_t *thresh,
                                      const uint8_t *maxVal,
                                      const int64_t batch_size) {
    const auto x_idx = threadIdx.x + blockIdx.x * blockDim.x;
    const auto y_idx = threadIdx.y + blockIdx.y * blockDim.y;
    const auto z_idx = threadIdx.z + blockIdx.z * blockDim.z;
    if (x_idx >= width || y_idx >= height || z_idx >= batch_size)
        return;
    uint8_t th = thresh[z_idx];
#pragma unroll
    for (int i = 0; i < channels; i++) {
        auto ip = input.template at<T>(z_idx, y_idx, x_idx, i);
        output.template at<T>(z_idx, y_idx, x_idx, i) = ip > th ? ip : 0;
    }
}

template <typename T, typename SRC, typename DST>
__global__ void tozeroinv_generic_kernel(SRC input, DST output, int64_t height,
                                         int64_t width, int64_t channels,
                                         const uint8_t *thresh,
                                         const uint8_t *maxVal,
                                         const int64_t batch_size) {
    const auto x_idx = threadIdx.x + blockIdx.x * blockDim.x;
    const auto y_idx = threadIdx.y + blockIdx.y * blockDim.y;
    const auto z_idx = threadIdx.z + blockIdx.z * blockDim.z;
    if (x_idx >= width || y_idx >= height || z_idx >= batch_size)
        return;
    uint8_t th = thresh[z_idx];
#pragma unroll
    for (int i = 0; i < channels; i++) {
        auto ip = input.template at<T>(z_idx, y_idx, x_idx, i);
        output.template at<T>(z_idx, y_idx, x_idx, i) = ip > th ? 0 : ip;
    }
}
}  // namespace Device
}   // namespace Kernels