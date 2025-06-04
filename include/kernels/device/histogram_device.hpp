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
template <typename T, typename OUT, typename SRC, typename DST>
__global__ void histogram_kernel(SRC input, DST output, int64_t batch,
                                 int64_t height, int64_t width) {
    extern __shared__ __align__(sizeof(OUT)) unsigned char smem[];
    OUT *local_histogram = reinterpret_cast<OUT *>(smem);

    const auto z_idx = blockIdx.z;
    const auto gid = blockIdx.x * blockDim.x + threadIdx.x;
    const auto x_idx = gid % width;
    const auto y_idx = gid / width;

    // thread index in block
    const auto tid = threadIdx.x;  // histogram index

    local_histogram[tid] = 0;  // initialize the histogram

    __syncthreads();

    if (gid < height * width) {
        atomicAdd(
            &local_histogram[input.template at<T>(z_idx, y_idx, x_idx, 0)], 1);
    }
    __syncthreads();  // wait for all of the threads in this block to finish

    const auto hist_val =
        local_histogram[tid];  // get local value for this thread

    // this is the output histogram must be init to and atomicly added to.
    if (hist_val > 0) {
        atomicAdd(&output.template at<OUT>(0, z_idx, tid, 0), hist_val);
    }
}

template <typename T, typename OUT, typename SRC, typename DST, typename MASK>
__global__ void histogram_kernel(SRC input, DST output, MASK mask,
                                 int64_t batch, int64_t height, int64_t width) {
    extern __shared__ __align__(sizeof(OUT)) unsigned char smem[];
    OUT *local_histogram = reinterpret_cast<OUT *>(smem);

    const auto z_idx = blockIdx.z;
    const auto gid = blockIdx.x * blockDim.x + threadIdx.x;
    const auto x_idx = gid % width;
    const auto y_idx = gid / width;

    // thread index in block
    const auto tid = threadIdx.x;  // histogram index

    local_histogram[tid] = 0;  // initialize the histogram

    __syncthreads();

    if (gid < height * width) {
        if (mask.template at<T>(z_idx, y_idx, x_idx, 0)) {
            atomicAdd(
                &local_histogram[input.template at<T>(z_idx, y_idx, x_idx, 0)],
                1);
        }
    }
    __syncthreads();  // wait for all of the threads in this block to finish

    const auto hist_val =
        local_histogram[tid];  // get local value for this thread

    // this is the output histogram must be init to and atomicly added to.
    if (hist_val > 0) {
        atomicAdd(&output.template at<OUT>(0, z_idx, tid, 0), hist_val);
    }
}
}  // namespace Device
}  // namespace Kernels