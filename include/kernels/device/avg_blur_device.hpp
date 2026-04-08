/**
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

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

#include "core/detail/casting.hpp"
#include "core/detail/type_traits.hpp"
#include "core/detail/vector_utils.hpp"
#include "operator_types.h"

namespace Kernels {
namespace Device {

template <typename T, typename SrcWrapper, typename DstWrapper>
__global__ void avg_blur_2d(SrcWrapper input, DstWrapper output,
                            int kernelWidth, int kernelHeight,
                            int kernelAnchorX, int kernelAnchorY) {
    using namespace roccv::detail;
    using WorkType = MakeType<float, NumElements<T>>;

    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    const int b = blockIdx.z;

    if (x >= output.width() || y >= output.height() || b >= output.batches()) {
        return;
    }

    // Initialize accumulator
    WorkType sum = SetAll<WorkType>(0.0f);

    // Compute the sum over the kernel window
    for (int ky = 0; ky < kernelHeight; ++ky) {
        int srcY = y - kernelAnchorY + ky;

        for (int kx = 0; kx < kernelWidth; ++kx) {
            int srcX = x - kernelAnchorX + kx;

            // Read pixel through border wrapper (handles out-of-bounds)
            T pixel = input.at(b, srcY, srcX, 0);

            // Accumulate as float to avoid overflow
            sum = sum + StaticCast<WorkType>(pixel);
        }
    }

    // Compute average by dividing by kernel area
    float kernelArea = static_cast<float>(kernelWidth * kernelHeight);
    WorkType average = sum / kernelArea;

    // Write result with saturation
    output.at(b, y, x, 0) = SaturateCast<T>(average);
}

/**
 * @brief Optimized 1D Horizontal Average Blur Kernel with Shared Memory Tiling
 *
 * First pass of separable average blur. Applies horizontal averaging using shared
 * memory to reduce global memory bandwidth. Each block loads a tile of input data
 * into shared memory, then each thread computes its output using only shared memory.
 */
template <typename T, int BLOCK_WIDTH, typename SrcWrapper, typename DstWrapper>
__global__ void avg_blur_horizontal(SrcWrapper input, DstWrapper output,
                                    int kernelWidth, int kernelAnchorX) {
    using namespace roccv::detail;
    using WorkType = MakeType<float, NumElements<T>>;

    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockIdx.y;
    const int b = blockIdx.z;

    // Check bounds for loading - all threads in block participate
    const bool validBlock = (y < output.height() && b < output.batches());

    // Shared memory tile: BLOCK_WIDTH + kernel halo on both sides
    extern __shared__ char smem[];
    T* tile = reinterpret_cast<T*>(smem);

    const int halo = kernelWidth - 1;
    const int tileWidth = BLOCK_WIDTH + halo;

    // Load data into shared memory with halo
    // All threads participate in loading to avoid divergence before syncthreads
    for (int i = threadIdx.x; i < tileWidth; i += blockDim.x) {
        int srcX = blockIdx.x * BLOCK_WIDTH + i - kernelAnchorX;
        if (validBlock) {
            tile[i] = input.at(b, y, srcX, 0);
        }
    }

    __syncthreads();

    // Now check if this specific thread has valid output to compute
    if (x >= output.width() || !validBlock) {
        return;
    }

    // Compute horizontal average using shared memory
    WorkType sum = SetAll<WorkType>(0.0f);

    int tileIdx = threadIdx.x + kernelAnchorX;
    for (int kx = 0; kx < kernelWidth; ++kx) {
        sum = sum + StaticCast<WorkType>(tile[tileIdx - kernelAnchorX + kx]);
    }

    float kernelSize = static_cast<float>(kernelWidth);
    WorkType average = sum / kernelSize;

    output.at(b, y, x, 0) = SaturateCast<T>(average);
}

/**
 * @brief Optimized 1D Vertical Average Blur Kernel with Shared Memory Tiling
 *
 * Second pass of separable average blur. Applies vertical averaging using shared
 * memory to reduce global memory bandwidth. Each block loads a column tile into
 * shared memory, then computes vertical averages.
 */
template <typename T, int BLOCK_HEIGHT, typename SrcWrapper, typename DstWrapper>
__global__ void avg_blur_vertical(SrcWrapper input, DstWrapper output,
                                  int kernelHeight, int kernelAnchorY) {
    using namespace roccv::detail;
    using WorkType = MakeType<float, NumElements<T>>;

    const int x = blockIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    const int b = blockIdx.z;

    // Check bounds for loading - all threads in block participate
    const bool validBlock = (x < output.width() && b < output.batches());

    // Shared memory tile: BLOCK_HEIGHT + kernel halo on both sides
    extern __shared__ char smem[];
    T* tile = reinterpret_cast<T*>(smem);

    const int halo = kernelHeight - 1;
    const int tileHeight = BLOCK_HEIGHT + halo;

    // Load data into shared memory with halo
    // All threads participate in loading to avoid divergence before syncthreads
    for (int i = threadIdx.y; i < tileHeight; i += blockDim.y) {
        int srcY = blockIdx.y * BLOCK_HEIGHT + i - kernelAnchorY;
        if (validBlock) {
            tile[i] = input.at(b, srcY, x, 0);
        }
    }

    __syncthreads();

    // Now check if this specific thread has valid output to compute
    if (y >= output.height() || !validBlock) {
        return;
    }

    // Compute vertical average using shared memory
    WorkType sum = SetAll<WorkType>(0.0f);

    int tileIdx = threadIdx.y + kernelAnchorY;
    for (int ky = 0; ky < kernelHeight; ++ky) {
        sum = sum + StaticCast<WorkType>(tile[tileIdx - kernelAnchorY + ky]);
    }

    float kernelSize = static_cast<float>(kernelHeight);
    WorkType average = sum / kernelSize;

    output.at(b, y, x, 0) = SaturateCast<T>(average);
}

}  // namespace Device
}  // namespace Kernels
