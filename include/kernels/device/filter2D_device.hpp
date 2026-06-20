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
#include "core/wrappers/image_wrapper.hpp"

namespace Kernels {
namespace Device {
template <typename SrcWrapper, typename DstWrapper, typename KernelWrapper>
__global__ void filter2D(SrcWrapper input, DstWrapper output, KernelWrapper kernel, int kernelWidth, int kernelHeight,
                         int anchorX, int anchorY) {
    using namespace roccv::detail;
    using dst_type = typename DstWrapper::ValueType;
    using work_type = MakeType<float, NumElements<dst_type>>;
    work_type result = SetAll<work_type>(0);

    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int batch = blockIdx.z;

    if (x >= output.width() || y >= output.height()) return;

    // kernel is vector or 1d in row major order
    int kernelIndex = 0;
    int3 coord{x, y, batch};

    for (int i = 0; i < kernelHeight; ++i) {
        coord.y = y - anchorY + i;
        for (int j = 0; j < kernelWidth; ++j) {
            coord.x = x - anchorX + j;
            result = result + StaticCast<work_type>(input.at(coord.z, coord.y, coord.x, 0)) * kernel[kernelIndex++];
        }
    }
    output.at(batch, y, x, 0) = SaturateCast<dst_type>(result);
}

template <typename T, int BLOCK_WIDTH, typename SrcWrapper, typename DstWrapper, typename KernelWrapper>
__global__ void filter2D_horizontal(SrcWrapper input, DstWrapper output, KernelWrapper kernel, int kernelWidth,
                                    int anchorX) {
    using namespace roccv::detail;
    using work_type = MakeType<float, NumElements<T>>;
    work_type result = SetAll<work_type>(0);

    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = blockIdx.y;
    const int b = blockIdx.z;

    // smem size is BLOCK_WIDTH + halo on each side
    extern __shared__ char smem[];
    T* tile = reinterpret_cast<T*>(smem);

    const int halo = kernelWidth - 1;
    const int tileWidth = BLOCK_WIDTH + halo;

    // load into shared memory
    for (int i = threadIdx.x; i < tileWidth; i += blockDim.x) {
        int srcX = blockIdx.x * BLOCK_WIDTH + i - anchorX;
        tile[i] = input.at(b, y, srcX, 0);
    }

    __syncthreads();

    if (x >= output.width()) {
        return;
    }

    int tileIdx = threadIdx.x + anchorX;
    for (int kx = 0; kx < kernelWidth; ++kx) {
        result = result + StaticCast<work_type>(tile[tileIdx - anchorX + kx]) * kernel[kx];
    }

    output.at(b, y, x, 0) = SaturateCast<T>(result);
}

template <typename T, int BLOCK_HEIGHT, typename SrcWrapper, typename DstWrapper, typename KernelWrapper>
__global__ void filter2D_vertical(SrcWrapper input, DstWrapper output, KernelWrapper kernel, int kernelHeight,
                                    int anchorY) {
    using namespace roccv::detail;
    using work_type = MakeType<float, NumElements<T>>;
    work_type result = SetAll<work_type>(0);

    const int x = blockIdx.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int b = blockIdx.z;

    // smem size is BLOCK_HEIGHT + halo on each side
    extern __shared__ char smem[];
    T* tile = reinterpret_cast<T*>(smem);

    const int halo = kernelHeight - 1;
    const int tileHeight = BLOCK_HEIGHT + halo;

    // load into shared memory
    for (int i = threadIdx.y; i < tileHeight; i += blockDim.y) {
        int srcY = blockIdx.y * BLOCK_HEIGHT + i - anchorY;
        tile[i] = input.at(b, srcY, x, 0);
    }

    __syncthreads();

    if (y >= output.height()) {
        return;
    }

    int tileIdx = threadIdx.y + anchorY;
    for (int ky = 0; ky < kernelHeight; ++ky) {
        result = result + StaticCast<work_type>(tile[tileIdx - anchorY + ky]) * kernel[ky];
    }

    output.at(b, y, x, 0) = SaturateCast<T>(result);
}
}  // namespace Device
}  // namespace Kernels