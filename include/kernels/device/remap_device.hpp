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

#include "core/detail/internal_structs.hpp"
#include "kernels/device/packed_apply.hpp"
#include "operator_types.h"

namespace Kernels {
namespace Device {

using namespace roccv::detail;

template <typename SrcWrapper, typename DstWrapper, typename MapWrapper>
__global__ void remap(SrcWrapper input, DstWrapper output, MapWrapper map, int mapBatchSize, RemapParams params) {
    using dst_type = typename DstWrapper::ValueType;

    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    const int b = blockIdx.z;
    if (y >= output.height() || b >= output.batches()) return;

    ApplyPackedGather(output, b, y, [=] __device__(int n, int yy, int x) -> dst_type {
        float2 dstCoord = make_float2(static_cast<float>(x), static_cast<float>(yy));

        float2 mapCoord;
        mapCoord.x = (dstCoord.x + params.dstOffset) * params.mapScale.x;
        mapCoord.y = (dstCoord.y + params.dstOffset) * params.mapScale.y;

        float2 mapValue = map.at((mapBatchSize == 1 ? 0 : n), mapCoord.y, mapCoord.x, 0);

        float2 srcCoord;
        srcCoord.x = dstCoord.x * params.srcScale.x + mapValue.x * params.valScale.x + params.srcOffset.x;
        srcCoord.y = dstCoord.y * params.srcScale.y + mapValue.y * params.valScale.y + params.srcOffset.y;

        return input.at(n, srcCoord.y, srcCoord.x, 0);
    });
}
};  // namespace Device
};  // namespace Kernels