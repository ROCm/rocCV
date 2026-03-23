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
#include "operator_types.h"

namespace Kernels {
namespace Host {

using namespace roccv::detail;
    
template <typename SrcWrapper, typename DstWrapper, typename MapWrapper>
void remap(SrcWrapper input, DstWrapper output, MapWrapper map, int mapBatchSize, RemapParams params) {
    
    float2 srcCoord = make_float2(0.f, 0.f);
    float2 mapCoord = make_float2(0.f, 0.f);
    float2 dstCoord = make_float2(0.f, 0.f);
    
    for (size_t b = 0; b < output.batches(); b++) {
        for (size_t y = 0; y < output.height(); y++) {
            for (size_t x = 0; x < output.width(); x++) {
                
                dstCoord.x = static_cast<float>(x);
                dstCoord.y = static_cast<float>(y);
                
                mapCoord.x = (dstCoord.x + params.dstOffset) * params.mapScale.x;
                mapCoord.y = (dstCoord.y + params.dstOffset) * params.mapScale.y;
                
                float2 mapValue = map.at((mapBatchSize == 1 ? 0 : b), mapCoord.y, mapCoord.x, 0);

                srcCoord.x = dstCoord.x * params.srcScale.x + mapValue.x * params.valScale.x + params.srcOffset.x;
                srcCoord.y = dstCoord.y * params.srcScale.y + mapValue.y * params.valScale.y + params.srcOffset.y;

                output.at(b, y, x, 0) = input.at(b, srcCoord.y, srcCoord.x, 0);
            }
        }
    }
}
} // namespace Host
} // namespace Kernels