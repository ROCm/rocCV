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
#include "core/detail/math/vectorized_type_math.hpp"
#include "core/detail/casting.hpp"

#include "operator_types.h"

namespace Kernels {
namespace Host {
template <typename SrcWrapper, typename DstWrapper>
void gamma_contrast(SrcWrapper input, DstWrapper output, float gamma) {
    using namespace roccv::detail;  // For RangeCast, NumElements, etc.
    using src_type = typename SrcWrapper::ValueType;
    using dst_type = typename DstWrapper::ValueType;
    using work_type = MakeType<float, NumElements<src_type>>;
    
    for (int batch = 0; batch < output.batches(); batch++) {
#pragma omp parallel for
        for (int y = 0; y < output.height(); y++) {
            for (int x = 0; x < output.width(); x++) {
                auto inVal = (RangeCast<work_type>(input.at(batch, y, x, 0)));
                work_type result = math::vpowf(inVal, gamma);
                if constexpr (NumElements<dst_type> == 4) {
                    output.at(batch, y, x, 0) = RangeCast<dst_type>((MakeType<float, 4>){result.x, result.y, result.z, inVal.w});
                } else {
                    output.at(batch, y, x, 0) = RangeCast<dst_type>(result);
                }
            }
        }
    }
}
}  // namespace Host
}  // namespace Kernels