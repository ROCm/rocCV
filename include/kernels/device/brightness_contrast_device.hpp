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
#include "core/wrappers/image_wrapper.hpp"
#include "kernels/device/packed_apply.hpp"

namespace Kernels {
namespace Device {
template <typename SrcWrapper, typename DstWrapper, typename BCWrappers>
__global__ void brightness_contrast(SrcWrapper input, DstWrapper output, BCWrappers bc_wrappers) {
    using namespace roccv::detail;
    using src_type = typename SrcWrapper::ValueType;
    using dst_type = typename DstWrapper::ValueType;
    using bc_type = typename BCWrappers::ValueType;
    using work_type = MakeType<bc_type, NumElements<dst_type>>;

    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int batch = blockIdx.z;
    if (y >= output.height() || batch >= output.batches()) return;

    // The brightness/contrast scalars depend only on the batch, so fetch them once per thread (outside the x-run).
    const bc_type brightness = bc_wrappers.brightnessWrapper.at(batch);
    const bc_type contrast = bc_wrappers.contrastWrapper.at(batch);
    const bc_type brightnessShift = bc_wrappers.brightnessShiftWrapper.at(batch);
    const bc_type contrastCenter = bc_wrappers.contrastCenterWrapper.at(batch);

    ApplyPackedTransform(
        output, batch, y,
        [=] __device__(int /*n*/, int /*yy*/, int /*x*/, src_type srcPixel) -> dst_type {
            work_type src_val = StaticCast<work_type>(srcPixel);
            work_type result = brightnessShift + brightness * (contrastCenter + contrast * (src_val - contrastCenter));
            return SaturateCast<dst_type>(result);
        },
        input);
}
}  // namespace Device
}  // namespace Kernels