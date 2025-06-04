/*
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

/**
 * @file normalize_device.hpp
 * @brief Contains the device kernel implementation for the Normalize operation.
 */

#pragma once

#include <hip/hip_runtime.h>

#include "core/detail/casting.hpp"
#include "core/detail/math/vectorized_type_math.hpp"
#include "core/detail/type_traits.hpp"
#include "core/detail/vector_utils.hpp"

namespace Kernels::Device {
template <bool ScaleStddev, typename SrcWrapper, typename DstWrapper, typename ScaleWrapper, typename BaseWrapper>
__global__ void normalize(SrcWrapper input, BaseWrapper base, ScaleWrapper scale, DstWrapper output, float globalScale,
                          float shift, float epsilon) {
    using namespace roccv::detail;
    using work_type = MakeType<float, NumComponents<typename SrcWrapper::ValueType>>;
    using result_type = DstWrapper::ValueType;

    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    const int b = blockIdx.z;

    if (x >= output.width() || y >= output.height()) return;

    const int baseBatchIdx = base.batches() == 1 ? 0 : b;
    const int baseHeightIdx = base.height() == 1 ? 0 : y;
    const int baseWidthIdx = base.width() == 1 ? 0 : x;

    const int scaleBatchIdx = scale.batches() == 1 ? 0 : b;
    const int scaleHeightIdx = scale.height() == 1 ? 0 : y;
    const int scaleWidthIdx = scale.width() == 1 ? 0 : x;

    work_type scaleVal;
    work_type s = StaticCast<work_type>(scale.at(scaleBatchIdx, scaleHeightIdx, scaleWidthIdx, 0));
    if constexpr (ScaleStddev) {
        // Scale tensor is the standard deviation, invert back to scale with epsilon added to avoid division by zero.
        scaleVal = math::vrsqrtf((s * s) + epsilon);
    } else {
        // Scale tensor remains normal, calculate assuming the values in the scale tensor are indeed the scale
        scaleVal = s;
    }
    work_type result = (StaticCast<work_type>(input.at(b, y, x, 0)) -
                        StaticCast<work_type>(base.at(baseBatchIdx, baseHeightIdx, baseWidthIdx, 0))) *
                           scaleVal * globalScale +
                       shift;

    // Saturate cast value back into the output tensor's value type
    output.at(b, y, x, 0) = SaturateCast<result_type>(result);
}
}  // namespace Kernels::Device