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
 * @file normalize_host.hpp
 * @brief Contains the host kernel implementation for the Normalize operation.
 */

#pragma once

#include <hip/hip_runtime.h>

#include "core/detail/casting.hpp"
#include "core/detail/math/vectorized_type_math.hpp"
#include "core/detail/type_traits.hpp"
#include "core/detail/vector_utils.hpp"

namespace Kernels::Host {
template <bool ScaleStddev, typename SrcWrapper, typename DstWrapper, typename ScaleWrapper, typename BaseWrapper>
void normalize(SrcWrapper input, BaseWrapper base, ScaleWrapper scale, DstWrapper output, float globalScale,
               float shift, float epsilon) {
    using namespace roccv::detail;
    using work_type = MakeType<float, NumComponents<typename SrcWrapper::ValueType>>;
    using result_type = typename DstWrapper::ValueType;

    const int batches = static_cast<int>(output.batches());
    const int height = static_cast<int>(output.height());
    const int width = static_cast<int>(output.width());

    // base/scale are broadcastable along any of N/H/W (extent 1 means the index collapses to 0).
    const bool baseBroadcastN = base.batches() == 1;
    const bool baseBroadcastH = base.height() == 1;
    const bool baseBroadcastW = base.width() == 1;
    const bool scaleBroadcastN = scale.batches() == 1;
    const bool scaleBroadcastH = scale.height() == 1;
    const bool scaleBroadcastW = scale.width() == 1;

    // Convert a scale sample to a multiplier, inverting standard deviations (epsilon guards against /0).
    auto resolveScale = [&](const work_type& s) -> work_type {
        if constexpr (ScaleStddev) {
            return 1.0f / (math::vsqrtf((s * s) + epsilon));
        } else {
            return s;
        }
    };

    // Collapse batch and row so work scales even when batch == 1 (HWC); rows are uniform, so schedule statically.
#pragma omp parallel for collapse(2) schedule(static)
    for (int b = 0; b < batches; b++) {
        for (int y = 0; y < height; y++) {
            const int baseBatchIdx = baseBroadcastN ? 0 : b;
            const int baseHeightIdx = baseBroadcastH ? 0 : y;
            const int scaleBatchIdx = scaleBroadcastN ? 0 : b;
            const int scaleHeightIdx = scaleBroadcastH ? 0 : y;

            // If base/scale don't vary across the row, resolve them once per row (hoists the stddev sqrt).
            work_type rowScale{};
            work_type rowBase{};
            if (scaleBroadcastW)
                rowScale = resolveScale(StaticCast<work_type>(scale.at(scaleBatchIdx, scaleHeightIdx, 0, 0)));
            if (baseBroadcastW) rowBase = StaticCast<work_type>(base.at(baseBatchIdx, baseHeightIdx, 0, 0));

            // Fast path: packed rows with row-constant base/scale become a unit-stride, vectorizable loop.
            if (scaleBroadcastW && baseBroadcastW && input.isRowContiguous() && output.isRowContiguous()) {
                const typename SrcWrapper::ValueType* __restrict__ inRow = &input.at(b, y, 0, 0);
                result_type* __restrict__ outRow = &output.at(b, y, 0, 0);
                for (int x = 0; x < width; x++) {
                    work_type result = (StaticCast<work_type>(inRow[x]) - rowBase) * rowScale * globalScale + shift;
                    outRow[x] = SaturateCast<result_type>(result);
                }
            } else {
                for (int x = 0; x < width; x++) {
                    const work_type scaleVal =
                        scaleBroadcastW
                            ? rowScale
                            : resolveScale(StaticCast<work_type>(scale.at(scaleBatchIdx, scaleHeightIdx, x, 0)));
                    const work_type baseVal =
                        baseBroadcastW ? rowBase : StaticCast<work_type>(base.at(baseBatchIdx, baseHeightIdx, x, 0));

                    work_type result =
                        (StaticCast<work_type>(input.at(b, y, x, 0)) - baseVal) * scaleVal * globalScale + shift;
                    output.at(b, y, x, 0) = SaturateCast<result_type>(result);
                }
            }
        }
    }
}
}  // namespace Kernels::Host