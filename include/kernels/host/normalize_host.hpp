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

    // The base/scale tensors are broadcastable: any of their N/H/W extents may be 1, in which case that index is
    // shared across the entire corresponding output dimension. Whether a dimension broadcasts is a property of the
    // tensor shapes, not of the current pixel, so resolve these flags once instead of re-testing them per pixel.
    const bool baseBroadcastN = base.batches() == 1;
    const bool baseBroadcastH = base.height() == 1;
    const bool baseBroadcastW = base.width() == 1;
    const bool scaleBroadcastN = scale.batches() == 1;
    const bool scaleBroadcastH = scale.height() == 1;
    const bool scaleBroadcastW = scale.width() == 1;

    // Turns a raw scale-tensor sample into the multiplicative scale. When the tensor holds standard deviations we
    // invert back to a scale, adding epsilon under the root to guard against division by zero.
    auto resolveScale = [&](const work_type& s) -> work_type {
        if constexpr (ScaleStddev) {
            return 1.0f / (math::vsqrtf((s * s) + epsilon));
        } else {
            return s;
        }
    };

    // Collapse the batch and row loops into one iteration space. Parallelizing over the batch alone leaves all but
    // one thread idle for single-image (HWC, batch == 1) inputs; collapsing exposes batches * height independent
    // rows so the work scales regardless of batch size. Each row does an identical amount of work, so static
    // scheduling partitions the iteration space up front and avoids the bookkeeping of dynamic scheduling. Strides
    // are honored throughout by going through the wrapper's at() accessor rather than assuming a contiguous layout.
#pragma omp parallel for collapse(2) schedule(static)
    for (int b = 0; b < batches; b++) {
        for (int y = 0; y < height; y++) {
            // Indices into the base/scale tensors only depend on b/y here, so resolve them once per row.
            const int baseBatchIdx = baseBroadcastN ? 0 : b;
            const int baseHeightIdx = baseBroadcastH ? 0 : y;
            const int scaleBatchIdx = scaleBroadcastN ? 0 : b;
            const int scaleHeightIdx = scaleBroadcastH ? 0 : y;

            // When base/scale broadcast across the width, their values are constant for the whole row. Hoist them
            // out of the x loop so the common per-channel parameter case (shape (1,1,1,C)) computes the base sample
            // and the (potentially sqrt-heavy) scale exactly once per row instead of once per pixel.
            work_type rowScale{};
            work_type rowBase{};
            if (scaleBroadcastW)
                rowScale = resolveScale(StaticCast<work_type>(scale.at(scaleBatchIdx, scaleHeightIdx, 0, 0)));
            if (baseBroadcastW) rowBase = StaticCast<work_type>(base.at(baseBatchIdx, baseHeightIdx, 0, 0));

            for (int x = 0; x < width; x++) {
                const work_type scaleVal =
                    scaleBroadcastW
                        ? rowScale
                        : resolveScale(StaticCast<work_type>(scale.at(scaleBatchIdx, scaleHeightIdx, x, 0)));
                const work_type baseVal =
                    baseBroadcastW ? rowBase : StaticCast<work_type>(base.at(baseBatchIdx, baseHeightIdx, x, 0));

                work_type result =
                    (StaticCast<work_type>(input.at(b, y, x, 0)) - baseVal) * scaleVal * globalScale + shift;

                // Saturate cast value back into the output tensor's value type
                output.at(b, y, x, 0) = SaturateCast<result_type>(result);
            }
        }
    }
}
}  // namespace Kernels::Host