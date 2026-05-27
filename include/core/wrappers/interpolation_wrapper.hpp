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

#pragma once

#include "core/detail/casting.hpp"
#include "core/detail/sampling_helpers.hpp"
#include "core/detail/vector_utils.hpp"
#include "core/wrappers/border_wrapper.hpp"
#include "operator_types.h"

namespace roccv {
/**
 * @brief A kernel-friendly wrapper which provides interpolation logic on top of a BorderWrapper.
 *
 * Templated directly on the BorderWrapper type so the redundant border-mode and underlying-wrapper template
 * parameters need only be spelled once (in the BorderWrapper type). Recover the border mode via
 * BW::kBorderType and the underlying wrapper type via BW::WrapperType.
 *
 * Read-only access; do not use for output tensors.
 *
 * @tparam I  Interpolation type to use.
 * @tparam BW The BorderWrapper type to wrap. Must expose ValueType plus at(n,h,w,c), width(n), height(n).
 */
template <eInterpolationType I, typename BW>
class InterpolationWrapper {
   public:
    using ValueType = typename BW::ValueType;
    using BorderType = BW;
    static constexpr eInterpolationType kInterpolationType = I;

    /**
     * @brief Wraps a BorderWrapper in an InterpolationWrapper. Extends capabilities to interpolate pixel values
     * when given non-integer coordinates.
     *
     * @param borderWrapper The BorderWrapper to wrap.
     */
    InterpolationWrapper(BW borderWrapper) : m_desc(borderWrapper) {}

    /**
     * @brief This function calculates the weighting coefficients for the Catmull-Rom cubic interpolation.
     * @param dist The distance between the current point and the previous data point.
     * @param weight The pointer to the array of weights.
     * @return None.
     */
    __device__ __host__ inline void CalBicubicWeights(float dist, float* weight) const {
        const float d = dist;
        // Fused multiply-add: single rounding vs separate mul+add (matches kernels/device style).
        weight[0] = fmaf(fmaf(fmaf(-0.5f, d, 1.0f), d, -0.5f), d, 0.f);
        weight[1] = fmaf(fmaf(fmaf(1.5f, d, -2.5f), d, 0.f), d, 1.0f);
        weight[2] = fmaf(fmaf(fmaf(-1.5f, d, 2.f), d, 0.5f), d, 0.f);
        weight[3] = 1.f - weight[0] - weight[1] - weight[2];
    }

    /**
     * @brief Retrieves an interpolated value at given image batch coordinates.
     *
     * @param n Batch index.
     * @param h Height coordinates.
     * @param w Width coordinates.
     * @return An interpolated value.
     */
    inline __device__ __host__ ValueType at(int64_t n, float h, float w, int64_t c) const {
        if constexpr (I == eInterpolationType::INTERP_TYPE_NEAREST) {
            return m_desc.at(n, detail::interp_nearest_i64(h), detail::interp_nearest_i64(w), c);
        } else if constexpr (I == eInterpolationType::INTERP_TYPE_LINEAR) {
            // Bilinear interpolation implementation
            // v1 -- v2
            // -     -
            // v3 -- v4

            using WorkType = detail::MakeType<float, detail::NumElements<ValueType>>;

            const int64_t x0 = detail::interp_floor_i64(w);
            const int64_t y0 = detail::interp_floor_i64(h);
            const int64_t x1 = x0 + 1;
            const int64_t y1 = y0 + 1;
            const float fx = w - static_cast<float>(x0);
            const float fy = h - static_cast<float>(y0);
            const float omfx = 1.f - fx;
            const float omfy = 1.f - fy;

            if (x0 >= 0 && y0 >= 0 && x1 < m_desc.width() && y1 < m_desc.height()) {
                auto v1 = detail::RangeCast<WorkType>(m_desc.at_inbounds(n, y0, x0, c));
                auto v2 = detail::RangeCast<WorkType>(m_desc.at_inbounds(n, y0, x1, c));
                auto v3 = detail::RangeCast<WorkType>(m_desc.at_inbounds(n, y1, x0, c));
                auto v4 = detail::RangeCast<WorkType>(m_desc.at_inbounds(n, y1, x1, c));
                auto q1 = v1 * omfx + v2 * fx;
                auto q2 = v3 * omfx + v4 * fx;
                auto q = q1 * omfy + q2 * fy;
                return detail::RangeCast<ValueType>(q);
            }

            auto v1 = detail::RangeCast<WorkType>(m_desc.at(n, y0, x0, c));
            auto v2 = detail::RangeCast<WorkType>(m_desc.at(n, y0, x1, c));
            auto v3 = detail::RangeCast<WorkType>(m_desc.at(n, y1, x0, c));
            auto v4 = detail::RangeCast<WorkType>(m_desc.at(n, y1, x1, c));

            auto q1 = v1 * omfx + v2 * fx;
            auto q2 = v3 * omfx + v4 * fx;
            auto q = q1 * omfy + q2 * fy;

            return detail::RangeCast<ValueType>(q);
        } else if constexpr (I == eInterpolationType::INTERP_TYPE_CUBIC) {
            using namespace roccv::detail;
            using WorkType = detail::MakeType<float, detail::NumElements<ValueType>>;

            const int64_t int_x = detail::interp_floor_i64(w);
            const int64_t int_y = detail::interp_floor_i64(h);

            float weight_x[4], weight_y[4];
            CalBicubicWeights(w - static_cast<float>(int_x), weight_x);
            CalBicubicWeights(h - static_cast<float>(int_y), weight_y);

            float wxy[16];
            int k = 0;
#pragma unroll
            for (int j = 0; j < 4; j++) {
#pragma unroll
                for (int i = 0; i < 4; i++) {
                    wxy[k++] = weight_y[j] * weight_x[i];
                }
            }

            WorkType sum = SetAll<WorkType>(0.0f);
            const bool cubic_fast =
                int_x >= 1 && int_y >= 1 && (int_x + 2) < m_desc.width() && (int_y + 2) < m_desc.height();
            k = 0;
            if (cubic_fast) {
#pragma unroll
                for (int index_y = -1; index_y <= 2; index_y++) {
#pragma unroll
                    for (int index_x = -1; index_x <= 2; index_x++) {
                        sum = sum +
                              detail::RangeCast<WorkType>(m_desc.at_inbounds(n, int_y + index_y, int_x + index_x, c)) *
                                  wxy[k++];
                    }
                }
            } else {
#pragma unroll
                for (int index_y = -1; index_y <= 2; index_y++) {
#pragma unroll
                    for (int index_x = -1; index_x <= 2; index_x++) {
                        sum = sum +
                              detail::RangeCast<WorkType>(m_desc.at(n, int_y + index_y, int_x + index_x, c)) * wxy[k++];
                    }
                }
            }

            return detail::RangeCast<ValueType>(sum);
        }
    }

    __device__ __host__ inline int64_t height(int64_t n = 0) const { return m_desc.height(n); }
    __device__ __host__ inline int64_t width(int64_t n = 0) const { return m_desc.width(n); }
    __device__ __host__ inline int64_t batches() const { return m_desc.batches(); }
    __device__ __host__ inline int64_t channels() const { return m_desc.channels(); }

   private:
    BW m_desc;
};

/**
 * @brief Factory for InterpolationWrapper. Deduces the BorderWrapper type BW (and its border mode +
 * underlying wrapper) from the argument; callers only need to spell the interpolation policy.
 *
 * @tparam I The interpolation type.
 * @param borderWrap An already-constructed BorderWrapper (typically via MakeBorderWrapper<B>(...)).
 */
template <eInterpolationType I, typename BW>
auto MakeInterpolationWrapper(BW borderWrap) {
    return InterpolationWrapper<I, BW>(borderWrap);
}

}  // namespace roccv