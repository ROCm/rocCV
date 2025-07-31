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
#include "core/detail/math/vectorized_type_math.hpp"
#include "core/wrappers/border_wrapper.hpp"
#include "operator_types.h"

namespace roccv {

/**
 * @brief A kernel-friendly wrapper which provides interpolation logic based on the given
 * coordinates. This tensor wrapper is typically only used for input tensors and does not provide write access to its
 * underlying data.
 *
 * @tparam T Underlying data type of the tensor data.
 * @tparam C Number of channels in data type.
 * @tparam B Border type to use for interpolation.
 * @tparam I Interpolation type to use.
 */
template <typename T, eBorderType B, eInterpolationType I>
class InterpolationWrapper {
   public:
    /**
     * @brief Wraps a roccv::Tensor in an InterpolationWrapper to provide pixel interpolation when accessing
     * non-integer coordinate mappings.
     *
     * @param tensor The tensor to wrap.
     * @param border_value A fallback border value to use in the case of a constant border mode.
     */
    InterpolationWrapper(const Tensor& tensor, T border_value) : m_desc(tensor, border_value) {}

    /**
     * @brief Wraps a BorderWrapper in an Interpolation wrapper. Extends capabilities to interpolate pixel values when
     * given non-integer coordinates.
     *
     * @param borderWrapper The BorderWrapper to wrap.
     */
    InterpolationWrapper(BorderWrapper<T, B> borderWrapper) : m_desc(borderWrapper) {}

    /**
     * @brief Retrieves an interpolated value at given image batch coordinates.
     *
     * @param n Batch index.
     * @param h Height coordinates.
     * @param w Width coordinates.
     * @return An interpolated value.
     */
    inline __device__ __host__ const T at(int64_t n, float h, float w, int64_t c) const {
        if constexpr (I == eInterpolationType::INTERP_TYPE_NEAREST) {
            // Nearest neighbor interpolation implementation
            return m_desc.at(n, static_cast<int64_t>(std::round(h)), static_cast<int64_t>(std::round(w)), c);
        } else if constexpr (I == eInterpolationType::INTERP_TYPE_LINEAR) {
            // Bilinear interpolation implementation
            // v1 -- v2
            // -     -
            // v3 -- v4

            using WorkType = detail::MakeType<float, detail::NumElements<T>>;

            int64_t x0 = static_cast<int64_t>(floorf(w));
            int64_t x1 = x0 + 1;
            int64_t y0 = static_cast<int64_t>(floorf(h));
            int64_t y1 = y0 + 1;

            auto v1 = detail::RangeCast<WorkType>(m_desc.at(n, y0, x0, c));
            auto v2 = detail::RangeCast<WorkType>(m_desc.at(n, y0, x1, c));
            auto v3 = detail::RangeCast<WorkType>(m_desc.at(n, y1, x0, c));
            auto v4 = detail::RangeCast<WorkType>(m_desc.at(n, y1, x1, c));

            auto q1 = v1 * (x1 - w) + v2 * (w - x0);
            auto q2 = v3 * (x1 - w) + v4 * (w - x0);
            auto q = q1 * (y1 - h) + q2 * (h - y0);

            return detail::RangeCast<T>(q);
        }

        // TODO: Support other interpolation methods.

        else {
            static_assert(false, "Provided interpolation type is not supported.");
        }
    }

    __device__ __host__ inline int64_t height() const { return m_desc.height(); }
    __device__ __host__ inline int64_t width() const { return m_desc.width(); }
    __device__ __host__ inline int64_t batches() const { return m_desc.batches(); }
    __device__ __host__ inline int64_t channels() const { return m_desc.channels(); }

   private:
    BorderWrapper<T, B> m_desc;
};
}  // namespace roccv