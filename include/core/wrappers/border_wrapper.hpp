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

#include "core/detail/sampling_helpers.hpp"
#include "core/wrappers/image_wrapper.hpp"
#include "operator_types.h"

namespace roccv {

namespace detail {
/**
 * @brief Map one axis coordinate for OpenCV-style @c BORDER_REFLECT (edge pixels duplicated; not @c BORDER_REFLECT101).
 * @param coord Possibly out-of-bounds coordinate along the axis (width or height index space).
 * @param extent Positive extent of the axis (number of samples, e.g. image width or height).
 * @return In-bounds index in <tt>[0, extent)</tt> after reflection.
 * @note Period is <tt>2 * extent</tt>. Implementation uses Euclidean modulo then
 *       <tt>min(val, 2*extent - 1 - val)</tt>, which matches comparing @c val to @c extent with a ternary, without a
 *       separate branch on @p extent alone. On device, a 32-bit remainder path is used when @p extent and @p coord are
 *       in a safe range.
 */
__device__ __host__ inline int64_t reflect_border_coord_i64(int64_t coord, int64_t extent) {
#if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
    constexpr int64_t kLim = int64_t{1} << 30;
    if (extent > 0 && extent < kLim && coord > -kLim && coord < kLim) {
        const int32_t e = static_cast<int32_t>(extent);
        const int32_t scale = e * 2;
        int32_t val = static_cast<int32_t>(coord) % scale;
        if (val < 0) val += scale;
        const int32_t inv = scale - 1 - val;
        return static_cast<int64_t>(min_i32(val, inv));
    }
#endif
    const int64_t scale = extent * 2;
    const int64_t val = euclid_mod_i64(coord, scale);
    const int64_t inv = scale - 1 - val;
    return min_i64(val, inv);
}

/**
 * @brief Map one axis coordinate for OpenCV-style @c BORDER_REFLECT101 (endpoints are not repeated in the reflection).
 * @param coord Possibly out-of-bounds coordinate along the axis.
 * @param extent Positive extent of the axis (number of samples). If @p extent is at most 1, returns @c 0.
 * @return In-bounds index in <tt>[0, extent)</tt> after reflection.
 * @note Period is <tt>2 * extent - 2</tt> when @p extent is greater than 1. Uses Euclidean modulo then folds with
 *       <tt>(extent - 1) - abs((extent - 1) - v)</tt>. On device, a 32-bit path applies when values fit a fixed bound.
 */
__device__ __host__ inline int64_t reflect101_border_coord_i64(int64_t coord, int64_t extent) {
    if (extent <= 1) {
        return 0;
    }
    const int64_t scale = 2 * extent - 2;
#if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
    constexpr int64_t kLim = int64_t{1} << 30;
    if (extent < kLim && coord > -kLim && coord < kLim && scale > 0 && scale < kLim) {
        const int32_t e = static_cast<int32_t>(extent);
        const int32_t s = e * 2 - 2;
        int32_t v = euclid_mod_i32(static_cast<int32_t>(coord), s);
        const int32_t inner = (e - 1) - v;
        const int32_t a = abs_i32(inner);
        return static_cast<int64_t>((e - 1) - a);
    }
#endif
    const int64_t v = euclid_mod_i64_fast(coord, scale);
    const int64_t inner = (extent - 1) - v;
    return (extent - 1) - abs_i64(inner);
}
}  // namespace detail

/**
 * @brief Wrapper class which adds border-handling behavior on top of an underlying image wrapper.
 *
 * Templated on the wrapper type W (e.g. ImageWrapper<T>, VarShapeImageWrapper<T>) so the same border math
 * serves both uniform-shape and variable-shape image batches. The pixel value type T is recovered from
 * W::ValueType. W must expose: ValueType, at(n,h,w,c), width(n), height(n), batches(), channels().
 *
 * @tparam BorderType The border type to use when coordinates are out of bounds.
 * @tparam W          The underlying image wrapper type.
 */
template <eBorderType BorderType, typename W>
class BorderWrapper {
   public:
    using ValueType = typename W::ValueType;

    /**
     * @brief Constructs a BorderWrapper from an existing image wrapper. Extends its capabilities to handle out of
     * bound coordinates.
     *
     * @param image_wrapper The image wrapper to wrap.
     * @param border_value  The fallback border color to use when using a constant border mode.
     */
    BorderWrapper(W image_wrapper, ValueType border_value) : m_desc(image_wrapper), m_border_value(border_value) {}

    /**
     * @brief Sample the underlying image with no border logic. Caller must ensure coordinates are in-range.
     */
    __device__ __host__ inline const ValueType at_inbounds(int64_t n, int64_t h, int64_t w, int64_t c) const {
        return m_desc.at(n, h, w, c);
    }

    /**
     * @brief Returns a reference to the underlying data given image coordinates. If the coordinates fall out of bounds,
     * a fallback reference based on the provided border type will be given instead.
     *
     * @param n The batch index.
     * @param h The height index.
     * @param w The width index.
     * @param c The channel index.
     * @return A reference to the underlying data or a fallback border value of type T.
     */
    __device__ __host__ const ValueType at(int64_t n, int64_t h, int64_t w, int64_t c) const {
        const int64_t imgWidth = width(n);
        const int64_t imgHeight = height(n);

        // Constant border type implementation. This is a special case which doesn't remap values, but rather returns
        // the provided constant value.
        if constexpr (BorderType == eBorderType::BORDER_TYPE_CONSTANT) {
            if (w < 0 || w >= imgWidth || h < 0 || h >= imgHeight)
                return m_border_value;
            else
                return m_desc.at(n, h, w, c);
        }

        // We can return early if our coordinates are within the bounds. This is to avoid expensive calculations
        // required at image borders. While this may cause branch divergence, a good bulk of the pixels should fall
        // within image bounds and will take the same branch. This is preferred over having to do expensive calculations
        // for EVERY pixel in the image (most of which do not require said calculations).
        if (w >= 0 && w < imgWidth && h >= 0 && h < imgHeight) {
            return m_desc.at(n, h, w, c);
        }

        // Otherwise, do some additional calculations to map the provided x and y coordinates to be within bounds.
        int64_t x = w, y = h;

        // Reflect border type implementation. (Note: This is NOT REFLECT101, pixels at the border will be duplicated as
        // is the intended behavior for this border mode.)
        if constexpr (BorderType == eBorderType::BORDER_TYPE_REFLECT) {
            if (w < 0 || w >= imgWidth) {
                x = detail::reflect_border_coord_i64(w, imgWidth);
            }
            if (h < 0 || h >= imgHeight) {
                y = detail::reflect_border_coord_i64(h, imgHeight);
            }
        }

        if constexpr (BorderType == eBorderType::BORDER_TYPE_REFLECT101) {
            x = detail::reflect101_border_coord_i64(w, imgWidth);
            y = detail::reflect101_border_coord_i64(h, imgHeight);
        }

        // Replicate: clamp to edge. Equivalent to per-axis OOB snap; min/max maps cleanly to GPU integer ops.
        if constexpr (BorderType == eBorderType::BORDER_TYPE_REPLICATE) {
            x = detail::clamp_i64(w, 0, imgWidth - 1);
            y = detail::clamp_i64(h, 0, imgHeight - 1);
        }

        // Wrap border type implementation
        if constexpr (BorderType == eBorderType::BORDER_TYPE_WRAP) {
            x = detail::euclid_mod_i64_fast(w, imgWidth);
            y = detail::euclid_mod_i64_fast(h, imgHeight);
        }

        return m_desc.at(n, y, x, c);
    }

    /**
     * @brief Retrives the height of the image at batch index n.
     *
     * @param n Batch index. Ignored when W is a uniform-shape wrapper.
     * @return Image height.
     */
    __device__ __host__ inline int64_t height(int64_t n = 0) const { return m_desc.height(n); }

    /**
     * @brief Retrieves the width of the image at batch index n.
     *
     * @param n Batch index. Ignored when W is a uniform-shape wrapper.
     * @return Image width.
     */
    __device__ __host__ inline int64_t width(int64_t n = 0) const { return m_desc.width(n); }

    /**
     * @brief Retrieves the number of batches in the image tensor.
     *
     * @return Number of batches.
     */
    __device__ __host__ inline int64_t batches() const { return m_desc.batches(); }

    /**
     * @brief Retrieves the number of channels in the image.
     *
     * @return Image channels.
     */
    __device__ __host__ inline int64_t channels() const { return m_desc.channels(); }

   private:
    W m_desc;
    ValueType m_border_value;
};
}  // namespace roccv
