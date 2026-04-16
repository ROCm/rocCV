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

#include <hip/hip_runtime.h>

#include <cstdint>

#include "core/wrappers/image_wrapper.hpp"
#include "operator_types.h"

namespace roccv {
namespace detail {

/** Branchless absolute value for int64 (two's complement); avoids libm/std::abs on GPU. */
__device__ __host__ __forceinline__ int64_t abs_i64(int64_t v) {
    const int64_t mask = v >> 63;
    return (v ^ mask) - mask;
}

__device__ __host__ __forceinline__ int32_t abs_i32(int32_t v) {
    const int32_t mask = v >> 31;
    return (v ^ mask) - mask;
}

__device__ __host__ __forceinline__ int32_t min_i32(int32_t a, int32_t b) { return a < b ? a : b; }

__device__ __host__ __forceinline__ int64_t min_i64(int64_t a, int64_t b) { return a < b ? a : b; }

__device__ __host__ __forceinline__ int64_t max_i64(int64_t a, int64_t b) { return a > b ? a : b; }

/** Clamp v to [lo, hi]. */
__device__ __host__ __forceinline__ int64_t clamp_i64(int64_t v, int64_t lo, int64_t hi) {
    return min_i64(max_i64(v, lo), hi);
}

__device__ __host__ inline int32_t euclid_mod_i32(int32_t a, int32_t modulus) {
    int32_t r = a % modulus;
    if (r < 0) r += modulus;
    return r;
}

/** Euclidean modulo: result in [0, modulus) for modulus > 0. One hardware remainder vs (a%m+m)%m. */
__device__ __host__ inline int64_t euclid_mod_i64(int64_t a, int64_t modulus) {
    int64_t r = a % modulus;
    if (r < 0) r += modulus;
    return r;
}

/**
 * On GPU, use 32-bit remainder when operands fit; avoids 64-bit integer division in REFLECT/WRAP hot paths.
 * Host always uses euclid_mod_i64. Values must stay in range for correctness.
 */
__device__ __host__ inline int64_t euclid_mod_i64_fast(int64_t a, int64_t modulus) {
#if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
    constexpr int64_t kLim = int64_t{1} << 30;
    if (modulus > 0 && modulus < kLim && a > -kLim && a < kLim) {
        int32_t ai = static_cast<int32_t>(a);
        int32_t m = static_cast<int32_t>(modulus);
        int32_t r = ai % m;
        if (r < 0) r += m;
        return static_cast<int64_t>(r);
    }
#endif
    return euclid_mod_i64(a, modulus);
}

/**
 * OpenCV-style BORDER_REFLECT axis map: period 2*extent, edge pixels duplicated (not REFLECT101).
 * Equivalent to: val = euclid_mod(coord, 2*extent); min(val, 2*extent - 1 - val).
 * The min form avoids a branch on val < extent and fuses well with 32-bit mod on GPU.
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
 * BORDER_REFLECT101 axis map: period (2*extent - 2), endpoints excluded from reflection.
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
 * @brief Wrapper class for ImageWrapper. This extends the descriptors by defining behaviors for when tensor
 * coordinates go out of scope.
 *
 * @tparam T The underlying data type of the tensor.
 * @tparam BorderType The border type to use when coordinates are out of bounds.
 */
template <typename T, eBorderType BorderType>
class BorderWrapper {
   public:
    /**
     * @brief Wraps an ImageWrapper and extends its capabilities to handle out of bounds coordinates.
     *
     * @param tensor The tensor to wrap.
     * @param border_value The fallback border color to use when using a constant border mode.
     */
    BorderWrapper(const Tensor& tensor, T border_value) : m_desc(tensor), m_border_value(border_value) {}

    /**
     * @brief Constructs a BorderWrapper from an existing ImageWrapper. Extends its capabilities to handle out of bound
     * coordinates.
     *
     * @param image_wrapper The ImageWrapper to wrap around the BorderWrapper.
     * @param border_value The fallback border color to use when using a constant border mode.
     */
    BorderWrapper(ImageWrapper<T> image_wrapper, T border_value)
        : m_desc(image_wrapper), m_border_value(border_value) {}

    /**
     * @brief Sample the underlying image with no border logic. Caller must ensure coordinates are in-range.
     */
    __device__ __host__ inline const T at_inbounds(int64_t n, int64_t h, int64_t w, int64_t c) const {
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
    __device__ __host__ const T at(int64_t n, int64_t h, int64_t w, int64_t c) const {
        // Constant border type implementation. This is a special case which doesn't remap values, but rather returns
        // the provided constant value.
        if constexpr (BorderType == eBorderType::BORDER_TYPE_CONSTANT) {
            if (w < 0 || w >= width() || h < 0 || h >= height())
                return m_border_value;
            else
                return m_desc.at(n, h, w, c);
        }

        // We can return early if our coordinates are within the bounds. This is to avoid expensive calculations
        // required at image borders. While this may cause branch divergence, a good bulk of the pixels should fall
        // within image bounds and will take the same branch. This is preferred over having to do expensive calculations
        // for EVERY pixel in the image (most of which do not require said calculations).
        if (w >= 0 && w < width() && h >= 0 && h < height()) {
            return m_desc.at(n, h, w, c);
        }

        // Otherwise, do some additional calculations to map the provided x and y coordinates to be within bounds.
        int64_t x = w, y = h;
        int64_t imgWidth = width(), imgHeight = height();

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
     * @brief Retrives the height of the images.
     *
     * @return Image height.
     */
    __device__ __host__ inline int64_t height() const { return m_desc.height(); }

    /**
     * @brief Retrieves the width of the image.
     *
     * @return Image width.
     */
    __device__ __host__ inline int64_t width() const { return m_desc.width(); }

    /**
     * @brief Retrieves the number of batches in the image tensor.
     *
     * @return Number of batches.
     */
    __device__ __host__ inline int64_t batches() const { return m_desc.batches(); }

    /**
     * @brief Retries the number of channels in the image.
     *
     * @return Image channels.
     */
    __device__ __host__ inline int64_t channels() const { return m_desc.channels(); }

   private:
    ImageWrapper<T> m_desc;
    T m_border_value;
};
}  // namespace roccv