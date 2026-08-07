/*
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

/**
 * @file sampling_helpers.hpp
 * @brief Small host/device helpers for border coordinate math and interpolation index conversion.
 */

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>

namespace roccv {
namespace detail {

/**
 * @brief Branchless absolute value of a signed 64-bit integer (two's complement).
 * @param v Input value.
 * @return Non-negative absolute value of @p v.
 * @note Avoids @c std::abs in device code; useful on GPU hot paths.
 */
__device__ __host__ __forceinline__ int64_t abs_i64(int64_t v) {
    const int64_t mask = v >> 63;
    return (v ^ mask) - mask;
}

/**
 * @brief Branchless absolute value of a signed 32-bit integer (two's complement).
 * @param v Input value.
 * @return Non-negative absolute value of @p v.
 */
__device__ __host__ __forceinline__ int32_t abs_i32(int32_t v) {
    const int32_t mask = v >> 31;
    return (v ^ mask) - mask;
}

/**
 * @brief Minimum of two 32-bit signed integers.
 * @param a First operand.
 * @param b Second operand.
 * @return The lesser of @p a and @p b.
 */
__device__ __host__ __forceinline__ int32_t min_i32(int32_t a, int32_t b) { return a < b ? a : b; }

/**
 * @brief Minimum of two 64-bit signed integers.
 * @param a First operand.
 * @param b Second operand.
 * @return The lesser of @p a and @p b.
 */
__device__ __host__ __forceinline__ int64_t min_i64(int64_t a, int64_t b) { return a < b ? a : b; }

/**
 * @brief Maximum of two 32-bit signed integers.
 * @param a First operand.
 * @param b Second operand.
 * @return The greater of @p a and @p b.
 */
__device__ __host__ __forceinline__ int32_t max_i32(int32_t a, int32_t b) { return a > b ? a : b; }

/**
 * @brief Maximum of two 64-bit signed integers.
 * @param a First operand.
 * @param b Second operand.
 * @return The greater of @p a and @p b.
 */
__device__ __host__ __forceinline__ int64_t max_i64(int64_t a, int64_t b) { return a > b ? a : b; }

/**
 * @brief Clamp a signed 32-bit integer to a closed interval.
 * @param v Value to clamp.
 * @param lo Lower bound (inclusive).
 * @param hi Upper bound (inclusive); must satisfy @p lo <= @p hi.
 * @return @p v restricted to the inclusive interval between @p lo and @p hi.
 */
__device__ __host__ __forceinline__ int32_t clamp_i32(int32_t v, int32_t lo, int32_t hi) {
    return min_i32(max_i32(v, lo), hi);
}

/**
 * @brief Clamp a signed 64-bit integer to a closed interval.
 * @param v Value to clamp.
 * @param lo Lower bound (inclusive).
 * @param hi Upper bound (inclusive); must satisfy @p lo <= @p hi.
 * @return @p v restricted to the inclusive interval between @p lo and @p hi.
 */
__device__ __host__ __forceinline__ int64_t clamp_i64(int64_t v, int64_t lo, int64_t hi) {
    return min_i64(max_i64(v, lo), hi);
}

/**
 * @brief Clamp a signed integer to a closed interval (templated version).
 * @tparam IndexT Index type (int32_t or int64_t).
 * @param v Value to clamp.
 * @param lo Lower bound (inclusive).
 * @param hi Upper bound (inclusive); must satisfy @p lo <= @p hi.
 * @return @p v restricted to the inclusive interval between @p lo and @p hi.
 */
template <typename IndexT>
__device__ __host__ __forceinline__ IndexT clamp(IndexT v, IndexT lo, IndexT hi) {
    if constexpr (std::is_same_v<IndexT, int32_t>) {
        return clamp_i32(v, lo, hi);
    } else {
        return clamp_i64(v, lo, hi);
    }
}

/**
 * @brief Euclidean (non-negative) modulo for 32-bit operands.
 * @param a Dividend.
 * @param modulus Strictly positive modulus.
 * @return Remainder in the half-open range <tt>[0, modulus)</tt>, congruent to @p a modulo @p modulus.
 */
__device__ __host__ inline int32_t euclid_mod_i32(int32_t a, int32_t modulus) {
    int32_t r = a % modulus;
    if (r < 0) r += modulus;
    return r;
}

/**
 * @brief Euclidean (non-negative) modulo for 64-bit operands.
 * @param a Dividend.
 * @param modulus Strictly positive modulus.
 * @return Remainder in the half-open range <tt>[0, modulus)</tt>, congruent to @p a modulo @p modulus.
 * @note Uses a single remainder and correction instead of repeating modulo and add.
 */
__device__ __host__ inline int64_t euclid_mod_i64(int64_t a, int64_t modulus) {
    int64_t r = a % modulus;
    if (r < 0) r += modulus;
    return r;
}

/**
 * @brief Euclidean modulo with a 32-bit fast path on the GPU when operands are in a safe range.
 * @param a Dividend.
 * @param modulus Strictly positive modulus.
 * @return Same as euclid_mod_i64() when operands fit the device fast path; otherwise defers to euclid_mod_i64().
 * @note On device, uses 32-bit remainder when @p modulus and @p a are sufficiently small to avoid 64-bit division.
 *       On host, always uses euclid_mod_i64(). Caller must keep values in range when relying on the fast path.
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
 * @brief Convert a subpixel coordinate to the integer grid index below @p x (floor).
 * @tparam IndexT Index type (int32_t or int64_t).
 * @param x Source coordinate in pixels.
 * @return Largest IndexT not greater than @p x (i.e. floor), suitable as the left/top neighbor index for
 * bilinear/cubic.
 * @note On device, uses a floor intrinsic compatible with HIP @c __float2ll_rd lowering; on host uses @c floorf().
 */
template <typename IndexT>
__device__ __host__ __forceinline__ IndexT interp_floor(float x) {
#if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
    return static_cast<IndexT>(static_cast<long long>(__builtin_elementwise_floor(x)));
#else
    return static_cast<IndexT>(floorf(x));
#endif
}

/**
 * @brief Nearest-neighbor rounding of a subpixel coordinate to an integer index.
 * @tparam IndexT Index type (int32_t or int64_t).
 * @param x Source coordinate in pixels.
 * @return Integer closest to @p x, with half values rounded away from zero.
 */
template <typename IndexT>
__device__ __host__ __forceinline__ IndexT interp_nearest(float x) {
    if constexpr (std::is_same_v<IndexT, int32_t>) {
        return static_cast<int32_t>(std::lroundf(x));
    } else {
        return static_cast<int64_t>(std::llroundf(x));
    }
}
}  // namespace detail
}  // namespace roccv
