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

#include <algorithm>
#include <cmath>

#include "core/detail/type_traits.hpp"

namespace roccv::detail {

/**
 * @brief Rounds a floating-point value to the nearest integer using IEEE
 * half-to-even rounding (the default rounding mode). Matches the semantics of
 * __float2int_rn on device. Selects single- vs double-precision based on the
 * argument type to avoid silent precision loss when U is double.
 */
template <typename U>
__device__ __host__ inline U IEEERound(U v) {
    static_assert(std::is_floating_point_v<U>, "IEEERound requires a floating-point input");
#ifdef __HIP_DEVICE_COMPILE__
    if constexpr (std::is_same_v<U, float>) {
        return rintf(v);
    } else {
        return rint(v);
    }
#else
    return std::rint(v);
#endif
}

/**
 * @brief Clamps v to [lo, hi]. Uses fminf/fmin/fmaxf/fmax on device to avoid
 * the branchy std::clamp implementation.
 */
template <typename U>
__device__ __host__ inline U FpClamp(U v, U lo, U hi) {
    static_assert(std::is_floating_point_v<U>, "FpClamp requires a floating-point input");
#ifdef __HIP_DEVICE_COMPILE__
    if constexpr (std::is_same_v<U, float>) {
        return fminf(fmaxf(v, lo), hi);
    } else {
        return fmin(fmax(v, lo), hi);
    }
#else
    return std::clamp(v, lo, hi);
#endif
}

/**
 * @brief ScalarSaturateCast is for implementation purposes only. Use SaturateCast directly.
 */
template <typename T, typename U, class = std::enable_if_t<!IsCompound<T> && !IsCompound<U>>>
__device__ __host__ T ScalarSaturateCast(U v) {
    constexpr bool smallToBig = sizeof(U) <= sizeof(T);
    constexpr bool bigToSmall = !smallToBig;

    if constexpr (std::is_integral_v<T> && std::is_floating_point_v<U>) {
        // Float -> integral: clamp to [min, max] then round (IEEE half-to-even).
        constexpr U minVal = static_cast<U>(std::numeric_limits<T>::lowest());
        constexpr U maxVal = static_cast<U>(std::numeric_limits<T>::max());

        if constexpr (sizeof(T) <= 2) {
            // 8/16 bit integer cases. These can be represented exactly in floating point.
            return static_cast<T>(IEEERound(FpClamp(v, minVal, maxVal)));
        } else {
            // 32/64 bit integer cases. maxVal may round up to an unrepresentable
            // value when cast back, so compare against the rounded source.
            const U rounded = IEEERound(v);
            return rounded >= maxVal   ? std::numeric_limits<T>::max()
                   : rounded <= minVal ? std::numeric_limits<T>::min()
                                       : static_cast<T>(rounded);
        }
    }

    else if constexpr (std::is_integral_v<T> && std::is_integral_v<U> && std::is_signed_v<U> && std::is_unsigned_v<T> &&
                       smallToBig) {
        // Signed -> unsigned, small to big: clamp negative to 0
        // Branchless: max(v, 0) handles negative values
        return static_cast<T>(max(v, U{0}));
    }

    else if constexpr (std::is_integral_v<U> && std::is_integral_v<T> &&
                       ((std::is_signed_v<U> && std::is_signed_v<T>) ||
                        (std::is_unsigned_v<U> && std::is_unsigned_v<T>)) &&
                       bigToSmall) {
        // Same signedness, big -> small: clamp to [min, max]
        constexpr U minVal = static_cast<U>(std::numeric_limits<T>::min());
        constexpr U maxVal = static_cast<U>(std::numeric_limits<T>::max());
        return static_cast<T>(min(max(v, minVal), maxVal));
    }

    else if constexpr (std::is_integral_v<U> && std::is_unsigned_v<U> && std::is_integral_v<T> && std::is_signed_v<T>) {
        // Unsigned -> signed: clamp to max (can't exceed min since unsigned)
        constexpr U maxVal = static_cast<U>(std::numeric_limits<T>::max());
        return static_cast<T>(min(v, maxVal));
    }

    else if constexpr (std::is_integral_v<U> && std::is_signed_v<U> && std::is_integral_v<T> && std::is_unsigned_v<T> &&
                       bigToSmall) {
        // Signed -> unsigned, big -> small: clamp to [0, max]
        constexpr U maxVal = static_cast<U>(std::numeric_limits<T>::max());
        return static_cast<T>(min(max(v, U{0}), maxVal));
    }

    else {
        return static_cast<T>(v);
    }
}

/**
 * @brief Performs a saturation cast from one type to another. Each type must have type traits supported. A
 * saturation cast converts one type to another, clamping values down to the minimum/maximum of the desired cast if the
 * value being cast goes out of bounds. For example, casting a float with a value of 256.0f to a uchar would result in a
 * uchar with value 255.
 *
 * @tparam T The type to cast <v> to. The number of elements in this type must be <= to that of v.
 * @param[in] v The value to cast.
 *
 * @return The values in v saturate casted to type T.
 */
template <typename T, typename U,
          class = std::enable_if_t<(HasTypeTraits<T> && HasTypeTraits<U>) && (NumElements<T> <= NumElements<U>)>>
__device__ __host__ T SaturateCast(U v) {
    if constexpr (std::is_same_v<T, U>) {
        return v;
    }

    T ret{};

    GetElement(ret, 0) = ScalarSaturateCast<BaseType<T>>(GetElement(v, 0));
    if constexpr (NumElements<T> >= 2) GetElement(ret, 1) = ScalarSaturateCast<BaseType<T>>(GetElement(v, 1));
    if constexpr (NumElements<T> >= 3) GetElement(ret, 2) = ScalarSaturateCast<BaseType<T>>(GetElement(v, 2));
    if constexpr (NumElements<T> >= 4) GetElement(ret, 3) = ScalarSaturateCast<BaseType<T>>(GetElement(v, 3));

    return ret;
}

/**
 * @brief ScalarRangeCast is for implementation purposes only. Use RangeCast directly instead.
 */
template <typename T, typename U,
          class = std::enable_if_t<(HasTypeTraits<T> && HasTypeTraits<U>) && (!IsCompound<T> && !IsCompound<U>)>>
__device__ __host__ T ScalarRangeCast(U v) {
    if constexpr (std::is_same_v<T, U>) {
        // Types are the same, no work needed
        return v;
    }

    else if constexpr (std::is_integral_v<T> && std::is_floating_point_v<U> && std::is_signed_v<T>) {
        // Float to signed integer. Map [-1, 1] -> [min, max] with IEEE half-to-even rounding.
        constexpr U scale = static_cast<U>(std::numeric_limits<T>::max());

        if constexpr (sizeof(T) <= 2) {
            // 8/16 bit signed cases. These can be represented exactly in floating point,
            // so clamp first then round.
            return static_cast<T>(IEEERound(FpClamp(v, U{-1}, U{1}) * scale));
        } else {
            // 32/64 bit signed cases.
            return v >= U{1}    ? std::numeric_limits<T>::max()
                   : v <= U{-1} ? std::numeric_limits<T>::min()
                                : static_cast<T>(IEEERound(scale * v));
        }
    }

    else if constexpr (std::is_integral_v<T> && std::is_floating_point_v<U> && std::is_unsigned_v<T>) {
        // float to unsigned integers
        constexpr U scale = static_cast<U>(std::numeric_limits<T>::max());

        if constexpr (sizeof(T) <= 2) {
            // 8/16 bit integer cases. These can be represented exactly in floating point.
#ifdef __HIP_DEVICE_COMPILE__
            if constexpr (std::is_same_v<U, float>) {
                return static_cast<T>(__float2int_rn(__saturatef(v) * scale));
            } else {
                return static_cast<T>(IEEERound(FpClamp(v, U{0}, U{1}) * scale));
            }
#else
            return static_cast<T>(IEEERound(FpClamp(v, U{0}, U{1}) * scale));
#endif
        } else {
            // 32/64 bit integer cases.
            return v >= U{1} ? std::numeric_limits<T>::max() : v <= U{0} ? T{0} : static_cast<T>(IEEERound(v * scale));
        }
    }

    else if constexpr (std::is_floating_point_v<T> && std::is_integral_v<U> && std::is_signed_v<U>) {
        // Signed integer to float
        constexpr T invmax = T{1} / static_cast<T>(std::numeric_limits<U>::max());
        T out = static_cast<T>(v) * invmax;
        return out < T{-1} ? T{-1} : out;
    }

    else if constexpr (std::is_floating_point_v<T> && std::is_integral_v<U> && std::is_unsigned_v<U>) {
        // Unsigned integer to float
        constexpr T invmax = T{1} / static_cast<T>(std::numeric_limits<U>::max());
        return static_cast<T>(v) * invmax;
    }

    else {
        // All other cases reduce to a saturate cast
        return ScalarSaturateCast<T>(v);
    }
}

/**
 * @brief Performs a range cast from the source type of v to the type specified by T. Range conversions are defined
 * based on the types provided and their numeric limits. When range casting from an integral type to another integral
 * type, this operation reduces to a saturate cast.
 *
 * When casting from an integral to a float, unsigned integers will map to [0.0f, 1.0f] and signed integers will map to
 * [-1.0f, 1.0f]. When casting from a float to an integral, provided values must be in [0.0f, 1.0f] for casts to
 * unsigned integers, and [-1.0f, 1.0f] for signed integers. If floating point values are not provided in this range,
 * they will be implicitly clamped to the appropriate range before the range cast is performed.
 *
 * This operations accepts vectorized and scalar types as long as they have type traits (HasTypeTraits<type> should be
 * true.)
 *
 * @tparam T The type to cast to.
 * @param[in] v The value to be casted.
 *
 * @return The values in v range casted to type T.
 */
template <typename T, typename U,
          class = std::enable_if_t<(HasTypeTraits<T> && HasTypeTraits<U>) && NumElements<T> <= NumElements<U>>>
__device__ __host__ T RangeCast(U v) {
    if constexpr (std::is_same_v<T, U>) {
        return v;
    }

    T ret{};

    GetElement(ret, 0) = ScalarRangeCast<BaseType<T>>(GetElement(v, 0));
    if constexpr (NumElements<T> >= 2) GetElement(ret, 1) = ScalarRangeCast<BaseType<T>>(GetElement(v, 1));
    if constexpr (NumElements<T> >= 3) GetElement(ret, 2) = ScalarRangeCast<BaseType<T>>(GetElement(v, 2));
    if constexpr (NumElements<T> >= 4) GetElement(ret, 3) = ScalarRangeCast<BaseType<T>>(GetElement(v, 3));

    return ret;
}

/**
 * @brief Performs a static cast for vectorized types.
 *
 * @tparam T The vectorized type to cast to.
 * @param[in] v The value to cast to type T.
 *
 * @return The value v casted to vectorized type T.
 */
template <typename T, typename U,
          class = std::enable_if_t<(HasTypeTraits<T> && HasTypeTraits<U>) && NumElements<T> <= NumElements<U>>>
__device__ __host__ T StaticCast(U v) {
    if constexpr (std::is_same_v<T, U>) {
        // Both same type, just return the value.
        return v;
    } else if constexpr (!IsCompound<T> && !IsCompound<U>) {
        // Both scalar values. Reduces to a standard static cast.
        return static_cast<T>(v);
    } else {
        // Vector types. Perform casting on each element.
        T ret{};
        GetElement(ret, 0) = StaticCast<BaseType<T>>(GetElement(v, 0));
        if constexpr (NumElements<T> >= 2) GetElement(ret, 1) = StaticCast<BaseType<T>>(GetElement(v, 1));
        if constexpr (NumElements<T> >= 3) GetElement(ret, 2) = StaticCast<BaseType<T>>(GetElement(v, 2));
        if constexpr (NumElements<T> >= 4) GetElement(ret, 3) = StaticCast<BaseType<T>>(GetElement(v, 3));

        return ret;
    }
}
}  // namespace roccv::detail