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

#include "core/detail/type_traits.hpp"

namespace roccv::detail {

/**
 * @brief ScalarSaturateCast is for implementation purposes only. Use SaturateCast directly.
 */
template <typename T, typename U, class = std::enable_if_t<!IsCompound<T> && !IsCompound<U>>>
__device__ __host__ T ScalarSaturateCast(U v) {
    constexpr bool smallToBig = sizeof(U) <= sizeof(T);
    constexpr bool bigToSmall = !smallToBig;

    if constexpr (std::is_integral_v<T> && std::is_floating_point_v<U>) {
        // Any float -> any integral
        return static_cast<T>(std::clamp<U>(std::round(v), static_cast<U>(std::numeric_limits<T>::min()),
                                            static_cast<U>(std::numeric_limits<T>::max())));
    } else if constexpr (std::is_integral_v<T> && std::is_integral_v<U> && std::is_signed_v<U> && std::is_signed_v<T> &&
                         smallToBig) {
        // Any integral signed -> Any integral unsigned, small -> big or equal
        return v <= 0 ? 0 : static_cast<T>(v);
    } else if constexpr (std::is_integral_v<U> && std::is_integral_v<T> &&
                         ((std::is_signed_v<U> && std::is_signed_v<T>) ||
                          (std::is_unsigned_v<U> && std::is_unsigned_v<T>)) &&
                         bigToSmall) {
        // Any integral signed -> Any integral signed, big -> small
        // Any integral unsigned -> Any integral unsigned, big -> small
        return v <= std::numeric_limits<T>::min()
                   ? std::numeric_limits<T>::min()
                   : (v >= std::numeric_limits<T>::max() ? std::numeric_limits<T>::max() : static_cast<T>(v));
    } else if constexpr (std::is_integral_v<U> && std::is_unsigned_v<U> && std::is_integral_v<T> &&
                         std::is_signed_v<T>) {
        // Any integral unsigned -> Any integral signed, small -> big or equal
        return v >= std::numeric_limits<T>::max() ? std::numeric_limits<T>::max() : static_cast<T>(v);
    } else if constexpr (std::is_integral_v<U> && std::is_signed_v<U> && std::is_integral_v<T> &&
                         std::is_unsigned_v<T> && bigToSmall) {
        // Any integral signed -> Any integral unsigned, big -> small
        return v <= static_cast<U>(std::numeric_limits<T>::min())
                   ? std::numeric_limits<T>::min()
                   : (v >= static_cast<U>(std::numeric_limits<T>::max()) ? std::numeric_limits<T>::max
                                                                         : static_cast<T>(v));
    } else {
        // All other cases fall into this
        return v;
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
        // Float to signed integers
        return v >= 1.0f    ? std::numeric_limits<T>::max()
               : v <= -1.0f ? std::numeric_limits<T>::min()
                            : static_cast<T>(std::round(static_cast<U>(std::numeric_limits<T>::max()) * v));
    }

    else if constexpr (std::is_integral_v<T> && std::is_floating_point_v<U> && std::is_unsigned_v<T>) {
        // float to unsigned integers
        return v >= 1.0f   ? std::numeric_limits<T>::max()
               : v <= 0.0f ? 0
                           : static_cast<T>(std::round(static_cast<U>(std::numeric_limits<T>::max()) * v));
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