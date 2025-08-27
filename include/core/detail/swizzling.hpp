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

#include "core/detail/type_traits.hpp"
#include "core/image_format.hpp"

namespace roccv::detail {

template <eSwizzle Pattern, int N>
struct SwizzleIndexMap;

// clang-format off

// This section defines compile-time index mapping for all defined eSwizzle patterns using template specialization.

// XYZW
template <int N>
struct SwizzleIndexMap<eSwizzle::XYZW, N> {
    using Seq = std::make_integer_sequence<size_t, N>;
};

// ZYXW
template <> struct SwizzleIndexMap<eSwizzle::ZYXW, 1> { using Seq = std::integer_sequence<size_t, 0>; };
template <> struct SwizzleIndexMap<eSwizzle::ZYXW, 3> { using Seq = std::integer_sequence<size_t, 2, 1, 0>; };
template <> struct SwizzleIndexMap<eSwizzle::ZYXW, 4> { using Seq = std::integer_sequence<size_t, 2, 1, 0, 3>; };
// clang-format on

template <typename T, size_t... Indices>
__host__ __device__ constexpr T SwizzleIndicesImpl(const T &v) {
    return T{v[Indices]...};
}

template <typename T, size_t... I>
__host__ __device__ constexpr T ExpandSwizzle(const T &v, std::integer_sequence<size_t, I...>) {
    static_assert(((I < NumElements<T>) && ...), "Swizzle index out of range");
    return SwizzleIndicesImpl<T, I...>(v);
}

/**
 * @brief Rearranges the components of a vector according to a given swizzle pattern.
 *
 * @tparam Pattern The pattern used to rearrange vector components.
 * @tparam T The vector type.
 * @param v The vector to swizzle.
 * @return \p v with its components rearranged according to \p Pattern.
 *
 * @par Example:
 * @code
 * uchar3 vec = make_uchar3(1, 2, 3);
 * uchar3 vecSwizzled = Swizzle<eSwizzle::ZYXW>(vec);
 * // vecSwizzled = {3, 2, 1}
 * @endcode
 *
 */
template <eSwizzle Pattern, typename T>
__host__ __device__ constexpr T Swizzle(const T &v) {
    static_assert(HasTypeTraits<T> && IsCompound<T>, "Type must have type traits and be a compound type");

    constexpr int N = NumElements<T>;
    static_assert(N >= 1 && N <= 4, "Unsupported element count");

    using Seq = typename SwizzleIndexMap<Pattern, N>::Seq;
    return ExpandSwizzle<T>(v, Seq{});
}

}  // namespace roccv::detail