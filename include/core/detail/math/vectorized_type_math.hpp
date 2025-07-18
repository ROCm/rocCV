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
 * @file vectorized_type_math.hpp
 * @brief This header defines common arithmetic operators for HIP vectorized types.
 *
 */

#pragma once

#include <hip/hip_runtime.h>

#include <type_traits>

#include "core/detail/type_traits.hpp"

namespace roccv::detail::math {
#define DEFINE_VECTOR_BINARY_OP(OPERATOR, NAME)                                                         \
    template <typename T, typename U>                                                                   \
    __host__ __device__ auto NAME(T a, U b) {                                                           \
        using RT = MakeType<decltype(std::declval<BaseType<T>>() OPERATOR std::declval<BaseType<U>>()), \
                            NumComponents<T> == 0 ? NumComponents<U> : NumComponents<T>>;               \
        if constexpr (NumComponents<T> == 0) {                                                          \
            if constexpr (NumComponents<RT> == 1)                                                       \
                return RT{a OPERATOR b.x};                                                              \
            else if constexpr (NumComponents<RT> == 2)                                                  \
                return RT{a OPERATOR b.x, a OPERATOR b.y};                                              \
            else if constexpr (NumComponents<RT> == 3)                                                  \
                return RT{a OPERATOR b.x, a OPERATOR b.y, a OPERATOR b.z};                              \
            else if constexpr (NumComponents<RT> == 4)                                                  \
                return RT{a OPERATOR b.x, a OPERATOR b.y, a OPERATOR b.z, a OPERATOR b.w};              \
                                                                                                        \
        } else if constexpr (NumComponents<U> == 0) {                                                   \
            if constexpr (NumComponents<RT> == 1)                                                       \
                return RT{a.x OPERATOR b};                                                              \
            else if constexpr (NumComponents<RT> == 2)                                                  \
                return RT{a.x OPERATOR b, a.y OPERATOR b};                                              \
            else if constexpr (NumComponents<RT> == 3)                                                  \
                return RT{a.x OPERATOR b, a.y OPERATOR b, a.z OPERATOR b};                              \
            else if constexpr (NumComponents<RT> == 4)                                                  \
                return RT{a.x OPERATOR b, a.y OPERATOR b, a.z OPERATOR b, a.w OPERATOR b};              \
                                                                                                        \
        } else {                                                                                        \
            if constexpr (NumComponents<RT> == 1)                                                       \
                return RT{a.x OPERATOR b.x};                                                            \
            else if constexpr (NumComponents<RT> == 2)                                                  \
                return RT{a.x OPERATOR b.x, a.y OPERATOR b.y};                                          \
            else if constexpr (NumComponents<RT> == 3)                                                  \
                return RT{a.x OPERATOR b.x, a.y OPERATOR b.y, a.z OPERATOR b.z};                        \
            else if constexpr (NumComponents<RT> == 4)                                                  \
                return RT{a.x OPERATOR b.x, a.y OPERATOR b.y, a.z OPERATOR b.z, a.w OPERATOR b.w};      \
        }                                                                                               \
    }

#define DEFINE_VECTOR_BINARY_FUNC(FUNC, NAME)                                                         \
    template <typename T, typename U>                                                                 \
    __host__ __device__ auto NAME(T a, U b) {                                                         \
        using RT = MakeType<decltype(FUNC(std::declval<BaseType<T>>(), std::declval<BaseType<U>>())), \
                            NumComponents<T> == 0 ? NumComponents<U> : NumComponents<T>>;             \
        if constexpr (NumComponents<T> == 0) {                                                        \
            if constexpr (NumComponents<RT> == 1)                                                     \
                return RT{FUNC(a, b.x)};                                                              \
            else if constexpr (NumComponents<RT> == 2)                                                \
                return RT{FUNC(a, b.x), FUNC(a, b.y)};                                                \
            else if constexpr (NumComponents<RT> == 3)                                                \
                return RT{FUNC(a, b.x), FUNC(a, b.y), FUNC(a, b.z)};                                  \
            else if constexpr (NumComponents<RT> == 4)                                                \
                return RT{FUNC(a, b.x), FUNC(a, b.y), FUNC(a, b.z), FUNC(a, b.w)};                    \
                                                                                                      \
        } else if constexpr (NumComponents<U> == 0) {                                                 \
            if constexpr (NumComponents<RT> == 1)                                                     \
                return RT{FUNC(a.x, b)};                                                              \
            else if constexpr (NumComponents<RT> == 2)                                                \
                return RT{FUNC(a.x, b), FUNC(a.y, b)};                                                \
            else if constexpr (NumComponents<RT> == 3)                                                \
                return RT{FUNC(a.x, b), FUNC(a.y, b), FUNC(a.z, b)};                                  \
            else if constexpr (NumComponents<RT> == 4)                                                \
                return RT{FUNC(a.x, b), FUNC(a.y, b), FUNC(a.z, b), FUNC(a.w, b)};                    \
                                                                                                      \
        } else {                                                                                      \
            if constexpr (NumComponents<RT> == 1)                                                     \
                return RT{FUNC(a.x, b.x)};                                                            \
            else if constexpr (NumComponents<RT> == 2)                                                \
                return RT{FUNC(a.x, b.x), FUNC(a.y, b.y)};                                            \
            else if constexpr (NumComponents<RT> == 3)                                                \
                return RT{FUNC(a.x, b.x), FUNC(a.y, b.y), FUNC(a.z, b.z)};                            \
            else if constexpr (NumComponents<RT> == 4)                                                \
                return RT{FUNC(a.x, b.x), FUNC(a.y, b.y), FUNC(a.z, b.z), FUNC(a.w, b.w)};            \
        }                                                                                             \
    }

#define DEFINE_VECTOR_UNARY_FUNC(FUNC, NAME)                          \
    template <typename U, class = std::enable_if_t<HasTypeTraits<U>>> \
    inline __host__ __device__ U NAME(U v) {                          \
        if constexpr (NumComponents<U> == 0)                          \
            return FUNC(v);                                           \
        else if constexpr (NumComponents<U> == 1) {                   \
            return U{FUNC(v.x)};                                      \
        } else if constexpr (NumComponents<U> == 2) {                 \
            return U{FUNC(v.x), FUNC(v.y)};                           \
        } else if constexpr (NumComponents<U> == 3) {                 \
            return U{FUNC(v.x), FUNC(v.y), FUNC(v.z)};                \
        } else if constexpr (NumComponents<U> == 4) {                 \
            return U{FUNC(v.x), FUNC(v.y), FUNC(v.z), FUNC(v.w)};     \
        }                                                             \
    }

DEFINE_VECTOR_BINARY_OP(+, add);
DEFINE_VECTOR_BINARY_OP(*, mul);
DEFINE_VECTOR_BINARY_OP(/, div);
DEFINE_VECTOR_BINARY_OP(-, sub);

DEFINE_VECTOR_BINARY_FUNC(fdividef, vfdividef);
DEFINE_VECTOR_BINARY_FUNC(powf, vpowf);

DEFINE_VECTOR_UNARY_FUNC(sqrtf, vsqrtf);
DEFINE_VECTOR_UNARY_FUNC(rsqrtf, vrsqrtf);

}  // namespace roccv::detail::math