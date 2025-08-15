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

#include <hip/hip_runtime.h>

#include "core/detail/type_traits.hpp"
#include "core/image_format.hpp"

namespace roccv::detail {

template <typename T, uint8_t... Indices>
__host__ __device__ constexpr T Swizzle(T &v) {
    return T{v.data[Indices]...};
}

template <eSwizzle SwizzlePattern, typename T>
__host__ __device__ constexpr T Swizzle(T &v) {
    if constexpr (SwizzlePattern == eSwizzle::XYZW) {
        return v;
    } else if constexpr (SwizzlePattern == eSwizzle::ZYXW) {
        if constexpr (detail::NumElements<T> == 1)
            return Swizzle<T, 0>(v);
        else if constexpr (detail::NumElements<T> == 3)
            return Swizzle<T, 2, 1, 0>(v);
        else if constexpr (detail::NumElements<T> == 4)
            return Swizzle<T, 2, 1, 0, 3>(v);
    }
}

}  // namespace roccv::detail