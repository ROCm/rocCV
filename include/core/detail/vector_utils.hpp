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

#include "core/detail/type_traits.hpp"

namespace roccv::detail {

/**
 * @brief Create a vectorized type with all values set to v.
 *
 * @tparam T The vectorized type.
 * @param v The value to set. Value must be the same as the base type of type T.
 * @return A vectorized type with all values set to v.
 */
template <typename T, typename Base = BaseType<T>, class = std::enable_if_t<HasTypeTraits<T>>>
__device__ __host__ constexpr T SetAll(Base v) {
    if constexpr (!IsCompound<T>) {
        return v;
    } else {
        if constexpr (NumElements<T> == 1)
            return (T){v};
        else if constexpr (NumElements<T> == 2)
            return (T){v, v};
        else if constexpr (NumElements<T> == 3)
            return (T){v, v, v};
        else if constexpr (NumElements<T> == 4)
            return (T){v, v, v, v};
    }
}
}  // namespace roccv::detail