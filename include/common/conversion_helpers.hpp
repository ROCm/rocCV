/**
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
#pragma once

#include <hip/hip_runtime.h>

#include <limits>

template <typename U>
inline __device__ __host__ U RoundImplementationsToYUV(U u) {
    U rounded = std::round(u);
    if (std::abs(rounded - u) == U(0.5) && (static_cast<int64_t>(rounded) & 1)) {
        rounded -= std::copysign(U(1.0), u);
    }
    return rounded;
}

template <typename U>
inline __device__ __host__ U RoundImplementationsFromYUV(U u) {
    U rounded = std::round(u);
    if (std::abs(rounded - u) <= U(0.5) && (static_cast<int64_t>(rounded) & 1)) {
        rounded -= std::copysign(U(1.0), u);
    }
    return rounded;
}

template <typename T, typename U>
inline __device__ __host__ T Clamp(U value, T lo, T hi) {
    return value < lo ? lo : value > hi ? hi : static_cast<T>(value);
}
