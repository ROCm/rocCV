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

#include <stdint.h>

#include <cstddef>
#include <type_traits>

namespace roccv::detail {

inline constexpr size_t NextPowerOfTwo(size_t value) noexcept {
    if (value <= 1) return 1;
#if defined(__GNUC__) || defined(__clang__)
    constexpr int numBits = sizeof(size_t) * 8;
    return 1UL << (numBits - __builtin_clzl(value - 1));
#else
    value--;
    for (size_t i = 1; i < sizeof(size_t) * 8; i <<= 1) value |= (value >> i);
    return value + 1;
#endif
}

/**
 * @brief Returns true if the given value is a power of two.
 *
 * @param value The value to check.
 * @return True if the given value is a power of two, false otherwise.
 */
template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
inline constexpr bool IsPowerOfTwo(T value) noexcept {
    return value > 0 && (value & (value - 1)) == 0;
}

/**
 * @brief Aligns the given value to the nearest multiple of the given alignment.
 *
 * @param value The value to align.
 * @param alignment The alignment to align to (must be > 0).
 * @return The aligned value (same type as value).
 */
template <typename T, typename U, typename = std::enable_if_t<std::is_integral_v<T> && std::is_integral_v<U>>>
inline constexpr T AlignUp(T value, U alignment) noexcept {
    return alignment > 0
               ? (value + static_cast<T>(alignment) - 1) / static_cast<T>(alignment) * static_cast<T>(alignment)
               : value;
}
}  // namespace roccv::detail