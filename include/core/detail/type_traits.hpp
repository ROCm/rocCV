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

#pragma once

namespace roccv::detail {

template <typename T>
struct HasTypeTraits_t {
    static constexpr bool value = false;
};

template <typename T, int C>
struct MakeType_t;

/**
 * @brief Returns the type traits associated with a datatype.
 *
 * @tparam T
 */
template <typename T>
struct TypeTraits;

/**
 * @brief Creates a compound type with a base type and number of elements.
 *
 * @tparam T The base type.
 * @tparam C Number of components.
 */

#define DEFINE_TYPE_TRAITS(COMPOUND_TYPE, BASE_TYPE, ELEMENTS, COMPONENTS) \
    template <>                                                            \
    struct TypeTraits<COMPOUND_TYPE> {                                     \
        using base_type = BASE_TYPE;                                       \
        static constexpr int elements = ELEMENTS;                          \
        static constexpr int components = COMPONENTS;                      \
    };                                                                     \
                                                                           \
    template <>                                                            \
    struct MakeType_t<BASE_TYPE, COMPONENTS> {                             \
        using type = COMPOUND_TYPE;                                        \
    };                                                                     \
                                                                           \
    template <>                                                            \
    struct HasTypeTraits_t<COMPOUND_TYPE> {                                \
        static constexpr bool value = true;                                \
    }

#define DEFINE_TYPE_TRAITS_0_TO_4(COMPOUND_TYPE, BASE_TYPE) \
    DEFINE_TYPE_TRAITS(BASE_TYPE, BASE_TYPE, 1, 0);         \
    DEFINE_TYPE_TRAITS(COMPOUND_TYPE##1, BASE_TYPE, 1, 1);  \
    DEFINE_TYPE_TRAITS(COMPOUND_TYPE##2, BASE_TYPE, 2, 2);  \
    DEFINE_TYPE_TRAITS(COMPOUND_TYPE##3, BASE_TYPE, 3, 3);  \
    DEFINE_TYPE_TRAITS(COMPOUND_TYPE##4, BASE_TYPE, 4, 4)

// Define compound/scalar types for use with type traits
DEFINE_TYPE_TRAITS_0_TO_4(uchar, unsigned char);
DEFINE_TYPE_TRAITS_0_TO_4(char, signed char);
DEFINE_TYPE_TRAITS_0_TO_4(float, float);
DEFINE_TYPE_TRAITS_0_TO_4(uint, unsigned int);
DEFINE_TYPE_TRAITS_0_TO_4(int, signed int);
DEFINE_TYPE_TRAITS_0_TO_4(short, signed short);

/**
 * @brief Returns the number of elements in a HIP vectorized type. For example: uchar3 will return 3, int2 will
 * return 2.
 *
 * @tparam T
 */
template <typename T>
constexpr int NumElements = TypeTraits<T>::elements;

template <typename T, int C>
using MakeType = MakeType_t<T, C>::type;

/**
 * @brief Returns the number of components in a given type. Scalar types will have 0 components, while vectorized types
 * will have 1-4 components typically.
 *
 * @tparam T
 */
template <typename T>
constexpr int NumComponents = TypeTraits<T>::components;

template <typename T>
constexpr bool IsCompound = TypeTraits<T>::components != 0;

/**
 * @brief Returns the base type of a given HIP vectorized type.
 *
 * @tparam T A HIP vectorized type.
 */
template <typename T>
using BaseType = typename TypeTraits<T>::base_type;

/**
 * @brief Returns whether the datatype passed in has type traits associated with it.
 *
 * @tparam T
 */
template <typename T>
constexpr bool HasTypeTraits = HasTypeTraits_t<T>::value;

template <typename T, typename RT = BaseType<T>, class = std::enable_if_t<HasTypeTraits<T>>>
__host__ __device__ RT &GetElement(T &v, int idx) {
    if constexpr (IsCompound<T>) {
        assert(idx < NumElements<T>);
        return reinterpret_cast<RT *>(&v)[idx];
    } else {
        return v;
    }
}

}  // namespace roccv::detail