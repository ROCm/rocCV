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

#include <core/detail/casting.hpp>

#include "test_helpers.hpp"

using namespace roccv::detail;
using namespace roccv::tests;
using namespace roccv;

int main(int argc, char **argv) {
    (void)argc;
    (void)argv;
    TEST_CASES_BEGIN();

    // ----- Scalar same-type early return -----
    TEST_CASE(EXPECT_EQ(StaticCast<int>(int{42}), 42));
    TEST_CASE(EXPECT_EQ(StaticCast<float>(1.5f), 1.5f));
    TEST_CASE(EXPECT_EQ(StaticCast<double>(2.5), 2.5));

    // ----- Scalar conversions: behave exactly like static_cast -----
    // Float -> int: truncates toward zero, no clamping or rounding.
    TEST_CASE(EXPECT_EQ(StaticCast<int>(3.7f), 3));
    TEST_CASE(EXPECT_EQ(StaticCast<int>(-3.7f), -3));
    TEST_CASE(EXPECT_EQ(StaticCast<int>(0.999f), 0));
    // int -> float: exact for small values.
    TEST_CASE(EXPECT_EQ(StaticCast<float>(int{42}), 42.0f));
    TEST_CASE(EXPECT_EQ(StaticCast<float>(int{-42}), -42.0f));
    // Widening / narrowing integer conversions follow C++ rules (no clamping).
    TEST_CASE(EXPECT_EQ(StaticCast<int32_t>(int8_t{-1}), -1));
    TEST_CASE(EXPECT_EQ(StaticCast<uint8_t>(int32_t{300}), static_cast<uint8_t>(300)));
    // double -> float
    TEST_CASE(EXPECT_EQ(StaticCast<float>(1.5), 1.5f));

    // ----- Vector same-type early return -----
    TEST_CASE(EXPECT_TRUE((StaticCast<float4>(float4{1.0f, 2.0f, 3.0f, 4.0f}) == float4{1.0f, 2.0f, 3.0f, 4.0f})));
    TEST_CASE(EXPECT_TRUE((StaticCast<uchar4>(uchar4{1, 2, 3, 4}) == uchar4{1, 2, 3, 4})));

    // ----- Vector conversions across base types (same arity) -----
    TEST_CASE(EXPECT_TRUE((StaticCast<float4>(uchar4{1, 2, 3, 4}) == float4{1.0f, 2.0f, 3.0f, 4.0f})));
    TEST_CASE(EXPECT_TRUE((StaticCast<int4>(float4{1.7f, -2.7f, 3.3f, -3.3f}) == int4{1, -2, 3, -3})));
    TEST_CASE(EXPECT_TRUE((StaticCast<float3>(uchar3{10, 20, 30}) == float3{10.0f, 20.0f, 30.0f})));
    TEST_CASE(EXPECT_TRUE((StaticCast<float2>(int2{-5, 5}) == float2{-5.0f, 5.0f})));

    // ----- Partial-element extraction (NumElements<T> < NumElements<U>) -----
    // Per the enable_if (NumElements<T> <= NumElements<U>), narrower vectors are allowed.
    TEST_CASE(EXPECT_TRUE((StaticCast<float2>(float4{1.0f, 2.0f, 3.0f, 4.0f}) == float2{1.0f, 2.0f})));
    TEST_CASE(EXPECT_TRUE((StaticCast<float3>(float4{1.0f, 2.0f, 3.0f, 4.0f}) == float3{1.0f, 2.0f, 3.0f})));
    TEST_CASE(EXPECT_TRUE((StaticCast<uchar2>(uchar4{10, 20, 30, 40}) == uchar2{10, 20})));
    // Cross-type partial extraction
    TEST_CASE(EXPECT_TRUE((StaticCast<int2>(float4{1.7f, -2.7f, 3.3f, -3.3f}) == int2{1, -2})));

    // ----- Scalar destination from compound source -----
    // NumElements<T> == 1 with compound U: takes element 0 only.
    TEST_CASE(EXPECT_EQ(StaticCast<float>(float4{7.0f, 1.0f, 2.0f, 3.0f}), 7.0f));
    TEST_CASE(EXPECT_EQ(StaticCast<int>(float2{4.7f, 9.0f}), 4));

    // ----- No clamping on overflow (this is what distinguishes StaticCast from SaturateCast) -----
    // float -> uint8 with out-of-range input: result is implementation-defined per C++,
    // but specifically does NOT clamp like SaturateCast would.
    // We only assert that the values DIFFER from the saturate-cast behaviour to lock
    // in StaticCast's pass-through semantics.
    TEST_CASE(EXPECT_NE(static_cast<int>(StaticCast<uint8_t>(int32_t{300})),
                        static_cast<int>(SaturateCast<uint8_t>(int32_t{300}))));

    TEST_CASES_END();
}
