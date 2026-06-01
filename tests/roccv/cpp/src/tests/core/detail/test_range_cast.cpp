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

#include <core/detail/casting.hpp>

#include "test_helpers.hpp"

using namespace roccv::detail;
using namespace roccv::tests;
using namespace roccv;

int main(int argc, char **argv) {
    (void)argc;
    (void)argv;
    TEST_CASES_BEGIN();

    // clang-format off

    // Test float -> unsigned/signed integer casting
    TEST_CASE(EXPECT_EQ(RangeCast<int>(1.0f), std::numeric_limits<int>::max()));
    TEST_CASE(EXPECT_EQ(RangeCast<int>(-1.0f), std::numeric_limits<int>::min()));
    TEST_CASE(EXPECT_EQ(RangeCast<uint>(1.0f), std::numeric_limits<uint>::max()));
    TEST_CASE(EXPECT_EQ(RangeCast<uint>(-1.0f), 0));
    TEST_CASE(EXPECT_EQ(RangeCast<uint>(0.0f), 0));


    // Test unsigned/signed integer -> float casting
    TEST_CASE(EXPECT_EQ(RangeCast<float>(std::numeric_limits<int>::max()), 1.0f));
    TEST_CASE(EXPECT_EQ(RangeCast<float>(std::numeric_limits<int>::min()), -1.0f));
    TEST_CASE(EXPECT_EQ(RangeCast<float>(std::numeric_limits<uint>::max()), 1.0f));
    TEST_CASE(EXPECT_EQ(RangeCast<float>(0), 0.0f));

    // Test double -> unsigned/signed integer casting
    TEST_CASE(EXPECT_EQ(RangeCast<int>(1.0), std::numeric_limits<int>::max()));
    TEST_CASE(EXPECT_EQ(RangeCast<int>(-1.0), std::numeric_limits<int>::min()));
    TEST_CASE(EXPECT_EQ(RangeCast<uint>(1.0), std::numeric_limits<uint>::max()));
    TEST_CASE(EXPECT_EQ(RangeCast<uint>(-1.0), 0));

    // Testing unsigned/signed integer -> double casting
    TEST_CASE(EXPECT_EQ(RangeCast<double>(std::numeric_limits<int>::max()), 1.0f));
    TEST_CASE(EXPECT_EQ(RangeCast<double>(std::numeric_limits<int>::min()), -1.0f));
    TEST_CASE(EXPECT_EQ(RangeCast<double>(std::numeric_limits<uint>::max()), 1.0f));
    TEST_CASE(EXPECT_EQ(RangeCast<double>(0), 0.0f));

    // ----- 8/16-bit signed fast path -----
    TEST_CASE(EXPECT_EQ(RangeCast<int8_t>(1.0f), 127));
    TEST_CASE(EXPECT_EQ(RangeCast<int8_t>(-1.0f), -127));
    TEST_CASE(EXPECT_EQ(RangeCast<int8_t>(0.0f), 0));
    TEST_CASE(EXPECT_EQ(RangeCast<int8_t>(2.0f), 127));     // out-of-range positive clamps
    TEST_CASE(EXPECT_EQ(RangeCast<int8_t>(-2.0f), -127));   // out-of-range negative clamps
    TEST_CASE(EXPECT_EQ(RangeCast<int16_t>(1.0f), 32767));
    TEST_CASE(EXPECT_EQ(RangeCast<int16_t>(-1.0f), -32767));
    TEST_CASE(EXPECT_EQ(RangeCast<int16_t>(2.0f), 32767));
    TEST_CASE(EXPECT_EQ(RangeCast<int16_t>(-2.0f), -32767));

    // ----- 8/16-bit unsigned fast path -----
    TEST_CASE(EXPECT_EQ(RangeCast<uint8_t>(1.0f), 255));
    TEST_CASE(EXPECT_EQ(RangeCast<uint8_t>(0.0f), 0));
    TEST_CASE(EXPECT_EQ(RangeCast<uint8_t>(2.0f), 255));    // clamp positive
    TEST_CASE(EXPECT_EQ(RangeCast<uint8_t>(-0.5f), 0));     // clamp negative
    TEST_CASE(EXPECT_EQ(RangeCast<uint16_t>(1.0f), 65535));
    TEST_CASE(EXPECT_EQ(RangeCast<uint16_t>(-1.0f), 0));

    // ----- Rounding mode: must be IEEE half-to-even -----
    TEST_CASE(EXPECT_EQ(RangeCast<uint8_t>(0.5f / 255.0f), 0));   // would be 1 with std::round
    TEST_CASE(EXPECT_EQ(RangeCast<uint8_t>(1.5f / 255.0f), 2));   // round half to even
    TEST_CASE(EXPECT_EQ(RangeCast<uint8_t>(2.5f / 255.0f), 2));   // round half to even (down)
    TEST_CASE(EXPECT_EQ(RangeCast<int8_t>(0.5f / 127.0f), 0));    // signed: same rounding rule
    TEST_CASE(EXPECT_EQ(RangeCast<int8_t>(-0.5f / 127.0f), 0));
    TEST_CASE(EXPECT_EQ(RangeCast<int8_t>(-1.5f / 127.0f), -2));

    // ----- Double precision in float -> int -----
    TEST_CASE(EXPECT_EQ(RangeCast<int>(0.5), std::numeric_limits<int>::max() / 2 + 1));
    TEST_CASE(EXPECT_EQ(RangeCast<int>(0.0), 0));
    TEST_CASE(EXPECT_EQ(RangeCast<uint>(0.0), 0u));

    // ----- int -> float clamping: signed min hits the -1.008... clamp -----
    // numeric_limits<int8_t>::min() / max() = -128 / 127 = -1.0078..., must clamp to -1.
    TEST_CASE(EXPECT_EQ(RangeCast<float>(int8_t{-128}), -1.0f));
    TEST_CASE(EXPECT_EQ(RangeCast<float>(int8_t{127}), 1.0f));
    TEST_CASE(EXPECT_EQ(RangeCast<float>(int8_t{0}), 0.0f));
    TEST_CASE(EXPECT_EQ(RangeCast<float>(int16_t{-32768}), -1.0f));
    TEST_CASE(EXPECT_EQ(RangeCast<float>(int16_t{32767}), 1.0f));

    // ----- uint -> float -----
    TEST_CASE(EXPECT_EQ(RangeCast<float>(uint8_t{255}), 1.0f));
    TEST_CASE(EXPECT_EQ(RangeCast<float>(uint8_t{0}), 0.0f));
    TEST_CASE(EXPECT_EQ(RangeCast<float>(uint16_t{65535}), 1.0f));

    // ----- Integer -> integer falls back to SaturateCast -----
    TEST_CASE(EXPECT_EQ(RangeCast<int8_t>(int32_t{300}), 127));
    TEST_CASE(EXPECT_EQ(RangeCast<uint8_t>(int32_t{-1}), 0));

    // ----- Vector types -----
    // float -> uchar4: 0.0 -> 0, 0.5 -> 128, 1.0 -> 255 (with banker's rounding at half)
    TEST_CASE(EXPECT_TRUE(
        (RangeCast<uchar4>(float4{0.0f, 0.5f, 1.0f, -0.5f}) == uchar4{0, 128, 255, 0})));
    // uchar4 -> float4: 0 -> 0.0, 255 -> 1.0
    {
        float4 result = RangeCast<float4>(uchar4{0, 128, 255, 64});
        TEST_CASE(EXPECT_EQ(result.x, 0.0f));
        TEST_CASE(EXPECT_EQ(result.z, 1.0f));
        TEST_CASE(EXPECT_TRUE(std::abs(result.y - (128.0f / 255.0f)) < 1e-6f));
        TEST_CASE(EXPECT_TRUE(std::abs(result.w - (64.0f / 255.0f)) < 1e-6f));
    }
    // 2- and 3-element vectors
    TEST_CASE(EXPECT_TRUE((RangeCast<uchar2>(float2{0.5f, -10.0f}) == uchar2{128, 0})));
    TEST_CASE(EXPECT_TRUE((RangeCast<char3>(float3{1.0f, -1.0f, 0.0f}) == char3{127, -127, 0})));

    // clang-format on

    TEST_CASES_END();
}