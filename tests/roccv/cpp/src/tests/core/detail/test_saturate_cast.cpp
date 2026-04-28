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

    TEST_CASE(EXPECT_EQ(SaturateCast<int>(1.0f), 1));
    TEST_CASE(EXPECT_EQ(SaturateCast<int>(-1.0f), -1));
    TEST_CASE(EXPECT_EQ(SaturateCast<uint>(1.0f), 1));
    TEST_CASE(EXPECT_EQ(SaturateCast<uint>(-1.0f), 0));
    TEST_CASE(EXPECT_EQ(SaturateCast<float>(1), 1.0f));
    TEST_CASE(EXPECT_EQ(SaturateCast<float>(-1), -1.0f));
    TEST_CASE(EXPECT_EQ(SaturateCast<double>(1), 1.0));
    TEST_CASE(EXPECT_EQ(SaturateCast<double>(-1), -1.0));

    // Test numeric limits
    TEST_CASE(EXPECT_EQ(SaturateCast<int>(std::numeric_limits<float>::max()), std::numeric_limits<int>::max()));
    TEST_CASE(EXPECT_EQ(SaturateCast<uint8_t>(std::numeric_limits<float>::max()), std::numeric_limits<uint8_t>::max()));
    TEST_CASE(EXPECT_EQ(SaturateCast<long>(std::numeric_limits<float>::max()), std::numeric_limits<long>::max()));
    TEST_CASE(EXPECT_EQ(SaturateCast<ulong>(std::numeric_limits<float>::lowest()), 0UL));

    // Test vectorized types
    TEST_CASE(EXPECT_TRUE((SaturateCast<float4>(uchar4{255, 128, 0, 255}) == float4{255.0f, 128.0f, 0.0f, 255.0f})));
    TEST_CASE(EXPECT_TRUE(
        (SaturateCast<float4>(char4{-128, -128, -128, -128}) == float4{-128.0f, -128.0f, -128.0f, -128.0f})));

    // ----- Rounding mode: must be IEEE half-to-even (banker's rounding) -----
    // These regression-guard against accidentally switching back to std::round
    // (half-away-from-zero), which would diverge from the device fast-paths.
    TEST_CASE(EXPECT_EQ(SaturateCast<int>(0.5f), 0));  // halfway -> nearest even (0)
    TEST_CASE(EXPECT_EQ(SaturateCast<int>(1.5f), 2));  // halfway -> nearest even (2)
    TEST_CASE(EXPECT_EQ(SaturateCast<int>(2.5f), 2));  // halfway -> nearest even (2)
    TEST_CASE(EXPECT_EQ(SaturateCast<int>(-0.5f), 0));
    TEST_CASE(EXPECT_EQ(SaturateCast<int>(-1.5f), -2));
    TEST_CASE(EXPECT_EQ(SaturateCast<int>(-2.5f), -2));
    // Same rounding rules in the 8/16-bit clamp-then-round path.
    TEST_CASE(EXPECT_EQ(SaturateCast<uint8_t>(0.5f), 0));
    TEST_CASE(EXPECT_EQ(SaturateCast<uint8_t>(1.5f), 2));
    TEST_CASE(EXPECT_EQ(SaturateCast<uint8_t>(2.5f), 2));
    TEST_CASE(EXPECT_EQ(SaturateCast<int16_t>(-1.5f), -2));
    // Non-half values should still round to nearest as expected.
    TEST_CASE(EXPECT_EQ(SaturateCast<int>(1.4f), 1));
    TEST_CASE(EXPECT_EQ(SaturateCast<int>(1.6f), 2));
    TEST_CASE(EXPECT_EQ(SaturateCast<int>(-1.4f), -1));
    TEST_CASE(EXPECT_EQ(SaturateCast<int>(-1.6f), -2));

    // ----- Double precision: must NOT be silently truncated to float -----
    TEST_CASE(EXPECT_EQ(SaturateCast<int>(1234567890.7), 1234567891));
    TEST_CASE(EXPECT_EQ(SaturateCast<int>(-1234567890.7), -1234567891));
    TEST_CASE(EXPECT_EQ(SaturateCast<int>(16777217.0), 16777217));          // 2^24+1, not exact in float
    TEST_CASE(EXPECT_EQ(SaturateCast<int64_t>(1234567890.5), 1234567890));  // half-to-even

    // ----- Float clamping: out-of-range floats clamp to numeric limits -----
    TEST_CASE(EXPECT_EQ(SaturateCast<uint8_t>(300.0f), 255));
    TEST_CASE(EXPECT_EQ(SaturateCast<uint8_t>(-1.0f), 0));
    TEST_CASE(EXPECT_EQ(SaturateCast<uint8_t>(-100.0f), 0));
    TEST_CASE(EXPECT_EQ(SaturateCast<int8_t>(200.0f), 127));
    TEST_CASE(EXPECT_EQ(SaturateCast<int8_t>(-200.0f), -128));
    TEST_CASE(EXPECT_EQ(SaturateCast<int16_t>(40000.0f), 32767));
    TEST_CASE(EXPECT_EQ(SaturateCast<int16_t>(-40000.0f), -32768));
    TEST_CASE(EXPECT_EQ(SaturateCast<uint16_t>(70000.0f), 65535));
    TEST_CASE(EXPECT_EQ(SaturateCast<uint16_t>(-1.0f), 0));

    // ----- Integer narrowing: same signedness -----
    TEST_CASE(EXPECT_EQ(SaturateCast<int8_t>(int32_t{300}), 127));
    TEST_CASE(EXPECT_EQ(SaturateCast<int8_t>(int32_t{-300}), -128));
    TEST_CASE(EXPECT_EQ(SaturateCast<int8_t>(int32_t{42}), 42));  // in-range, passthrough
    TEST_CASE(EXPECT_EQ(SaturateCast<uint8_t>(uint32_t{300}), 255));
    TEST_CASE(EXPECT_EQ(SaturateCast<uint8_t>(uint32_t{42}), 42));
    TEST_CASE(EXPECT_EQ(SaturateCast<int16_t>(int64_t{-100000}), -32768));

    // ----- Integer cross-signedness narrowing -----
    // Signed -> unsigned, big -> small: clamp negatives to 0, big to max
    TEST_CASE(EXPECT_EQ(SaturateCast<uint8_t>(int32_t{-1}), 0));
    TEST_CASE(EXPECT_EQ(SaturateCast<uint8_t>(int32_t{300}), 255));
    TEST_CASE(EXPECT_EQ(SaturateCast<uint8_t>(int32_t{42}), 42));
    // Unsigned -> signed: clamp values exceeding signed max
    TEST_CASE(EXPECT_EQ(SaturateCast<int8_t>(uint32_t{300}), 127));
    TEST_CASE(EXPECT_EQ(SaturateCast<int8_t>(uint32_t{42}), 42));
    TEST_CASE(EXPECT_EQ(SaturateCast<int16_t>(uint32_t{70000}), 32767));

    // ----- Integer cross-signedness widening -----
    // Signed -> unsigned, small to big: clamp negatives to 0
    TEST_CASE(EXPECT_EQ(SaturateCast<uint32_t>(int8_t{-1}), 0u));
    TEST_CASE(EXPECT_EQ(SaturateCast<uint32_t>(int8_t{-128}), 0u));
    TEST_CASE(EXPECT_EQ(SaturateCast<uint32_t>(int8_t{42}), 42u));
    // Unsigned -> signed widening: always representable, no clamping
    TEST_CASE(EXPECT_EQ(SaturateCast<int32_t>(uint8_t{255}), 255));
    TEST_CASE(EXPECT_EQ(SaturateCast<int32_t>(uint8_t{0}), 0));

    // ----- Same-type early-return path -----
    TEST_CASE(EXPECT_EQ(SaturateCast<int>(int{42}), 42));
    TEST_CASE(EXPECT_EQ(SaturateCast<float>(1.5f), 1.5f));
    TEST_CASE(EXPECT_EQ(SaturateCast<uint8_t>(uint8_t{200}), 200));

    // ----- Additional vector coverage: 2- and 3-element types, integer narrowing -----
    TEST_CASE(EXPECT_TRUE((SaturateCast<uchar2>(int2{300, -50}) == uchar2{255, 0})));
    TEST_CASE(EXPECT_TRUE(
        (SaturateCast<uchar3>(float3{300.0f, -10.0f, 127.5f}) == uchar3{255, 0, 128})));  // 127.5 rounds to even (128)
    TEST_CASE(EXPECT_TRUE((SaturateCast<char4>(float4{200.0f, -200.0f, 0.5f, -0.5f}) == char4{127, -128, 0, 0})));

    TEST_CASES_END();
}