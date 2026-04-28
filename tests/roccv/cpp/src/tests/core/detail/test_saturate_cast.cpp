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

    TEST_CASES_END();
}