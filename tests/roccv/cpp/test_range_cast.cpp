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

eTestStatusType test_range_cast(int argc, char **argv) {
    TEST_CASES_BEGIN();

    // clang-format off

    // Test float -> unsigned/signed integer casting
    TEST_CASE(EXPECT_EQ(RangeCast<int>(1.0f), std::numeric_limits<int>::max()));
    TEST_CASE(EXPECT_EQ(RangeCast<int>(-1.0f), std::numeric_limits<int>::min()));
    TEST_CASE(EXPECT_EQ(RangeCast<uint>(1.0f), std::numeric_limits<uint>::max()));
    TEST_CASE(EXPECT_EQ(RangeCast<uint>(-1.0f), 0));

    // Test unsigned/signed integer -> float casting
    TEST_CASE(EXPECT_EQ(RangeCast<float>(std::numeric_limits<int>::max()), 1.0f));
    TEST_CASE(EXPECT_EQ(RangeCast<float>(std::numeric_limits<int>::min()), -1.0f));
    TEST_CASE(EXPECT_EQ(RangeCast<float>(std::numeric_limits<uint>::max()), 1.0f));
    TEST_CASE(EXPECT_EQ(RangeCast<float>(0), 0.0f));

    // clang-format on

    TEST_CASES_END();
}