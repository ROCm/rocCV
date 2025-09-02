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

#include <core/detail/swizzling.hpp>
#include <core/detail/type_traits.hpp>

#include "test_helpers.hpp"

using namespace roccv;
using namespace roccv::tests;

namespace {

/**
 * @brief Tests correctness for the Swizzle helper function.
 *
 * @tparam T The type of the vector to swizzle.
 * @tparam SwizzlePattern The swizzling pattern to test.
 * @param input The input vector to swizzle.
 * @param expected The expected vector to compare against after input has been rearranged.
 */
template <typename T, eSwizzle SwizzlePattern>
void TestCorrectness(T input, T expected) {
    T output = detail::Swizzle<SwizzlePattern>(input);
    for (int i = 0; i < detail::NumElements<T>; i++) {
        EXPECT_EQ(output[i], expected[i]);
    }
}
}  // namespace

eTestStatusType test_swizzling(int argc, char **argv) {
    TEST_CASES_BEGIN();

    // ZYXW swizzling
    TEST_CASE((TestCorrectness<uchar3, eSwizzle::ZYXW>({1, 2, 3}, {3, 2, 1})));
    TEST_CASE((TestCorrectness<uchar4, eSwizzle::ZYXW>({1, 2, 3, 4}, {3, 2, 1, 4})));
    TEST_CASE((TestCorrectness<char3, eSwizzle::ZYXW>({1, 2, 3}, {3, 2, 1})));
    TEST_CASE((TestCorrectness<char4, eSwizzle::ZYXW>({1, 2, 3, 4}, {3, 2, 1, 4})));
    TEST_CASE((TestCorrectness<ushort3, eSwizzle::ZYXW>({1, 2, 3}, {3, 2, 1})));
    TEST_CASE((TestCorrectness<ushort4, eSwizzle::ZYXW>({1, 2, 3, 4}, {3, 2, 1, 4})));
    TEST_CASE((TestCorrectness<short3, eSwizzle::ZYXW>({1, 2, 3}, {3, 2, 1})));
    TEST_CASE((TestCorrectness<short4, eSwizzle::ZYXW>({1, 2, 3, 4}, {3, 2, 1, 4})));
    TEST_CASE((TestCorrectness<uint3, eSwizzle::ZYXW>({1, 2, 3}, {3, 2, 1})));
    TEST_CASE((TestCorrectness<uint4, eSwizzle::ZYXW>({1, 2, 3, 4}, {3, 2, 1, 4})));
    TEST_CASE((TestCorrectness<int3, eSwizzle::ZYXW>({1, 2, 3}, {3, 2, 1})));
    TEST_CASE((TestCorrectness<int4, eSwizzle::ZYXW>({1, 2, 3, 4}, {3, 2, 1, 4})));
    TEST_CASE((TestCorrectness<float3, eSwizzle::ZYXW>({1.0f, 2.0f, 3.0f}, {3.0f, 2.0f, 1.0f})));
    TEST_CASE((TestCorrectness<float4, eSwizzle::ZYXW>({1.0f, 2.0f, 3.0f, 4.0f}, {3.0f, 2.0f, 1.0f, 4.0f})));
    TEST_CASE((TestCorrectness<double3, eSwizzle::ZYXW>({1.0, 2.0, 3.0}, {3.0, 2.0, 1.0})));
    TEST_CASE((TestCorrectness<double4, eSwizzle::ZYXW>({1.0, 2.0, 3.0, 4.0}, {3.0, 2.0, 1.0, 4.0})));

    // XYZW swizzling (identity function)
    TEST_CASE((TestCorrectness<uchar3, eSwizzle::XYZW>({1, 2, 3}, {1, 2, 3})));
    TEST_CASE((TestCorrectness<uchar4, eSwizzle::XYZW>({1, 2, 3, 4}, {1, 2, 3, 4})));
    TEST_CASE((TestCorrectness<char3, eSwizzle::XYZW>({1, 2, 3}, {1, 2, 3})));
    TEST_CASE((TestCorrectness<char4, eSwizzle::XYZW>({1, 2, 3, 4}, {1, 2, 3, 4})));
    TEST_CASE((TestCorrectness<ushort3, eSwizzle::XYZW>({1, 2, 3}, {1, 2, 3})));
    TEST_CASE((TestCorrectness<ushort4, eSwizzle::XYZW>({1, 2, 3, 4}, {1, 2, 3, 4})));
    TEST_CASE((TestCorrectness<short3, eSwizzle::XYZW>({1, 2, 3}, {1, 2, 3})));
    TEST_CASE((TestCorrectness<short4, eSwizzle::XYZW>({1, 2, 3, 4}, {1, 2, 3, 4})));
    TEST_CASE((TestCorrectness<uint3, eSwizzle::XYZW>({1, 2, 3}, {1, 2, 3})));
    TEST_CASE((TestCorrectness<uint4, eSwizzle::XYZW>({1, 2, 3, 4}, {1, 2, 3, 4})));
    TEST_CASE((TestCorrectness<int3, eSwizzle::XYZW>({1, 2, 3}, {1, 2, 3})));
    TEST_CASE((TestCorrectness<int4, eSwizzle::XYZW>({1, 2, 3, 4}, {1, 2, 3, 4})));
    TEST_CASE((TestCorrectness<float3, eSwizzle::XYZW>({1.0f, 2.0f, 3.0f}, {1.0f, 2.0f, 3.0f})));
    TEST_CASE((TestCorrectness<float4, eSwizzle::XYZW>({1.0f, 2.0f, 3.0f, 4.0f}, {1.0f, 2.0f, 3.0f, 4.0f})));
    TEST_CASE((TestCorrectness<double3, eSwizzle::XYZW>({1.0, 2.0, 3.0}, {1.0, 2.0, 3.0})));
    TEST_CASE((TestCorrectness<double4, eSwizzle::XYZW>({1.0, 2.0, 3.0, 4.0}, {1.0, 2.0, 3.0, 4.0})));

    TEST_CASES_END();
}