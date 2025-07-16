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

#include <core/detail/type_traits.hpp>
#include <core/wrappers/image_wrapper.hpp>

#include "test_helpers.hpp"

using namespace roccv;
using namespace roccv::tests;

namespace {

template <typename T, typename BT = detail::BaseType<T>>
void TestCorrectness(int numImages, Size2D size) {
    int numElements = numImages * size.w * size.h;
    const int channels = detail::NumElements<T>;

    std::vector<BT> ref(numElements);
    FillVector(ref);

    ImageWrapper<T> input(ref, numImages, size.w, size.h);

    // To determine if coordinates are pointing to the proper values in memory, iterate through the reference vector
    // element-by-element and iterate through the ImageWrapper coordinate-wise. All values should be the same if
    // everything lines up.

    int i = 0;

    for (int b = 0; b < numImages; ++b) {
        for (int y = 0; y < size.h; ++y) {
            for (int x = 0; x < size.w; ++x) {
                for (int c = 0; c < channels; ++c) {
                    BT wrapperVal = detail::GetElement(input.at(b, y, x, 0), c);
                    EXPECT_EQ(wrapperVal, ref[i]);
                    i++;
                }
            }
        }
    }
}

template <typename T>
void TestImageWrapperConstructor(int imageCount, Size2D imageSize, ImageFormat format) {
    Tensor input(imageCount, imageSize, format);
    ImageWrapper<T> wrapper(input);

    EXPECT_EQ(wrapper.batches(), imageCount);
    EXPECT_EQ(wrapper.channels(), format.channels());
    EXPECT_EQ(wrapper.width(), imageSize.w);
    EXPECT_EQ(wrapper.height(), imageSize.h);
}
}  // namespace

eTestStatusType test_image_wrapper(int argc, char** argv) {
    TEST_CASES_BEGIN();

    TEST_CASE(TestImageWrapperConstructor<uchar3>(2, {54, 67}, FMT_RGB8));

    TEST_CASE(TestCorrectness<uchar1>(1, {10, 10}));
    TEST_CASE(TestCorrectness<uchar3>(2, {43, 9}));
    TEST_CASE(TestCorrectness<uchar4>(3, {1, 93}));

    TEST_CASE(TestCorrectness<char1>(1, {10, 10}));
    TEST_CASE(TestCorrectness<char3>(2, {43, 9}));
    TEST_CASE(TestCorrectness<char4>(3, {1, 93}));

    TEST_CASE(TestCorrectness<short1>(1, {10, 10}));
    TEST_CASE(TestCorrectness<short2>(2, {43, 9}));
    TEST_CASE(TestCorrectness<short3>(3, {1, 93}));

    TEST_CASE(TestCorrectness<uint1>(1, {10, 10}));
    TEST_CASE(TestCorrectness<uint3>(2, {43, 9}));
    TEST_CASE(TestCorrectness<uint4>(3, {1, 93}));

    TEST_CASE(TestCorrectness<int1>(1, {10, 10}));
    TEST_CASE(TestCorrectness<int3>(2, {43, 9}));
    TEST_CASE(TestCorrectness<int4>(3, {1, 93}));

    TEST_CASE(TestCorrectness<float1>(1, {10, 10}));
    TEST_CASE(TestCorrectness<float3>(2, {43, 9}));
    TEST_CASE(TestCorrectness<float4>(3, {1, 93}));

    TEST_CASES_END();
}