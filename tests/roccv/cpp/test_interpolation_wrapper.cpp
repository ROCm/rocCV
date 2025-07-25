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
#include <core/detail/type_traits.hpp>
#include <core/wrappers/interpolation_wrapper.hpp>

#include "test_helpers.hpp"

using namespace roccv;
using namespace roccv::tests;

namespace {
template <typename T, eBorderType BorderType>
T GoldenInterpolationAt(BorderWrapper<T, BorderType> input, int64_t sample, int64_t y, int64_t x,
                        const eInterpolationType interp) {
    switch (interp) {
        case eInterpolationType::INTERP_TYPE_NEAREST: {
            // Nearest neighbor interpolation. Rounds given floating point values to the nearest integer.
            return input.at(sample, lroundf(y), lroundf(x), 0);
        }

        default:
            return input.at(sample, lroundf(y), lroundf(x), 0);
    }
}

template <typename T, eBorderType BorderType, eInterpolationType InterpType>
void TestCorrectness(int64_t batchSize, Size2D imageSize, float4 borderValue, float idxDelta) {
    // Convert borderValue into the same type as image data
    T borderVal = detail::RangeCast<T>(borderValue);

    int channels = detail::NumElements<T>;
    size_t numElements = batchSize * imageSize.h * imageSize.w * channels;

    std::vector<detail::BaseType<T>> input(numElements);
    std::vector<detail::BaseType<T>> actualOutput;
    std::vector<detail::BaseType<T>> goldenOutput;

    // Use roccv::InterpolationWrapper to get actual output
    InterpolationWrapper<T, BorderType, InterpType> actualWrap(
        (BorderWrapper<T, BorderType>(ImageWrapper<T>(input, batchSize, imageSize.w, imageSize.h), borderVal)));
    BorderWrapper<T, BorderType> goldenWrap(ImageWrapper<T>(input, batchSize, imageSize.w, imageSize.h), borderVal);

    for (int b = 0; b < batchSize; b++) {
        for (float y = 0; y < imageSize.h; y += idxDelta) {
            for (float x = 0; x < imageSize.w; x += idxDelta) {
                // Get actual result from interpolation wrapper
                T actualVal = actualWrap.at(b, y, x, 0);
                T goldenVal = GoldenInterpolationAt(goldenWrap, b, y, x, eInterpolationType::INTERP_TYPE_NEAREST);

                for (int c = 0; c < channels; c++) {
                    actualOutput.push_back(detail::GetElement(actualVal, c));
                    goldenOutput.push_back(detail::GetElement(goldenVal, c));
                }
            }
        }
    }

    CompareVectorsNear(actualOutput, goldenOutput);
}
}  // namespace

eTestStatusType test_interpolation_wrapper(int argc, char **argv) {
    TEST_CASES_BEGIN();

    // clang-format off
    TEST_CASE((TestCorrectness<uchar3, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_NEAREST>(1, {20, 53}, make_float4(0, 0, 0, 1), 0.5f)));
    // clang-format on

    TEST_CASES_END();
}