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
T GoldenInterpolationAt(BorderWrapper<T, BorderType> input, int64_t sample, float y, float x,
                        const eInterpolationType interp) {
    switch (interp) {
        case eInterpolationType::INTERP_TYPE_NEAREST: {
            // Nearest neighbor interpolation. Rounds given floating point values to the nearest integer.
            return input.at(sample, lroundf(y), lroundf(x), 0);
        }

        case eInterpolationType::INTERP_TYPE_LINEAR: {
            // Bilinear interpolation. Implementation based on the repeated linear interpolation computation method
            // described in https://en.wikipedia.org/wiki/Bilinear_interpolation.

            // Defines the vectorized float type for intermediate calculations.
            using WorkType = detail::MakeType<float, detail::NumComponents<T>>;

            // Grab four known points around the given area
            int64_t x1 = static_cast<int64_t>(floor(x));
            int64_t x2 = x1 + 1;
            int64_t y1 = static_cast<int64_t>(floor(y));
            int64_t y2 = y1 + 1;

            // Values of each of the known points, these are casted to a floating point representation as we require
            // floating point arithmetic for linear interpolation.
            //
            // Note: We do not need to normalize these values using a RangeCast, since all input values in this
            // calculation are within the same domain as type T.
            WorkType q11 = detail::StaticCast<WorkType>(input.at(sample, y1, x1, 0));
            WorkType q12 = detail::StaticCast<WorkType>(input.at(sample, y2, x1, 0));
            WorkType q21 = detail::StaticCast<WorkType>(input.at(sample, y1, x2, 0));
            WorkType q22 = detail::StaticCast<WorkType>(input.at(sample, y2, x2, 0));

            // Perform linear interpolation on the x-axis first
            WorkType fxy1 = (x2 - x) * q11 + (x - x1) * q21;
            WorkType fxy2 = (x2 - x) * q12 + (x - x1) * q22;

            // Then, begin interpolation in the y-direction to obtain desired result
            WorkType fxy = (y2 - y) * fxy1 + (y - y1) * fxy2;

            // Cast values back to type T, this essentially truncates the decimal place and ensures the resulting value
            // is clamped to the range of the domain of T.
            return detail::SaturateCast<T>(fxy);
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

    // Test nearest neighbor interpolation with all supported datatypes
    TEST_CASE((TestCorrectness<uchar1, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_NEAREST>(1, {20, 53}, make_float4(0, 0, 0, 1), 0.5f)));
    TEST_CASE((TestCorrectness<uchar3, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_NEAREST>(3, {38, 10}, make_float4(0, 0, 0, 1), 0.5f)));
    TEST_CASE((TestCorrectness<uchar4, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_NEAREST>(5, {65, 21}, make_float4(1, 0.5, 0.5, 1), 0.5f)));
    
    TEST_CASE((TestCorrectness<char1, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_NEAREST>(1, {20, 53}, make_float4(0, 0, 0, 1), 0.5f)));
    TEST_CASE((TestCorrectness<char3, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_NEAREST>(3, {38, 10}, make_float4(0, 0, 0, 1), 0.5f)));
    TEST_CASE((TestCorrectness<char4, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_NEAREST>(5, {65, 21}, make_float4(1, 0.5, 0.5, 1), 0.5f)));
    
    TEST_CASE((TestCorrectness<ushort1, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_NEAREST>(1, {20, 53}, make_float4(0, 0, 0, 1), 0.5f)));
    TEST_CASE((TestCorrectness<ushort3, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_NEAREST>(3, {38, 10}, make_float4(0, 0, 0, 1), 0.5f)));
    TEST_CASE((TestCorrectness<ushort4, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_NEAREST>(5, {65, 21}, make_float4(1, 0.5, 0.5, 1), 0.5f)));
    
    TEST_CASE((TestCorrectness<short1, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_NEAREST>(1, {20, 53}, make_float4(0, 0, 0, 1), 0.5f)));
    TEST_CASE((TestCorrectness<short3, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_NEAREST>(3, {38, 10}, make_float4(0, 0, 0, 1), 0.5f)));
    TEST_CASE((TestCorrectness<short4, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_NEAREST>(5, {65, 21}, make_float4(1, 0.5, 0.5, 1), 0.5f)));
    
    TEST_CASE((TestCorrectness<uint1, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_NEAREST>(1, {20, 53}, make_float4(0, 0, 0, 1), 0.5f)));
    TEST_CASE((TestCorrectness<uint3, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_NEAREST>(3, {38, 10}, make_float4(0, 0, 0, 1), 0.5f)));
    TEST_CASE((TestCorrectness<uint4, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_NEAREST>(5, {65, 21}, make_float4(1, 0.5, 0.5, 1), 0.5f)));
    
    TEST_CASE((TestCorrectness<int1, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_NEAREST>(1, {20, 53}, make_float4(0, 0, 0, 1), 0.5f)));
    TEST_CASE((TestCorrectness<int3, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_NEAREST>(3, {38, 10}, make_float4(0, 0, 0, 1), 0.5f)));
    TEST_CASE((TestCorrectness<int4, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_NEAREST>(5, {65, 21}, make_float4(1, 0.5, 0.5, 1), 0.5f)));

    TEST_CASE((TestCorrectness<float1, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_NEAREST>(1, {20, 53}, make_float4(0, 0, 0, 1), 0.5f)));
    TEST_CASE((TestCorrectness<float3, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_NEAREST>(3, {38, 10}, make_float4(0, 0, 0, 1), 0.5f)));
    TEST_CASE((TestCorrectness<float4, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_NEAREST>(5, {65, 21}, make_float4(1, 0.5, 0.5, 1), 0.5f)));

    // clang-format on

    TEST_CASES_END();
}