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

#include "core/detail/vector_utils.hpp"
#include "test_helpers.hpp"

using namespace roccv;
using namespace roccv::detail;
using namespace roccv::tests;

namespace {

/**
 * @brief Golden model for Bilinear pixel interpolation. This implementation is based on the repeated linear computation
 * method described in https://en.wikipedia.org/wiki/Bilinear_interpolation.
 *
 * @tparam T Image datatype.
 * @tparam BorderType Border type for boundary conditions.
 * @param input BorderWrapper containing input data.
 * @param sample The sample within the image batch to index into.
 * @param y A floating point describing the y position of the selected image.
 * @param x A floating point describing the x position of the selected image.
 * @return T The interpolated pixel.
 */
template <typename T, eBorderType BorderType>
T GoldenLinear(BorderWrapper<BorderType, ImageWrapper<T>> input, int64_t sample, float y, float x) {
    // Defines the vectorized float type for intermediate calculations.
    using WorkType = detail::MakeType<float, detail::NumComponents<T>>;

    // Grab four known points around the given area
    int64_t x1 = static_cast<int64_t>(floorf(x));
    int64_t x2 = x1 + 1;
    int64_t y1 = static_cast<int64_t>(floorf(y));
    int64_t y2 = y1 + 1;

    // Values of each of the known points, these are casted to a floating point representation as we require
    // floating point arithmetic for linear interpolation.
    WorkType q11 = detail::RangeCast<WorkType>(input.at(sample, y1, x1, 0));
    WorkType q12 = detail::RangeCast<WorkType>(input.at(sample, y2, x1, 0));
    WorkType q21 = detail::RangeCast<WorkType>(input.at(sample, y1, x2, 0));
    WorkType q22 = detail::RangeCast<WorkType>(input.at(sample, y2, x2, 0));

    // Perform linear interpolation on the x-axis first
    WorkType fxy1 = (x2 - x) * q11 + (x - x1) * q21;
    WorkType fxy2 = (x2 - x) * q12 + (x - x1) * q22;

    // Then, begin interpolation in the y-direction to obtain desired result
    WorkType fxy = (y2 - y) * fxy1 + (y - y1) * fxy2;

    // Cast values back to type T.
    return detail::RangeCast<T>(fxy);
}

/**
 * @brief Golden model for Nearest Neighbor interpolation. This is not based on any implementation due to its
 * simplicity. Rounds coordinates to the nearest integer.
 *
 * @tparam T Image datatype.
 * @tparam BorderType Border type for boundary conditions.
 * @param input BorderWrapper containing input data.
 * @param sample The sample within the image batch to index into.
 * @param y A floating point describing the y position of the selected image.
 * @param x A floating point describing the x position of the selected image.
 * @return T The interpolated pixel.
 */
template <typename T, eBorderType BorderType>
T GoldenNearest(BorderWrapper<BorderType, ImageWrapper<T>> input, int64_t sample, float y, float x) {
    // Nearest neighbor interpolation. Rounds given floating point values to the nearest integer.
    return input.at(sample, lroundf(y), lroundf(x), 0);
}

/**
 * @brief This function calculates the weighting coefficients for the Catmull-Rom cubic interpolation.
 * @param dist The distance between the current point and the previous data point.
 * @param weight The pointer to the array of weights.
 * @return None.
 */
void CalBicubicWeights(float dist, float* weight) {
    const float A = -0.5f;  // Note OpenCV sets alpha to -0.75f

    weight[0] = ((A * (dist + 1) - 5 * A) * (dist + 1) + 8 * A) * (dist + 1) - 4 * A;
    weight[1] = ((A + 2) * dist - (A + 3)) * dist * dist + 1;
    weight[2] = ((A + 2) * (1 - dist) - (A + 3)) * (1 - dist) * (1 - dist) + 1;
    weight[3] = 1.f - weight[0] - weight[1] - weight[2];
}

/**
 * @brief Golden model for Bicubic interpolation. This is the Catmull-Rom cubic interpolation commonly used in CV
 * libraries.
 *
 * @tparam T Image datatype.
 * @tparam BorderType Border type for boundary conditions.
 * @param input BorderWrapper containing input data.
 * @param sample The sample within the image batch to index into.
 * @param y A floating point describing the y position of the selected image.
 * @param x A floating point describing the x position of the selected image.
 * @return T The interpolated pixel.
 */
template <typename T, eBorderType BorderType>
T GoldenBicubic(BorderWrapper<BorderType, ImageWrapper<T>> input, int64_t sample, float y, float x) {
    // Defines the vectorized float type for intermediate calculations.
    using WorkType = detail::MakeType<float, detail::NumComponents<T>>;

    // Integer coordinates for pixel (x, y)
    int64_t intX = static_cast<int64_t>(floorf(x));
    int64_t intY = static_cast<int64_t>(floorf(y));

    // Calculate weights
    float weightX[4], weightY[4];
    CalBicubicWeights(x - intX, weightX);
    CalBicubicWeights(y - intY, weightY);

    // Weighted sum
    WorkType sum = SetAll<WorkType>(0.0f);
    for (int indexY = -1; indexY <= 2; indexY++) {
        for (int indexX = -1; indexX <= 2; indexX++) {
            sum += detail::RangeCast<WorkType>(input.at(sample, intY + indexY, intX + indexX, 0)) *
                   (weightX[indexX + 1] * weightY[indexY + 1]);
        }
    }

    return detail::RangeCast<T>(sum);
}

/**
 * @brief General Golden model for image interpolation. Does interpolation, but allows you to select a given
 * interpolation type.
 *
 * @tparam T Image datatype.
 * @tparam BorderType Border type for boundary conditions.
 * @param input BorderWrapper containing input data.
 * @param sample The sample within the image batch to index into.
 * @param y A floating point describing the y position of the selected image.
 * @param x A floating point describing the x position of the selected image.
 * @param interp The interpolation type to use.
 * @return T The interpolated pixel.
 */
template <typename T, eBorderType BorderType>
T GoldenInterpolationAt(BorderWrapper<BorderType, ImageWrapper<T>> input, int64_t sample, float y, float x,
                        eInterpolationType interp) {
    switch (interp) {
        case eInterpolationType::INTERP_TYPE_NEAREST:
            return GoldenNearest(input, sample, y, x);

        case eInterpolationType::INTERP_TYPE_LINEAR:
            return GoldenLinear(input, sample, y, x);

        case eInterpolationType::INTERP_TYPE_CUBIC:
            return GoldenBicubic(input, sample, y, x);

        default:
            throw std::runtime_error("Interpolation type does not have a golden model yet.");
    }
}

/**
 * @brief Compares the golden model with the rocCV version of image interpolation. This will iterate over the entire
 * randomly generated image, and compare each pixel in the actual and golden results.
 *
 * @tparam T Image datatype.
 * @tparam BorderType Border type to use for out-of-bounds conditions.
 * @tparam InterpType Interpolation type to test.
 * @param batchSize Number of images in the batch.
 * @param imageSize With and height of images in the batch.
 * @param borderValue Fallback value when CONSTANT border mode is used.
 * @param idxDelta A floating point delta to iterate by when iterating through the original image and interpolating
 * pixels. For example, given an image dimension of 2 and a delta of 0.25, coordinates at 0.0, 0.25, 0.5,
 * 0.75, 1.0, 1.25, 1.5, 1.75, 2.0 will be tested.
 */
template <typename T, eBorderType BorderType, eInterpolationType InterpType>
void TestCorrectness(int64_t batchSize, Size2D imageSize, float4 borderValue, float idxDelta) {
    // Convert borderValue into the same type as image data
    T borderVal = detail::RangeCast<T>(borderValue);

    int channels = detail::NumElements<T>;
    size_t numElements = batchSize * imageSize.h * imageSize.w * channels;

    std::vector<detail::BaseType<T>> input(numElements);
    FillVector(input);

    std::vector<detail::BaseType<T>> actualOutput;
    std::vector<detail::BaseType<T>> goldenOutput;

    // Use roccv::InterpolationWrapper to get actual output
    auto actualWrap = MakeInterpolationWrapper<InterpType>(
        MakeBorderWrapper<BorderType>(ImageWrapper<T>(input, batchSize, imageSize.w, imageSize.h), borderVal));
    auto goldenWrap =
        MakeBorderWrapper<BorderType>(ImageWrapper<T>(input, batchSize, imageSize.w, imageSize.h), borderVal);

    for (int b = 0; b < batchSize; b++) {
        for (float y = 0; y < imageSize.h; y += idxDelta) {
            for (float x = 0; x < imageSize.w; x += idxDelta) {
                // Get actual result from interpolation wrapper
                T actualVal = actualWrap.at(b, y, x, 0);
                T goldenVal = GoldenInterpolationAt(goldenWrap, b, y, x, InterpType);

                for (int c = 0; c < channels; c++) {
                    actualOutput.push_back(detail::GetElement(actualVal, c));
                    goldenOutput.push_back(detail::GetElement(goldenVal, c));
                }
            }
        }
    }
    if constexpr (std::is_integral_v<detail::BaseType<T>> && std::is_signed_v<detail::BaseType<T>> &&
                  sizeof(detail::BaseType<T>) == 4) {
        CompareVectorsNear(actualOutput, goldenOutput, NEAR_EQUAL_THRESHOLD * 2);
    } else {
        CompareVectorsNear(actualOutput, goldenOutput);
    }
}
}  // namespace

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;
    TEST_CASES_BEGIN();

    // clang-format off

    // Test nearest neighbor interpolation with all supported datatypes
    TEST_CASE((TestCorrectness<uchar1, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_NEAREST>(1, {20, 53}, make_float4(0, 0, 0, 1), 0.1f)));
    TEST_CASE((TestCorrectness<uchar3, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_NEAREST>(3, {38, 10}, make_float4(0, 0, 0, 1), 0.1f)));
    TEST_CASE((TestCorrectness<uchar4, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_NEAREST>(5, {65, 21}, make_float4(1, 0.5, 0.5, 0.5), 0.1f)));
    
    TEST_CASE((TestCorrectness<char1, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_NEAREST>(1, {20, 53}, make_float4(0, 0, 0, 1), 0.1f)));
    TEST_CASE((TestCorrectness<char3, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_NEAREST>(3, {38, 10}, make_float4(0, 0, 0, 1), 0.1f)));
    TEST_CASE((TestCorrectness<char4, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_NEAREST>(5, {65, 21}, make_float4(1, 0.5, 0.5, 1), 0.1f)));
    
    TEST_CASE((TestCorrectness<ushort1, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_NEAREST>(1, {20, 53}, make_float4(0, 0, 0, 1), 0.1f)));
    TEST_CASE((TestCorrectness<ushort3, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_NEAREST>(3, {38, 10}, make_float4(0, 0, 0, 1), 0.1f)));
    TEST_CASE((TestCorrectness<ushort4, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_NEAREST>(5, {65, 21}, make_float4(1, 0.5, 0.5, 1), 0.1f)));
    
    TEST_CASE((TestCorrectness<short1, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_NEAREST>(1, {20, 53}, make_float4(0, 0, 0, 1), 0.1f)));
    TEST_CASE((TestCorrectness<short3, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_NEAREST>(3, {38, 10}, make_float4(0, 0, 0, 1), 0.1f)));
    TEST_CASE((TestCorrectness<short4, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_NEAREST>(5, {65, 21}, make_float4(1, 0.5, 0.5, 1), 0.1f)));
    
    TEST_CASE((TestCorrectness<uint1, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_NEAREST>(1, {20, 53}, make_float4(0, 0, 0, 1), 0.1f)));
    TEST_CASE((TestCorrectness<uint3, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_NEAREST>(3, {38, 10}, make_float4(0, 0, 0, 1), 0.1f)));
    TEST_CASE((TestCorrectness<uint4, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_NEAREST>(5, {65, 21}, make_float4(1, 0.5, 0.5, 1), 0.1f)));
    
    TEST_CASE((TestCorrectness<int1, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_NEAREST>(1, {20, 53}, make_float4(0, 0, 0, 1), 0.1f)));
    TEST_CASE((TestCorrectness<int3, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_NEAREST>(3, {38, 10}, make_float4(0, 0, 0, 1), 0.1f)));
    TEST_CASE((TestCorrectness<int4, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_NEAREST>(5, {65, 21}, make_float4(1, 0.5, 0.5, 1), 0.1f)));

    TEST_CASE((TestCorrectness<float1, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_NEAREST>(1, {20, 53}, make_float4(0, 0, 0, 1), 0.1f)));
    TEST_CASE((TestCorrectness<float3, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_NEAREST>(3, {38, 10}, make_float4(0, 0, 0, 1), 0.1f)));
    TEST_CASE((TestCorrectness<float4, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_NEAREST>(5, {65, 21}, make_float4(1, 0.5, 0.5, 1), 0.1f)));


    // Test bilinear interpolation
    TEST_CASE((TestCorrectness<uchar1, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_LINEAR>(1, {20, 53}, make_float4(0, 0, 0, 1), 0.1f)));
    TEST_CASE((TestCorrectness<uchar3, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_LINEAR>(3, {38, 10}, make_float4(0, 0, 0, 1), 0.1f)));
    TEST_CASE((TestCorrectness<uchar4, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_LINEAR>(5, {65, 21}, make_float4(1, 0.5, 0.5, 0.5), 0.1f)));
    
    TEST_CASE((TestCorrectness<char1, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_LINEAR>(1, {20, 53}, make_float4(0, 0, 0, 1), 0.1f)));
    TEST_CASE((TestCorrectness<char3, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_LINEAR>(3, {38, 10}, make_float4(0, 0, 0, 1), 0.1f)));
    TEST_CASE((TestCorrectness<char4, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_LINEAR>(5, {65, 21}, make_float4(1, 0.5, 0.5, 1), 0.1f)));
    
    TEST_CASE((TestCorrectness<ushort1, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_LINEAR>(1, {20, 53}, make_float4(0, 0, 0, 1), 0.1f)));
    TEST_CASE((TestCorrectness<ushort3, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_LINEAR>(3, {38, 10}, make_float4(0, 0, 0, 1), 0.1f)));
    TEST_CASE((TestCorrectness<ushort4, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_LINEAR>(5, {65, 21}, make_float4(1, 0.5, 0.5, 1), 0.1f)));
    
    TEST_CASE((TestCorrectness<short1, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_LINEAR>(1, {20, 53}, make_float4(0, 0, 0, 1), 0.1f)));
    TEST_CASE((TestCorrectness<short3, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_LINEAR>(3, {38, 10}, make_float4(0, 0, 0, 1), 0.1f)));
    TEST_CASE((TestCorrectness<short4, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_LINEAR>(5, {65, 21}, make_float4(1, 0.5, 0.5, 1), 0.1f)));
    
    TEST_CASE((TestCorrectness<uint1, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_LINEAR>(1, {20, 53}, make_float4(0, 0, 0, 1), 0.1f)));
    TEST_CASE((TestCorrectness<uint3, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_LINEAR>(3, {38, 10}, make_float4(0, 0, 0, 1), 0.1f)));
    TEST_CASE((TestCorrectness<uint4, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_LINEAR>(5, {65, 21}, make_float4(1, 0.5, 0.5, 1), 0.1f)));
    
    TEST_CASE((TestCorrectness<int1, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_LINEAR>(1, {20, 53}, make_float4(0, 0, 0, 1), 0.1f)));
    TEST_CASE((TestCorrectness<int3, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_LINEAR>(3, {38, 10}, make_float4(0, 0, 0, 1), 0.1f)));
    TEST_CASE((TestCorrectness<int4, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_LINEAR>(5, {65, 21}, make_float4(1, 0.5, 0.5, 1), 0.1f)));

    TEST_CASE((TestCorrectness<float1, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_LINEAR>(1, {20, 53}, make_float4(0, 0, 0, 1), 0.1f)));
    TEST_CASE((TestCorrectness<float3, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_LINEAR>(3, {38, 10}, make_float4(0, 0, 0, 1), 0.1f)));
    TEST_CASE((TestCorrectness<float4, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_LINEAR>(5, {65, 21}, make_float4(1, 0.5, 0.5, 1), 0.1f)));

    // Test bicubic interpolation
    TEST_CASE((TestCorrectness<uchar1, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_CUBIC>(1, {20, 53}, make_float4(0, 0, 0, 1), 0.1f)));
    TEST_CASE((TestCorrectness<uchar3, eBorderType::BORDER_TYPE_REFLECT, eInterpolationType::INTERP_TYPE_CUBIC>(3, {38, 10}, make_float4(0, 0, 0, 1), 0.1f)));
    TEST_CASE((TestCorrectness<uchar4, eBorderType::BORDER_TYPE_REFLECT101, eInterpolationType::INTERP_TYPE_CUBIC>(5, {65, 21}, make_float4(1, 0.5, 0.5, 0.5), 0.1f)));

    TEST_CASE((TestCorrectness<char1, eBorderType::BORDER_TYPE_WRAP, eInterpolationType::INTERP_TYPE_CUBIC>(1, {20, 53}, make_float4(0, 0, 0, 1), 0.1f)));
    TEST_CASE((TestCorrectness<char3, eBorderType::BORDER_TYPE_REPLICATE, eInterpolationType::INTERP_TYPE_CUBIC>(3, {38, 10}, make_float4(0, 0, 0, 1), 0.1f)));
    TEST_CASE((TestCorrectness<char4, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_CUBIC>(5, {65, 21}, make_float4(1, 0.5, 0.5, 1), 0.1f)));

    TEST_CASE((TestCorrectness<ushort1, eBorderType::BORDER_TYPE_REFLECT, eInterpolationType::INTERP_TYPE_CUBIC>(1, {20, 53}, make_float4(0, 0, 0, 1), 0.1f)));
    TEST_CASE((TestCorrectness<ushort3, eBorderType::BORDER_TYPE_REPLICATE, eInterpolationType::INTERP_TYPE_CUBIC>(3, {38, 10}, make_float4(0, 0, 0, 1), 0.1f)));
    TEST_CASE((TestCorrectness<ushort4, eBorderType::BORDER_TYPE_WRAP, eInterpolationType::INTERP_TYPE_CUBIC>(5, {65, 21}, make_float4(1, 0.5, 0.5, 1), 0.1f)));

    TEST_CASE((TestCorrectness<short1, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_CUBIC>(1, {20, 53}, make_float4(0, 0, 0, 1), 0.1f)));
    TEST_CASE((TestCorrectness<short3, eBorderType::BORDER_TYPE_REFLECT, eInterpolationType::INTERP_TYPE_CUBIC>(3, {38, 10}, make_float4(0, 0, 0, 1), 0.1f)));
    TEST_CASE((TestCorrectness<short4, eBorderType::BORDER_TYPE_REPLICATE, eInterpolationType::INTERP_TYPE_CUBIC>(5, {65, 21}, make_float4(1, 0.5, 0.5, 1), 0.1f)));

    TEST_CASE((TestCorrectness<uint1, eBorderType::BORDER_TYPE_REFLECT, eInterpolationType::INTERP_TYPE_CUBIC>(1, {20, 53}, make_float4(0, 0, 0, 1), 0.1f)));
    TEST_CASE((TestCorrectness<uint3, eBorderType::BORDER_TYPE_WRAP, eInterpolationType::INTERP_TYPE_CUBIC>(3, {38, 10}, make_float4(0, 0, 0, 1), 0.1f)));
    TEST_CASE((TestCorrectness<uint4, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_CUBIC>(5, {65, 21}, make_float4(1, 0.5, 0.5, 1), 0.1f)));

    TEST_CASE((TestCorrectness<int1, eBorderType::BORDER_TYPE_WRAP, eInterpolationType::INTERP_TYPE_CUBIC>(1, {20, 53}, make_float4(0, 0, 0, 1), 0.1f)));
    TEST_CASE((TestCorrectness<int3, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_CUBIC>(3, {38, 10}, make_float4(0, 0, 0, 1), 0.1f)));
    TEST_CASE((TestCorrectness<int4, eBorderType::BORDER_TYPE_REFLECT101, eInterpolationType::INTERP_TYPE_CUBIC>(5, {65, 21}, make_float4(1, 0.5, 0.5, 1), 0.1f)));

    TEST_CASE((TestCorrectness<float1, eBorderType::BORDER_TYPE_REFLECT, eInterpolationType::INTERP_TYPE_CUBIC>(1, {20, 53}, make_float4(0, 0, 0, 1), 0.1f)));
    TEST_CASE((TestCorrectness<float3, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_CUBIC>(3, {38, 10}, make_float4(0, 0, 0, 1), 0.1f)));
    TEST_CASE((TestCorrectness<float4, eBorderType::BORDER_TYPE_WRAP, eInterpolationType::INTERP_TYPE_CUBIC>(5, {65, 21}, make_float4(1, 0.5, 0.5, 1), 0.1f)));
    // clang-format on

    TEST_CASES_END();
}