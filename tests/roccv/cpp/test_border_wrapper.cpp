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
#include <core/wrappers/border_wrapper.hpp>

#include "test_helpers.hpp"

using namespace roccv;
using namespace roccv::tests;
namespace {

/**
 * @brief Returns whether a coordinate would fall out of bounds.
 *
 * @param coordinate The coordinate value to check.
 * @param dimSize The size of the dimension to check along.
 * @return true if the coordinates are out of bounds.
 * @return false if the coordinates are in bounds.
 */
bool IsOutOfBounds(int32_t coordinate, int32_t dimSize) { return (coordinate < 0) || (coordinate >= dimSize); }

/**
 * @brief Calculate the coordinate of certain dimension (x or y) for the border pixel.
 * @param[in] u The coordinate of the border pixel.
 * @param[in] dimSize The size of the image in the dimension.
 * @param[in] borderMode The border mode to determine how the coordinate of the border pixel is mapped.
 * @return The in-image coordinate for the border pixel.
 */
int64_t GetCoordOfBorderPel(int64_t u, int64_t dimSize, const eBorderType borderMode) {
    int64_t v = u;
    switch (borderMode) {
        case eBorderType::BORDER_TYPE_REFLECT: {
            if (u < 0) {
                v = (-u - 1) % dimSize;
            } else if (u >= dimSize) {
                v = dimSize - (u % dimSize) - 1;
            } else {
                v = u;
            }
            break;
        }

        case eBorderType::BORDER_TYPE_REFLECT101: {
            if (dimSize == 1) {
                v = 0;
            } else {
                if (u < 0) {
                    v = (-u) % dimSize;
                } else if (u >= dimSize) {
                    v = u % dimSize;
                    v = (dimSize * 2 - 2 - v) % dimSize;
                } else {
                    v = u;
                }
            }
            break;
        }

        case eBorderType::BORDER_TYPE_REPLICATE: {
            if (u < 0) {
                v = 0;
            } else if (u >= dimSize) {
                v = dimSize - 1;
            } else {
                v = u;
            }
            break;
        }

        case eBorderType::BORDER_TYPE_WRAP: {
            v = ((u % dimSize) + dimSize) % dimSize;
            break;
        }

        default: {
            v = u;
        }
    }
    return v;
}
/**
 * @brief Golden implementation for handling out of bounds behavior during image accesses.
 *
 * @tparam T The underlying datatype of the image. (e.g. uchar3)
 * @tparam BT The base datatype of the image (e.g. unsigned char)
 * @param[in] input The input ImageWrapper referencing the underlying image data.
 * @param[in] borderMode The border mode used to handle out of bounds coordinates.
 * @param[in] borderValue The value to fallback to when handling out of bounds coordinates with the CONSTANT border
 * mode.
 * @param[in] sample The sample index of the image within the batch.
 * @param[in] y The y coordinates of the image.
 * @param[in] x The x coordinates of the image.
 * @param[in] channel The channel of the image.
 * @return The value at the requested coordinate of the image, or a value determined from the border mode if the
 * coordinates fall out of bounds.
 */
template <typename T, typename BT = detail::BaseType<T>>
BT GoldenBorderAt(ImageWrapper<T>& input, const eBorderType borderMode, T borderValue, int64_t sample, int64_t y,
                  int64_t x, int64_t channel) {
    int64_t outX = x, outY = y;

    if (borderMode == eBorderType::BORDER_TYPE_CONSTANT) {
        // Handle constant border type boundaries. This will return a constant value if either x or y coordinates
        // fall out of bounds.
        if (IsOutOfBounds(x, input.width()) || IsOutOfBounds(y, input.height())) {
            return detail::GetElement(borderValue, channel);
        }
    } else {
        outX = GetCoordOfBorderPel(x, input.width(), borderMode);
        outY = GetCoordOfBorderPel(y, input.height(), borderMode);
    }

    // Return the value at the modified outX, outY coordinates using the passed in ImageWrapper.
    return detail::GetElement(input.at(sample, outY, outX, 0), channel);
}

/**
 * @brief Runs a correctness test for BorderWrapper.
 *
 * @tparam T The underlying type of the data (e.g. uchar3)
 * @tparam BorderType The border type to use for boundary conditions.
 * @tparam BT The base type of the data (e.g. unsigned char)
 * @param[in] borderValue Value to use for constant border mode.
 * @param[in] batchSize Number of images in the batch.
 * @param[in] imageSize Width and height of the images in the batch.
 * @param[in] borderRadius The size of the borders around the image to test.
 * @throws std::runtime_error if the test does not pass.
 */
template <typename T, eBorderType BorderType, typename BT = detail::BaseType<T>>
void TestCorrectness(float4 borderValue, int32_t batchSize, Size2D imageSize, int32_t borderRadius) {
    int32_t channels = detail::NumElements<T>;
    int64_t numElements = batchSize * imageSize.w * imageSize.h * channels;
    int64_t numElementsWithBorder =
        batchSize * (imageSize.w + borderRadius * 2) * (imageSize.h + borderRadius * 2) * channels;

    // Convert borderValue to the same type as the image pixels
    T borderVal = detail::SaturateCast<T>(borderValue);

    // Generate synthetic data
    std::vector<BT> inputData(numElements);
    FillVector(inputData);

    // BorderWrapper to calculate the actual calculated values.
    BorderWrapper<T, BorderType> borderWrap(ImageWrapper<T>(inputData, batchSize, imageSize.w, imageSize.h), borderVal);
    std::vector<BT> actualOutput(numElementsWithBorder);
    int actualIndex = 0;
    for (int batch = 0; batch < batchSize; ++batch) {
        for (int y = -borderRadius; y < imageSize.h + borderRadius; ++y) {
            for (int x = -borderRadius; x < imageSize.w + borderRadius; ++x) {
                for (int c = 0; c < channels; ++c) {
                    T val = borderWrap.at(batch, y, x, 0);
                    actualOutput[actualIndex] = detail::GetElement(val, c);
                    actualIndex++;
                }
            }
        }
    }

    // ImageWrapper for use in the golden output generator. ImageWrapper is unit tested separately, and is
    // considered working at this point in the dependency chain.
    ImageWrapper<T> imageWrap(inputData, batchSize, imageSize.w, imageSize.h);
    std::vector<BT> goldenOutput(numElementsWithBorder);
    int goldenIndex = 0;
    for (int batch = 0; batch < batchSize; ++batch) {
        for (int y = -borderRadius; y < imageSize.h + borderRadius; ++y) {
            for (int x = -borderRadius; x < imageSize.w + borderRadius; ++x) {
                for (int c = 0; c < channels; ++c) {
                    goldenOutput[goldenIndex] = GoldenBorderAt(imageWrap, BorderType, borderVal, batch, y, x, c);
                    goldenIndex++;
                }
            }
        }
    }

    // Compare actual output to golden output to verify they are identical
    CompareVectors(actualOutput, goldenOutput);
}
}  // namespace

eTestStatusType test_border_wrapper(int argc, char** argv) {
    TEST_CASES_BEGIN();

    // clang-format off

    // U8 datatype tests
    TEST_CASE((TestCorrectness<uchar1, eBorderType::BORDER_TYPE_CONSTANT>(make_float4(1.0f, 1.0f, 1.0f, 1.0f), 1, {56, 13}, 9)));
    TEST_CASE((TestCorrectness<uchar3, eBorderType::BORDER_TYPE_CONSTANT>(make_float4(0.5f, 1.0f, 0.0f, 1.0f), 2, {1, 1}, 9)));
    TEST_CASE((TestCorrectness<uchar4, eBorderType::BORDER_TYPE_CONSTANT>(make_float4(1.0f, 1.0f, 1.0f, 0.5f), 3, {16, 83}, 9)));

    TEST_CASE((TestCorrectness<uchar1, eBorderType::BORDER_TYPE_REPLICATE>(make_float4(1.0f, 1.0f, 1.0f, 1.0f), 1, {56, 13}, 9)));
    TEST_CASE((TestCorrectness<uchar3, eBorderType::BORDER_TYPE_REPLICATE>(make_float4(0.5f, 1.0f, 0.0f, 1.0f), 2, {1, 1}, 9)));
    TEST_CASE((TestCorrectness<uchar4, eBorderType::BORDER_TYPE_REPLICATE>(make_float4(1.0f, 1.0f, 1.0f, 0.5f), 3, {16, 83}, 9)));

    TEST_CASE((TestCorrectness<uchar1, eBorderType::BORDER_TYPE_WRAP>(make_float4(1.0f, 1.0f, 1.0f, 1.0f), 1, {56, 13}, 9)));
    TEST_CASE((TestCorrectness<uchar3, eBorderType::BORDER_TYPE_WRAP>(make_float4(0.5f, 1.0f, 0.0f, 1.0f), 2, {1, 1}, 9)));
    TEST_CASE((TestCorrectness<uchar4, eBorderType::BORDER_TYPE_WRAP>(make_float4(1.0f, 1.0f, 1.0f, 0.5f), 3, {16, 83}, 9)));

    TEST_CASE((TestCorrectness<uchar1, eBorderType::BORDER_TYPE_REFLECT>(make_float4(1.0f, 1.0f, 1.0f, 1.0f), 1, {56, 13}, 9)));
    TEST_CASE((TestCorrectness<uchar3, eBorderType::BORDER_TYPE_REFLECT>(make_float4(0.5f, 1.0f, 0.0f, 1.0f), 2, {1, 1}, 9)));
    TEST_CASE((TestCorrectness<uchar4, eBorderType::BORDER_TYPE_REFLECT>(make_float4(1.0f, 1.0f, 1.0f, 0.5f), 3, {16, 83}, 9)));

    TEST_CASE((TestCorrectness<uchar1, eBorderType::BORDER_TYPE_REFLECT101>(make_float4(1.0f, 1.0f, 1.0f, 1.0f), 1, {56, 13}, 9)));
    TEST_CASE((TestCorrectness<uchar3, eBorderType::BORDER_TYPE_REFLECT101>(make_float4(0.5f, 1.0f, 0.0f, 1.0f), 2, {1, 1}, 9)));
    TEST_CASE((TestCorrectness<uchar4, eBorderType::BORDER_TYPE_REFLECT101>(make_float4(1.0f, 1.0f, 1.0f, 0.5f), 3, {16, 83}, 9)));

    // S8 datatype tests
    TEST_CASE((TestCorrectness<char1, eBorderType::BORDER_TYPE_CONSTANT>(make_float4(1.0f, 1.0f, 1.0f, 1.0f), 1, {56, 13}, 9)));
    TEST_CASE((TestCorrectness<char3, eBorderType::BORDER_TYPE_CONSTANT>(make_float4(0.5f, 1.0f, 0.0f, 1.0f), 2, {1, 1}, 9)));
    TEST_CASE((TestCorrectness<char4, eBorderType::BORDER_TYPE_CONSTANT>(make_float4(1.0f, 1.0f, 1.0f, 0.5f), 3, {16, 83}, 9)));

    TEST_CASE((TestCorrectness<char1, eBorderType::BORDER_TYPE_REPLICATE>(make_float4(1.0f, 1.0f, 1.0f, 1.0f), 1, {56, 13}, 9)));
    TEST_CASE((TestCorrectness<char3, eBorderType::BORDER_TYPE_REPLICATE>(make_float4(0.5f, 1.0f, 0.0f, 1.0f), 2, {1, 1}, 9)));
    TEST_CASE((TestCorrectness<char4, eBorderType::BORDER_TYPE_REPLICATE>(make_float4(1.0f, 1.0f, 1.0f, 0.5f), 3, {16, 83}, 9)));

    TEST_CASE((TestCorrectness<char1, eBorderType::BORDER_TYPE_WRAP>(make_float4(1.0f, 1.0f, 1.0f, 1.0f), 1, {56, 13}, 9)));
    TEST_CASE((TestCorrectness<char3, eBorderType::BORDER_TYPE_WRAP>(make_float4(0.5f, 1.0f, 0.0f, 1.0f), 2, {1, 1}, 9)));
    TEST_CASE((TestCorrectness<char4, eBorderType::BORDER_TYPE_WRAP>(make_float4(1.0f, 1.0f, 1.0f, 0.5f), 3, {16, 83}, 9)));

    TEST_CASE((TestCorrectness<char1, eBorderType::BORDER_TYPE_REFLECT>(make_float4(1.0f, 1.0f, 1.0f, 1.0f), 1, {56, 13}, 9)));
    TEST_CASE((TestCorrectness<char3, eBorderType::BORDER_TYPE_REFLECT>(make_float4(0.5f, 1.0f, 0.0f, 1.0f), 2, {1, 1}, 9)));
    TEST_CASE((TestCorrectness<char4, eBorderType::BORDER_TYPE_REFLECT>(make_float4(1.0f, 1.0f, 1.0f, 0.5f), 3, {16, 83}, 9)));

    TEST_CASE((TestCorrectness<char1, eBorderType::BORDER_TYPE_REFLECT101>(make_float4(1.0f, 1.0f, 1.0f, 1.0f), 1, {56, 13}, 9)));
    TEST_CASE((TestCorrectness<char3, eBorderType::BORDER_TYPE_REFLECT101>(make_float4(0.5f, 1.0f, 0.0f, 1.0f), 2, {1, 1}, 9)));
    TEST_CASE((TestCorrectness<char4, eBorderType::BORDER_TYPE_REFLECT101>(make_float4(1.0f, 1.0f, 1.0f, 0.5f), 3, {16, 83}, 9)));

    // U16 datatype tests
    TEST_CASE((TestCorrectness<ushort1, eBorderType::BORDER_TYPE_CONSTANT>(make_float4(1.0f, 1.0f, 1.0f, 1.0f), 1, {56, 13}, 9)));
    TEST_CASE((TestCorrectness<ushort3, eBorderType::BORDER_TYPE_CONSTANT>(make_float4(0.5f, 1.0f, 0.0f, 1.0f), 2, {1, 1}, 9)));
    TEST_CASE((TestCorrectness<ushort4, eBorderType::BORDER_TYPE_CONSTANT>(make_float4(1.0f, 1.0f, 1.0f, 0.5f), 3, {16, 83}, 9)));

    TEST_CASE((TestCorrectness<ushort1, eBorderType::BORDER_TYPE_REPLICATE>(make_float4(1.0f, 1.0f, 1.0f, 1.0f), 1, {56, 13}, 9)));
    TEST_CASE((TestCorrectness<ushort3, eBorderType::BORDER_TYPE_REPLICATE>(make_float4(0.5f, 1.0f, 0.0f, 1.0f), 2, {1, 1}, 9)));
    TEST_CASE((TestCorrectness<ushort4, eBorderType::BORDER_TYPE_REPLICATE>(make_float4(1.0f, 1.0f, 1.0f, 0.5f), 3, {16, 83}, 9)));

    TEST_CASE((TestCorrectness<ushort1, eBorderType::BORDER_TYPE_WRAP>(make_float4(1.0f, 1.0f, 1.0f, 1.0f), 1, {56, 13}, 9)));
    TEST_CASE((TestCorrectness<ushort3, eBorderType::BORDER_TYPE_WRAP>(make_float4(0.5f, 1.0f, 0.0f, 1.0f), 2, {1, 1}, 9)));
    TEST_CASE((TestCorrectness<ushort4, eBorderType::BORDER_TYPE_WRAP>(make_float4(1.0f, 1.0f, 1.0f, 0.5f), 3, {16, 83}, 9)));

    TEST_CASE((TestCorrectness<ushort1, eBorderType::BORDER_TYPE_REFLECT>(make_float4(1.0f, 1.0f, 1.0f, 1.0f), 1, {56, 13}, 9)));
    TEST_CASE((TestCorrectness<ushort3, eBorderType::BORDER_TYPE_REFLECT>(make_float4(0.5f, 1.0f, 0.0f, 1.0f), 2, {1, 1}, 9)));
    TEST_CASE((TestCorrectness<ushort4, eBorderType::BORDER_TYPE_REFLECT>(make_float4(1.0f, 1.0f, 1.0f, 0.5f), 3, {16, 83}, 9)));

    TEST_CASE((TestCorrectness<ushort1, eBorderType::BORDER_TYPE_REFLECT101>(make_float4(1.0f, 1.0f, 1.0f, 1.0f), 1, {56, 13}, 9)));
    TEST_CASE((TestCorrectness<ushort3, eBorderType::BORDER_TYPE_REFLECT101>(make_float4(0.5f, 1.0f, 0.0f, 1.0f), 2, {1, 1}, 9)));
    TEST_CASE((TestCorrectness<ushort4, eBorderType::BORDER_TYPE_REFLECT101>(make_float4(1.0f, 1.0f, 1.0f, 0.5f), 3, {16, 83}, 9)));

    // S16 datatype tests
    TEST_CASE((TestCorrectness<short1, eBorderType::BORDER_TYPE_CONSTANT>(make_float4(1.0f, 1.0f, 1.0f, 1.0f), 1, {56, 13}, 9)));
    TEST_CASE((TestCorrectness<short3, eBorderType::BORDER_TYPE_CONSTANT>(make_float4(0.5f, 1.0f, 0.0f, 1.0f), 2, {1, 1}, 9)));
    TEST_CASE((TestCorrectness<short4, eBorderType::BORDER_TYPE_CONSTANT>(make_float4(1.0f, 1.0f, 1.0f, 0.5f), 3, {16, 83}, 9)));

    TEST_CASE((TestCorrectness<short1, eBorderType::BORDER_TYPE_REPLICATE>(make_float4(1.0f, 1.0f, 1.0f, 1.0f), 1, {56, 13}, 9)));
    TEST_CASE((TestCorrectness<short3, eBorderType::BORDER_TYPE_REPLICATE>(make_float4(0.5f, 1.0f, 0.0f, 1.0f), 2, {1, 1}, 9)));
    TEST_CASE((TestCorrectness<short4, eBorderType::BORDER_TYPE_REPLICATE>(make_float4(1.0f, 1.0f, 1.0f, 0.5f), 3, {16, 83}, 9)));

    TEST_CASE((TestCorrectness<short1, eBorderType::BORDER_TYPE_WRAP>(make_float4(1.0f, 1.0f, 1.0f, 1.0f), 1, {56, 13}, 9)));
    TEST_CASE((TestCorrectness<short3, eBorderType::BORDER_TYPE_WRAP>(make_float4(0.5f, 1.0f, 0.0f, 1.0f), 2, {1, 1}, 9)));
    TEST_CASE((TestCorrectness<short4, eBorderType::BORDER_TYPE_WRAP>(make_float4(1.0f, 1.0f, 1.0f, 0.5f), 3, {16, 83}, 9)));

    TEST_CASE((TestCorrectness<short1, eBorderType::BORDER_TYPE_REFLECT>(make_float4(1.0f, 1.0f, 1.0f, 1.0f), 1, {56, 13}, 9)));
    TEST_CASE((TestCorrectness<short3, eBorderType::BORDER_TYPE_REFLECT>(make_float4(0.5f, 1.0f, 0.0f, 1.0f), 2, {1, 1}, 9)));
    TEST_CASE((TestCorrectness<short4, eBorderType::BORDER_TYPE_REFLECT>(make_float4(1.0f, 1.0f, 1.0f, 0.5f), 3, {16, 83}, 9)));

    TEST_CASE((TestCorrectness<short1, eBorderType::BORDER_TYPE_REFLECT101>(make_float4(1.0f, 1.0f, 1.0f, 1.0f), 1, {56, 13}, 9)));
    TEST_CASE((TestCorrectness<short3, eBorderType::BORDER_TYPE_REFLECT101>(make_float4(0.5f, 1.0f, 0.0f, 1.0f), 2, {1, 1}, 9)));
    TEST_CASE((TestCorrectness<short4, eBorderType::BORDER_TYPE_REFLECT101>(make_float4(1.0f, 1.0f, 1.0f, 0.5f), 3, {16, 83}, 9)));

    // S32 datatype tests
    TEST_CASE((TestCorrectness<int1, eBorderType::BORDER_TYPE_CONSTANT>(make_float4(1.0f, 1.0f, 1.0f, 1.0f), 1, {56, 13}, 9)));
    TEST_CASE((TestCorrectness<int3, eBorderType::BORDER_TYPE_CONSTANT>(make_float4(0.5f, 1.0f, 0.0f, 1.0f), 2, {1, 1}, 9)));
    TEST_CASE((TestCorrectness<int4, eBorderType::BORDER_TYPE_CONSTANT>(make_float4(1.0f, 1.0f, 1.0f, 0.5f), 3, {16, 83}, 9)));

    TEST_CASE((TestCorrectness<int1, eBorderType::BORDER_TYPE_REPLICATE>(make_float4(1.0f, 1.0f, 1.0f, 1.0f), 1, {56, 13}, 9)));
    TEST_CASE((TestCorrectness<int3, eBorderType::BORDER_TYPE_REPLICATE>(make_float4(0.5f, 1.0f, 0.0f, 1.0f), 2, {1, 1}, 9)));
    TEST_CASE((TestCorrectness<int4, eBorderType::BORDER_TYPE_REPLICATE>(make_float4(1.0f, 1.0f, 1.0f, 0.5f), 3, {16, 83}, 9)));

    TEST_CASE((TestCorrectness<int1, eBorderType::BORDER_TYPE_WRAP>(make_float4(1.0f, 1.0f, 1.0f, 1.0f), 1, {56, 13}, 9)));
    TEST_CASE((TestCorrectness<int3, eBorderType::BORDER_TYPE_WRAP>(make_float4(0.5f, 1.0f, 0.0f, 1.0f), 2, {1, 1}, 9)));
    TEST_CASE((TestCorrectness<int4, eBorderType::BORDER_TYPE_WRAP>(make_float4(1.0f, 1.0f, 1.0f, 0.5f), 3, {16, 83}, 9)));

    TEST_CASE((TestCorrectness<int1, eBorderType::BORDER_TYPE_REFLECT>(make_float4(1.0f, 1.0f, 1.0f, 1.0f), 1, {56, 13}, 9)));
    TEST_CASE((TestCorrectness<int3, eBorderType::BORDER_TYPE_REFLECT>(make_float4(0.5f, 1.0f, 0.0f, 1.0f), 2, {1, 1}, 9)));
    TEST_CASE((TestCorrectness<int4, eBorderType::BORDER_TYPE_REFLECT>(make_float4(1.0f, 1.0f, 1.0f, 0.5f), 3, {16, 83}, 9)));

    TEST_CASE((TestCorrectness<int1, eBorderType::BORDER_TYPE_REFLECT101>(make_float4(1.0f, 1.0f, 1.0f, 1.0f), 1, {56, 13}, 9)));
    TEST_CASE((TestCorrectness<int3, eBorderType::BORDER_TYPE_REFLECT101>(make_float4(0.5f, 1.0f, 0.0f, 1.0f), 2, {1, 1}, 9)));
    TEST_CASE((TestCorrectness<int4, eBorderType::BORDER_TYPE_REFLECT101>(make_float4(1.0f, 1.0f, 1.0f, 0.5f), 3, {16, 83}, 9)));

    // U32 datatype tests
    TEST_CASE((TestCorrectness<uint1, eBorderType::BORDER_TYPE_CONSTANT>(make_float4(1.0f, 1.0f, 1.0f, 1.0f), 1, {56, 13}, 9)));
    TEST_CASE((TestCorrectness<uint3, eBorderType::BORDER_TYPE_CONSTANT>(make_float4(0.5f, 1.0f, 0.0f, 1.0f), 2, {1, 1}, 9)));
    TEST_CASE((TestCorrectness<uint4, eBorderType::BORDER_TYPE_CONSTANT>(make_float4(1.0f, 1.0f, 1.0f, 0.5f), 3, {16, 83}, 9)));

    TEST_CASE((TestCorrectness<uint1, eBorderType::BORDER_TYPE_REPLICATE>(make_float4(1.0f, 1.0f, 1.0f, 1.0f), 1, {56, 13}, 9)));
    TEST_CASE((TestCorrectness<uint3, eBorderType::BORDER_TYPE_REPLICATE>(make_float4(0.5f, 1.0f, 0.0f, 1.0f), 2, {1, 1}, 9)));
    TEST_CASE((TestCorrectness<uint4, eBorderType::BORDER_TYPE_REPLICATE>(make_float4(1.0f, 1.0f, 1.0f, 0.5f), 3, {16, 83}, 9)));

    TEST_CASE((TestCorrectness<uint1, eBorderType::BORDER_TYPE_WRAP>(make_float4(1.0f, 1.0f, 1.0f, 1.0f), 1, {56, 13}, 9)));
    TEST_CASE((TestCorrectness<uint3, eBorderType::BORDER_TYPE_WRAP>(make_float4(0.5f, 1.0f, 0.0f, 1.0f), 2, {1, 1}, 9)));
    TEST_CASE((TestCorrectness<uint4, eBorderType::BORDER_TYPE_WRAP>(make_float4(1.0f, 1.0f, 1.0f, 0.5f), 3, {16, 83}, 9)));

    TEST_CASE((TestCorrectness<uint1, eBorderType::BORDER_TYPE_REFLECT>(make_float4(1.0f, 1.0f, 1.0f, 1.0f), 1, {56, 13}, 9)));
    TEST_CASE((TestCorrectness<uint3, eBorderType::BORDER_TYPE_REFLECT>(make_float4(0.5f, 1.0f, 0.0f, 1.0f), 2, {1, 1}, 9)));
    TEST_CASE((TestCorrectness<uint4, eBorderType::BORDER_TYPE_REFLECT>(make_float4(1.0f, 1.0f, 1.0f, 0.5f), 3, {16, 83}, 9)));

    TEST_CASE((TestCorrectness<uint1, eBorderType::BORDER_TYPE_REFLECT101>(make_float4(1.0f, 1.0f, 1.0f, 1.0f), 1, {56, 13}, 9)));
    TEST_CASE((TestCorrectness<uint3, eBorderType::BORDER_TYPE_REFLECT101>(make_float4(0.5f, 1.0f, 0.0f, 1.0f), 2, {1, 1}, 9)));
    TEST_CASE((TestCorrectness<uint4, eBorderType::BORDER_TYPE_REFLECT101>(make_float4(1.0f, 1.0f, 1.0f, 0.5f), 3, {16, 83}, 9)));

    // F32 datatype tests
    TEST_CASE((TestCorrectness<float1, eBorderType::BORDER_TYPE_CONSTANT>(make_float4(1.0f, 1.0f, 1.0f, 1.0f), 1, {56, 13}, 9)));
    TEST_CASE((TestCorrectness<float3, eBorderType::BORDER_TYPE_CONSTANT>(make_float4(0.5f, 1.0f, 0.0f, 1.0f), 2, {1, 1}, 9)));
    TEST_CASE((TestCorrectness<float4, eBorderType::BORDER_TYPE_CONSTANT>(make_float4(1.0f, 1.0f, 1.0f, 0.5f), 3, {16, 83}, 9)));

    TEST_CASE((TestCorrectness<float1, eBorderType::BORDER_TYPE_REPLICATE>(make_float4(1.0f, 1.0f, 1.0f, 1.0f), 1, {56, 13}, 9)));
    TEST_CASE((TestCorrectness<float3, eBorderType::BORDER_TYPE_REPLICATE>(make_float4(0.5f, 1.0f, 0.0f, 1.0f), 2, {1, 1}, 9)));
    TEST_CASE((TestCorrectness<float4, eBorderType::BORDER_TYPE_REPLICATE>(make_float4(1.0f, 1.0f, 1.0f, 0.5f), 3, {16, 83}, 9)));

    TEST_CASE((TestCorrectness<float1, eBorderType::BORDER_TYPE_WRAP>(make_float4(1.0f, 1.0f, 1.0f, 1.0f), 1, {56, 13}, 9)));
    TEST_CASE((TestCorrectness<float3, eBorderType::BORDER_TYPE_WRAP>(make_float4(0.5f, 1.0f, 0.0f, 1.0f), 2, {1, 1}, 9)));
    TEST_CASE((TestCorrectness<float4, eBorderType::BORDER_TYPE_WRAP>(make_float4(1.0f, 1.0f, 1.0f, 0.5f), 3, {16, 83}, 9)));

    TEST_CASE((TestCorrectness<float1, eBorderType::BORDER_TYPE_REFLECT>(make_float4(1.0f, 1.0f, 1.0f, 1.0f), 1, {56, 13}, 9)));
    TEST_CASE((TestCorrectness<float3, eBorderType::BORDER_TYPE_REFLECT>(make_float4(0.5f, 1.0f, 0.0f, 1.0f), 2, {1, 1}, 9)));
    TEST_CASE((TestCorrectness<float4, eBorderType::BORDER_TYPE_REFLECT>(make_float4(1.0f, 1.0f, 1.0f, 0.5f), 3, {16, 83}, 9)));

    TEST_CASE((TestCorrectness<float1, eBorderType::BORDER_TYPE_REFLECT101>(make_float4(1.0f, 1.0f, 1.0f, 1.0f), 1, {56, 13}, 9)));
    TEST_CASE((TestCorrectness<float3, eBorderType::BORDER_TYPE_REFLECT101>(make_float4(0.5f, 1.0f, 0.0f, 1.0f), 2, {1, 1}, 9)));
    TEST_CASE((TestCorrectness<float4, eBorderType::BORDER_TYPE_REFLECT101>(make_float4(1.0f, 1.0f, 1.0f, 0.5f), 3, {16, 83}, 9)));

    // F64 datatype tests
    TEST_CASE((TestCorrectness<double1, eBorderType::BORDER_TYPE_CONSTANT>(make_float4(1.0f, 1.0f, 1.0f, 1.0f), 1, {56, 13}, 9)));
    TEST_CASE((TestCorrectness<double3, eBorderType::BORDER_TYPE_CONSTANT>(make_float4(0.5f, 1.0f, 0.0f, 1.0f), 2, {1, 1}, 9)));
    TEST_CASE((TestCorrectness<double4, eBorderType::BORDER_TYPE_CONSTANT>(make_float4(1.0f, 1.0f, 1.0f, 0.5f), 3, {16, 83}, 9)));

    TEST_CASE((TestCorrectness<double1, eBorderType::BORDER_TYPE_REPLICATE>(make_float4(1.0f, 1.0f, 1.0f, 1.0f), 1, {56, 13}, 9)));
    TEST_CASE((TestCorrectness<double3, eBorderType::BORDER_TYPE_REPLICATE>(make_float4(0.5f, 1.0f, 0.0f, 1.0f), 2, {1, 1}, 9)));
    TEST_CASE((TestCorrectness<double4, eBorderType::BORDER_TYPE_REPLICATE>(make_float4(1.0f, 1.0f, 1.0f, 0.5f), 3, {16, 83}, 9)));

    TEST_CASE((TestCorrectness<double1, eBorderType::BORDER_TYPE_WRAP>(make_float4(1.0f, 1.0f, 1.0f, 1.0f), 1, {56, 13}, 9)));
    TEST_CASE((TestCorrectness<double3, eBorderType::BORDER_TYPE_WRAP>(make_float4(0.5f, 1.0f, 0.0f, 1.0f), 2, {1, 1}, 9)));
    TEST_CASE((TestCorrectness<double4, eBorderType::BORDER_TYPE_WRAP>(make_float4(1.0f, 1.0f, 1.0f, 0.5f), 3, {16, 83}, 9)));

    TEST_CASE((TestCorrectness<double1, eBorderType::BORDER_TYPE_REFLECT>(make_float4(1.0f, 1.0f, 1.0f, 1.0f), 1, {56, 13}, 9)));
    TEST_CASE((TestCorrectness<double3, eBorderType::BORDER_TYPE_REFLECT>(make_float4(0.5f, 1.0f, 0.0f, 1.0f), 2, {1, 1}, 9)));
    TEST_CASE((TestCorrectness<double4, eBorderType::BORDER_TYPE_REFLECT>(make_float4(1.0f, 1.0f, 1.0f, 0.5f), 3, {16, 83}, 9)));

    TEST_CASE((TestCorrectness<double1, eBorderType::BORDER_TYPE_REFLECT101>(make_float4(1.0f, 1.0f, 1.0f, 1.0f), 1, {56, 13}, 9)));
    TEST_CASE((TestCorrectness<double3, eBorderType::BORDER_TYPE_REFLECT101>(make_float4(0.5f, 1.0f, 0.0f, 1.0f), 2, {1, 1}, 9)));
    TEST_CASE((TestCorrectness<double4, eBorderType::BORDER_TYPE_REFLECT101>(make_float4(1.0f, 1.0f, 1.0f, 0.5f), 3, {16, 83}, 9)));
    // clang-format on

    TEST_CASES_END();
}