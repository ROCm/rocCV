/**
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <algorithm>
#include "core/detail/casting.hpp"
#include "core/detail/type_traits.hpp"
#include "core/detail/math/vectorized_type_math.hpp"
#include <core/wrappers/image_wrapper.hpp>
#include <core/wrappers/interpolation_wrapper.hpp>
#include <iostream>
#include <op_remap.hpp>
#include "operator_types.h"

#include "test_helpers.hpp"

using namespace roccv;
using namespace roccv::tests;
using namespace roccv::detail;

// Keep all non-entrypoint functions in an anonymous namespace to prevent redefinition errors across translation units.
namespace {
/**
 * @brief Verified golden C++ model for the Remap operation.
 *
 * @tparam T Vectorized datatype of the image's pixels.
 * @tparam BorderType Border mode to use.
 * @tparam InterpType Interpolation mode to use.
 * @tparam MapInterpType Interpolation mode to use for the map
 * @tparam BT Base type of the image's data.
 * @param[in] input An input vector containing image data.
 * @param[in] batchSize The number of images in the batch.
 * @param[in] width Image width.
 * @param[in] height Image height.
 * @param[in] map Tensor containing the remap coordinates.
 * @param[in] borderValue Border value to use as a fallback when going out of bounds.
 * @return Vector containing the results of the operation.
 */
template <typename T, eBorderType BorderType, eInterpolationType InterpType, eInterpolationType MapInterpType, typename BT = detail::BaseType<T>>
std::vector<BT> GoldenRemapAbsolute(std::vector<BT>& input, int32_t batchSize, int32_t width, int32_t height, Tensor& map, float4 borderValue) {

    // Create an output vector the same size as the input vector
    std::vector<BT> output(input.size());

    // Create interpolation wrapper for input vector
    InterpolationWrapper<T, BorderType, InterpType> src((BorderWrapper<T, BorderType>(
        ImageWrapper<T>(input, batchSize, width, height), detail::RangeCast<T>(borderValue))));

    // Wrap the output vector for simplified data access
    ImageWrapper<T> dst(output, batchSize, width, height);
    
    // Create an interpolation wrapper for the map tensor
    InterpolationWrapper<float2, BorderType, MapInterpType> wrappedMapTensor(map, make_float2(0, 0));

    for (int b = 0; b < batchSize; b++) {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                float2 mapCoordinates = wrappedMapTensor.at(b, y, x, 0);
                dst.at(b, y, x, 0) = src.at(b, mapCoordinates.y, mapCoordinates.x, 0);
            }
        }
    }
    return output;
}

/**
 * @brief Tests correctness for the Remap operator by comparing roccv::Remap results with the
 * defined golden model.
 *
 * @tparam T Image datatype.
 * @tparam BorderType Border mode to use.
 * @tparam InterpType Interpolation mode to use.
 * @tparam MapInterpType Interpolation mode to use for the map
 * @tparam BT Base type of the image's data.
 * @param batchSize Number of images within the batch.
 * @param width Width of the input image.
 * @param height Height of the input image.
 * @param format Format of the images (must match with T).
 * @param borderValue Border value to use as a fallback when going out of bounds.
 * @param mapType Type of remap to do, REMAP_ABSOLUTE, REMAP_ABSOLUTE_NORMALIZED, REMAP_RELATIVE_NORMALIZED
 * @param device The device to run the roccv::WarpPerspective operator on.
 */
template <typename T, eBorderType BorderType, eInterpolationType InterpType, eInterpolationType MapInterpType, typename BT = detail::BaseType<T>>
void TestCorrectness(int batchSize, int width, int height, ImageFormat format, float4 borderValue, eRemapType mapType, eDeviceType device) {
    
    // Create input and output tensor based on test parameters
    Tensor input(batchSize, {width, height}, format, device);
    Tensor output(batchSize, {width, height}, format, device);
    
    // Create a vector and fill it with random data.
    std::vector<BT> inputData(input.shape().size());
    FillVector(inputData);

    // Copy generated input data into input tensor
    CopyVectorIntoTensor(input, inputData);

    std::vector<float> colRemapTable;
    std::vector<float> rowRemapTable;

    float halfWidth = width / 2;
    for (int i = 0; i < height; i++) {
        int j = 0;
        for (; j < halfWidth; j++) {
            rowRemapTable.push_back(i);
            colRemapTable.push_back(halfWidth - j);
        }
        for (; j < width; j++) {
            rowRemapTable.push_back(i);
            colRemapTable.push_back(j);
        }
    }

    std::vector<float2> mapData(rowRemapTable.size());
    for (int i = 0; i < rowRemapTable.size(); i++) {
        mapData[i] = make_float2(colRemapTable[i], rowRemapTable[i]);
    }

    // Create map tensor and fill it with mapData
    TensorShape map_shape(TensorLayout(eTensorLayout::TENSOR_LAYOUT_HWC), {height, width, 2});
    DataType map_dtype(eDataType::DATA_TYPE_F32);
    Tensor mapTensor(map_shape, map_dtype, device);

    CopyVectorIntoTensor(mapTensor, mapData);

    hipStream_t stream;
    HIP_VALIDATE_NO_ERRORS(hipStreamCreate(&stream));
    Remap op;
    op(stream, input, output, mapTensor, InterpType, MapInterpType, mapType, false, BorderType, borderValue, device);
    HIP_VALIDATE_NO_ERRORS(hipStreamSynchronize(stream));
    HIP_VALIDATE_NO_ERRORS(hipStreamDestroy(stream));

    // Copy data from output tensor into a host allocated vector
    std::vector<BT> result(output.shape().size());
    CopyTensorIntoVector(result, output);

    std::vector<BT> ref = GoldenRemapAbsolute<T, BorderType, InterpType, MapInterpType>(inputData, batchSize, width, height, mapTensor, borderValue);

    // Compare data in actual output versus the generated golden reference image
    CompareVectors(result, ref);
}
}  // namespace

eTestStatusType test_op_remap(int argc, char** argv) {
    TEST_CASES_BEGIN();

    TEST_CASE((TestCorrectness<uchar1, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_NEAREST, eInterpolationType::INTERP_TYPE_NEAREST>(1, 480, 360, FMT_U8, make_float4(0.0f, 0.0f, 0.0f, 1.0f), REMAP_ABSOLUTE, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<uchar3, eBorderType::BORDER_TYPE_REPLICATE, eInterpolationType::INTERP_TYPE_NEAREST, eInterpolationType::INTERP_TYPE_LINEAR>(1, 480, 360, FMT_RGB8, make_float4(0.0f, 0.0f, 0.0f, 1.0f), REMAP_ABSOLUTE, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<uchar4, eBorderType::BORDER_TYPE_REFLECT, eInterpolationType::INTERP_TYPE_NEAREST, eInterpolationType::INTERP_TYPE_NEAREST>(1, 480, 360, FMT_RGBA8, make_float4(0.0f, 0.0f, 0.0f, 1.0f), REMAP_ABSOLUTE, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<uchar1, eBorderType::BORDER_TYPE_WRAP, eInterpolationType::INTERP_TYPE_NEAREST, eInterpolationType::INTERP_TYPE_LINEAR>(3, 480, 360, FMT_U8, make_float4(0.0f, 0.0f, 0.0f, 1.0f), REMAP_ABSOLUTE, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<uchar3, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_LINEAR, eInterpolationType::INTERP_TYPE_NEAREST>(3, 480, 360, FMT_RGB8, make_float4(0.0f, 0.0f, 0.0f, 1.0f), REMAP_ABSOLUTE, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<uchar4, eBorderType::BORDER_TYPE_REPLICATE, eInterpolationType::INTERP_TYPE_NEAREST, eInterpolationType::INTERP_TYPE_LINEAR>(3, 480, 360, FMT_RGBA8, make_float4(0.0f, 0.0f, 0.0f, 1.0f), REMAP_ABSOLUTE, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<uchar1, eBorderType::BORDER_TYPE_REFLECT, eInterpolationType::INTERP_TYPE_LINEAR, eInterpolationType::INTERP_TYPE_NEAREST>(5, 480, 360, FMT_U8, make_float4(0.0f, 0.0f, 0.0f, 1.0f), REMAP_ABSOLUTE, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<uchar3, eBorderType::BORDER_TYPE_WRAP, eInterpolationType::INTERP_TYPE_NEAREST, eInterpolationType::INTERP_TYPE_LINEAR>(5, 480, 360, FMT_RGB8, make_float4(0.0f, 0.0f, 0.0f, 1.0f), REMAP_ABSOLUTE, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<uchar4, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_LINEAR, eInterpolationType::INTERP_TYPE_NEAREST>(5, 480, 360, FMT_RGBA8, make_float4(0.0f, 0.0f, 0.0f, 1.0f), REMAP_ABSOLUTE, eDeviceType::GPU)));
    
    TEST_CASE((TestCorrectness<uchar1, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_NEAREST, eInterpolationType::INTERP_TYPE_NEAREST>(1, 480, 360, FMT_U8, make_float4(0.0f, 0.0f, 0.0f, 1.0f), REMAP_ABSOLUTE, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<uchar3, eBorderType::BORDER_TYPE_REPLICATE, eInterpolationType::INTERP_TYPE_NEAREST, eInterpolationType::INTERP_TYPE_LINEAR>(1, 480, 360, FMT_RGB8, make_float4(0.0f, 0.0f, 0.0f, 1.0f), REMAP_ABSOLUTE, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<uchar4, eBorderType::BORDER_TYPE_REFLECT, eInterpolationType::INTERP_TYPE_NEAREST, eInterpolationType::INTERP_TYPE_NEAREST>(1, 480, 360, FMT_RGBA8, make_float4(0.0f, 0.0f, 0.0f, 1.0f), REMAP_ABSOLUTE, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<uchar1, eBorderType::BORDER_TYPE_WRAP, eInterpolationType::INTERP_TYPE_NEAREST, eInterpolationType::INTERP_TYPE_LINEAR>(3, 480, 360, FMT_U8, make_float4(0.0f, 0.0f, 0.0f, 1.0f), REMAP_ABSOLUTE, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<uchar3, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_LINEAR, eInterpolationType::INTERP_TYPE_NEAREST>(3, 480, 360, FMT_RGB8, make_float4(0.0f, 0.0f, 0.0f, 1.0f), REMAP_ABSOLUTE, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<uchar4, eBorderType::BORDER_TYPE_REPLICATE, eInterpolationType::INTERP_TYPE_NEAREST, eInterpolationType::INTERP_TYPE_LINEAR>(3, 480, 360, FMT_RGBA8, make_float4(0.0f, 0.0f, 0.0f, 1.0f), REMAP_ABSOLUTE, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<uchar1, eBorderType::BORDER_TYPE_REFLECT, eInterpolationType::INTERP_TYPE_LINEAR, eInterpolationType::INTERP_TYPE_NEAREST>(5, 480, 360, FMT_U8, make_float4(0.0f, 0.0f, 0.0f, 1.0f), REMAP_ABSOLUTE, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<uchar3, eBorderType::BORDER_TYPE_WRAP, eInterpolationType::INTERP_TYPE_NEAREST, eInterpolationType::INTERP_TYPE_LINEAR>(5, 480, 360, FMT_RGB8, make_float4(0.0f, 0.0f, 0.0f, 1.0f), REMAP_ABSOLUTE, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<uchar4, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_LINEAR, eInterpolationType::INTERP_TYPE_NEAREST>(5, 480, 360, FMT_RGBA8, make_float4(0.0f, 0.0f, 0.0f, 1.0f), REMAP_ABSOLUTE, eDeviceType::CPU)));

    TEST_CASES_END();
}