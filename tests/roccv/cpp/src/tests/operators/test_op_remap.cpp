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
#include <core/wrappers/image_wrapper.hpp>
#include <core/wrappers/interpolation_wrapper.hpp>
#include <iostream>
#include <op_remap.hpp>
#include "core/detail/internal_structs.hpp"
#include "core/detail/casting.hpp"
#include "core/detail/math/vectorized_type_math.hpp"
#include "core/detail/type_traits.hpp"
#include "operator_types.h"
#include "test_helpers.hpp"

using namespace roccv;
using namespace roccv::tests;
using namespace roccv::detail;

// Keep all non-entrypoint functions in an anonymous namespace to prevent redefinition errors across translation units.
namespace {

RemapParams GetRemapParams(const int2 &srcSize, const int2 &dstSize, const int2 &mapSize, bool alignCorners, eRemapType mapValueType)
{
    RemapParams params;

    switch(mapValueType) {
        case REMAP_ABSOLUTE:
            params.srcScale = make_float2(0.f, 0.f);
            params.mapScale = StaticCast<float2>(mapSize) / StaticCast<float2>(dstSize);
            params.valScale = make_float2(1.f, 1.f);
            params.srcOffset = make_float2(0.f, 0.f);
            params.dstOffset = 0.f;
            break;
        case REMAP_ABSOLUTE_NORMALIZED:
            params.srcScale = make_float2(0.f, 0.f);
            params.mapScale = StaticCast<float2>(mapSize) / StaticCast<float2>(dstSize);
            params.valScale  = (StaticCast<float2>(srcSize) - (alignCorners ? 1.f : 0.f)) / 2.f;
            params.srcOffset = params.valScale - (alignCorners ? 0.f : .5f);
            params.dstOffset = 0.f;
            break;
        case REMAP_RELATIVE_NORMALIZED:
            params.srcScale = StaticCast<float2>(srcSize) / StaticCast<float2>(dstSize);
            params.mapScale = (StaticCast<float2>(mapSize) - 1.f) / StaticCast<float2>(dstSize);
            params.valScale = StaticCast<float2>(srcSize) - 1.f;
            params.dstOffset = alignCorners ? 0.f : .5f;
            params.srcOffset = params.srcScale * params.dstOffset - params.dstOffset;
            break;
    }
    return params;
}

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
template <typename T, eBorderType BorderType, eInterpolationType InterpType, eInterpolationType MapInterpType,
          typename BT = detail::BaseType<T>>
std::vector<BT> GoldenRemap(std::vector<BT>& input, int32_t batchSize, int32_t mapBatchSize, int32_t inWidth, int32_t inHeight, int32_t outWidth, 
                            int32_t outHeight, int32_t mapWidth, int32_t mapHeight, std::vector<float2>& mapData, eRemapType mapType, bool alignCorners, float4 borderValue) {
    
    int channels = detail::NumElements<T>;
    int outputSize = batchSize * outWidth * outHeight * channels;
    std::vector<BT> output(outputSize);

    // Create interpolation wrapper for input vector
    InterpolationWrapper<T, BorderType, InterpType> src((BorderWrapper<T, BorderType>(
        ImageWrapper<T>(input, batchSize, inWidth, inHeight), detail::SaturateCast<T>(borderValue))));

    // Wrap the output vector for simplified data access
    ImageWrapper<T> dst(output, batchSize, outWidth, outHeight);

    // Create an interpolation wrapper for the map tensor
    // InterpolationWrapper<float2, BorderType, MapInterpType> wrappedMapTensor(map, make_float2(0, 0));
    InterpolationWrapper<float2, BorderType, MapInterpType> map((BorderWrapper<float2, BorderType>(
        ImageWrapper<float2>(mapData.data(), mapBatchSize, mapWidth, mapHeight), detail::SaturateCast<float2>(borderValue))));

    int2 srcSize = make_int2(src.width(), src.height());
    int2 dstSize = make_int2(dst.width(), dst.height());
    int2 mapSize = make_int2(map.width(), map.height());

    float2 srcCoord = make_float2(0.f, 0.f);
    float2 mapCoord = make_float2(0.f, 0.f);
    float2 dstCoord = make_float2(0.f, 0.f);

    RemapParams params = GetRemapParams(srcSize, dstSize, mapSize, alignCorners, mapType);

    for (int b = 0; b < dst.batches(); b++) {
        for (int y = 0; y < dst.height(); y++) {
            for (int x = 0; x < dst.width(); x++) {
                
                dstCoord.x = static_cast<float>(x);
                dstCoord.y = static_cast<float>(y);
                
                mapCoord.x = (dstCoord.x + params.dstOffset) * params.mapScale.x;
                mapCoord.y = (dstCoord.y + params.dstOffset) * params.mapScale.y;
                
                float2 mapValue = map.at((mapBatchSize == 1 ? 0 : b), mapCoord.y, mapCoord.x, 0);

                srcCoord.x = dstCoord.x * params.srcScale.x + mapValue.x * params.valScale.x + params.srcOffset.x;
                srcCoord.y = dstCoord.y * params.srcScale.y + mapValue.y * params.valScale.y + params.srcOffset.y;

                dst.at(b, y, x, 0) = src.at(b, srcCoord.y, srcCoord.x, 0);
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
 * @param mapBatchSize Number of maps, either 1 or same as batchSize.
 * @param inWidth Width of the input image.
 * @param inHeight Height of the input image.
 * @param outWidth Width of the output image.
 * @param outHeight Height of the output image.
 * @param mapWidth Width of the map.
 * @param mapHeight Height of the map.
 * @param format Format of the images (must match with T).
 * @param borderValue Border value to use as a fallback when going out of bounds.
 * @param mapType Type of remap to do, REMAP_ABSOLUTE, REMAP_ABSOLUTE_NORMALIZED, REMAP_RELATIVE_NORMALIZED
 * @param device The device to run the roccv::Remap operator on.
 */
template <typename T, eBorderType BorderType, eInterpolationType InterpType, eInterpolationType MapInterpType,
          typename BT = detail::BaseType<T>>
void TestCorrectness(int batchSize, int mapBatchSize, int inWidth, int inHeight, int outWidth, int outHeight, int mapWidth, int mapHeight, ImageFormat format, float4 borderValue, eRemapType mapType,
                     bool alignCorners, eDeviceType device) {
    // Create input and output tensor based on test parameters
    Tensor input(batchSize, {inWidth, inHeight}, format, device);
    Tensor output(batchSize, {outWidth, outHeight}, format, device);

    // Create a vector and fill it with random data.
    std::vector<BT> inputData(input.shape().size());
    FillVector(inputData);

    // Copy generated input data into input tensor
    CopyVectorIntoTensor(input, inputData);
    
    int mapSize = mapBatchSize * mapWidth * mapHeight;

    std::vector<float2> mapData(mapSize);

    if (mapType == REMAP_ABSOLUTE) {
        for (int b = 0; b < mapBatchSize; b++) {
            for (int i = mapHeight - 1; i >= 0; i--) {
                for (int j = 0; j < mapWidth; j++) {
                    int idx = b * mapWidth * mapHeight + (mapHeight - 1 - i) * mapWidth + j;
                    mapData[idx] = make_float2(j, i);  // Direct coordinate mapping
                }
            }
        }
    }
    else if (mapType == REMAP_ABSOLUTE_NORMALIZED) {
        for (int b = 0; b < mapBatchSize; b++) {
            for (int y = 0; y < mapHeight; y++){
                for (int x = 0; x < mapWidth; x++){
                    float normX = ((2.0f * static_cast<float>(x)) / static_cast<float>(mapWidth - 1)) - 1.0f;
                    float normY = ((2.0f * static_cast<float>(y)) / static_cast<float>(mapHeight - 1)) - 1.0f;

                    float srcX = 0.0f - normX;
                    float srcY = 0.0f - normY;

                    int idx = b * mapWidth * mapHeight + y * mapWidth + x;
                    mapData[idx] = make_float2(srcX, srcY);
                }
            }
        }
    }
    else if (mapType == REMAP_RELATIVE_NORMALIZED) {
        for (int b = 0; b < mapBatchSize; b++) {
            for (int y = 0; y < mapHeight; y++){
                for (int x = 0; x < mapWidth; x++){
                    // Generate normalized coordinates in [-1, 1] range
                    float normX = ((2.0f * static_cast<float>(x)) / static_cast<float>(mapWidth - 1)) - 1.0f;
                    float normY = ((2.0f * static_cast<float>(y)) / static_cast<float>(mapHeight - 1)) - 1.0f;

                    // For relative mode, map values represent offsets from the base position
                    // Create a simple displacement pattern (e.g., horizontal flip offset)
                    float offsetX = -normX;  // Reverses horizontal direction
                    float offsetY = -normY;  // Reverses vertical direction

                    int idx = b * mapWidth * mapHeight + y * mapWidth + x;
                    mapData[idx] = make_float2(offsetX, offsetY);
                }
            }
        }
    }

    // Create map tensor and fill it with mapData
    TensorShape map_shape(TensorLayout(eTensorLayout::TENSOR_LAYOUT_NHWC), {mapBatchSize, mapHeight, mapWidth, 2});
    DataType map_dtype(eDataType::DATA_TYPE_F32);
    Tensor mapTensor(map_shape, map_dtype, device);

    CopyVectorIntoTensor(mapTensor, mapData);

    hipStream_t stream;
    HIP_VALIDATE_NO_ERRORS(hipStreamCreate(&stream));
    Remap op;
    op(stream, input, output, mapTensor, InterpType, MapInterpType, mapType, alignCorners, BorderType, borderValue, device);
    HIP_VALIDATE_NO_ERRORS(hipStreamSynchronize(stream));
    HIP_VALIDATE_NO_ERRORS(hipStreamDestroy(stream));

    // Copy data from output tensor into a host allocated vector
    std::vector<BT> result(output.shape().size());
    CopyTensorIntoVector(result, output);

    std::vector<BT> ref = GoldenRemap<T, BorderType, InterpType, MapInterpType>(inputData, batchSize, mapBatchSize, inWidth,
                                                                                        inHeight, outWidth, outHeight, 
                                                                                        mapWidth, mapHeight, mapData, mapType, alignCorners, borderValue);

    // Compare data in actual output versus the generated golden reference image
    CompareVectors(result, ref);
}
}  // namespace

int main(int argc, char** argv) {
    TEST_CASES_BEGIN();

    TEST_CASE((TestCorrectness<uchar1, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_NEAREST,
                               eInterpolationType::INTERP_TYPE_NEAREST>(
        1, 1, 480, 360, 480, 360, 480, 360, FMT_U8, make_float4(0.0f, 0.0f, 0.0f, 1.0f), REMAP_ABSOLUTE, false, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<uchar1, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_NEAREST,
                               eInterpolationType::INTERP_TYPE_NEAREST>(
        1, 1, 480, 360, 480, 360, 480, 360, FMT_U8, make_float4(0.0f, 0.0f, 0.0f, 1.0f), REMAP_ABSOLUTE_NORMALIZED, false, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<uchar1, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_NEAREST,
                               eInterpolationType::INTERP_TYPE_NEAREST>(
        1, 1, 480, 360, 480, 360, 480, 360, FMT_U8, make_float4(0.0f, 0.0f, 0.0f, 1.0f), REMAP_RELATIVE_NORMALIZED, false, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<uchar3, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_NEAREST,
                               eInterpolationType::INTERP_TYPE_NEAREST>(
        1, 1, 480, 360, 480, 360, 480, 360, FMT_RGB8, make_float4(0.0f, 0.0f, 0.0f, 1.0f), REMAP_ABSOLUTE, false, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<uchar3, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_NEAREST,
                               eInterpolationType::INTERP_TYPE_NEAREST>(
        1, 1, 480, 360, 480, 360, 480, 360, FMT_RGB8, make_float4(0.0f, 0.0f, 0.0f, 1.0f), REMAP_ABSOLUTE_NORMALIZED, false, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<uchar3, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_NEAREST,
                               eInterpolationType::INTERP_TYPE_NEAREST>(
        1, 1, 480, 360, 480, 360, 480, 360, FMT_RGB8, make_float4(0.0f, 0.0f, 0.0f, 1.0f), REMAP_RELATIVE_NORMALIZED, false, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<uchar4, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_NEAREST,
                               eInterpolationType::INTERP_TYPE_NEAREST>(
        1, 1, 480, 360, 480, 360, 480, 360, FMT_RGBA8, make_float4(0.0f, 0.0f, 0.0f, 1.0f), REMAP_ABSOLUTE, false, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<uchar4, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_NEAREST,
                               eInterpolationType::INTERP_TYPE_NEAREST>(
        1, 1, 480, 360, 480, 360, 480, 360, FMT_RGBA8, make_float4(0.0f, 0.0f, 0.0f, 1.0f), REMAP_ABSOLUTE_NORMALIZED, false, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<uchar4, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_NEAREST,
                               eInterpolationType::INTERP_TYPE_NEAREST>(
        1, 1, 480, 360, 480, 360, 480, 360, FMT_RGBA8, make_float4(0.0f, 0.0f, 0.0f, 1.0f), REMAP_RELATIVE_NORMALIZED, false, eDeviceType::GPU)));
    
    TEST_CASE((TestCorrectness<uchar1, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_NEAREST,
                               eInterpolationType::INTERP_TYPE_NEAREST>(
        1, 1, 480, 360, 480, 360, 480, 360, FMT_U8, make_float4(0.0f, 0.0f, 0.0f, 1.0f), REMAP_ABSOLUTE, true, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<uchar1, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_NEAREST,
                               eInterpolationType::INTERP_TYPE_NEAREST>(
        1, 1, 480, 360, 480, 360, 480, 360, FMT_U8, make_float4(0.0f, 0.0f, 0.0f, 1.0f), REMAP_ABSOLUTE_NORMALIZED, true, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<uchar1, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_NEAREST,
                               eInterpolationType::INTERP_TYPE_NEAREST>(
        1, 1, 480, 360, 480, 360, 480, 360, FMT_U8, make_float4(0.0f, 0.0f, 0.0f, 1.0f), REMAP_RELATIVE_NORMALIZED, true, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<uchar3, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_NEAREST,
                               eInterpolationType::INTERP_TYPE_NEAREST>(
        1, 1, 480, 360, 480, 360, 480, 360, FMT_RGB8, make_float4(0.0f, 0.0f, 0.0f, 1.0f), REMAP_ABSOLUTE, true, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<uchar3, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_NEAREST,
                               eInterpolationType::INTERP_TYPE_NEAREST>(
        1, 1, 480, 360, 480, 360, 480, 360, FMT_RGB8, make_float4(0.0f, 0.0f, 0.0f, 1.0f), REMAP_ABSOLUTE_NORMALIZED, true, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<uchar3, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_NEAREST,
                               eInterpolationType::INTERP_TYPE_NEAREST>(
        1, 1, 480, 360, 480, 360, 480, 360, FMT_RGB8, make_float4(0.0f, 0.0f, 0.0f, 1.0f), REMAP_RELATIVE_NORMALIZED, true, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<uchar4, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_NEAREST,
                               eInterpolationType::INTERP_TYPE_NEAREST>(
        1, 1, 480, 360, 480, 360, 480, 360, FMT_RGBA8, make_float4(0.0f, 0.0f, 0.0f, 1.0f), REMAP_ABSOLUTE, true, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<uchar4, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_NEAREST,
                               eInterpolationType::INTERP_TYPE_NEAREST>(
        1, 1, 480, 360, 480, 360, 480, 360, FMT_RGBA8, make_float4(0.0f, 0.0f, 0.0f, 1.0f), REMAP_ABSOLUTE_NORMALIZED, true, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<uchar4, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_NEAREST,
                               eInterpolationType::INTERP_TYPE_NEAREST>(
        1, 1, 480, 360, 480, 360, 480, 360, FMT_RGBA8, make_float4(0.0f, 0.0f, 0.0f, 1.0f), REMAP_RELATIVE_NORMALIZED, true, eDeviceType::GPU)));

    TEST_CASE((TestCorrectness<uchar1, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_NEAREST,
                               eInterpolationType::INTERP_TYPE_NEAREST>(
        2, 1, 480, 360, 480, 360, 480, 360, FMT_U8, make_float4(0.0f, 0.0f, 0.0f, 1.0f), REMAP_ABSOLUTE, false, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<uchar1, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_NEAREST,
                               eInterpolationType::INTERP_TYPE_NEAREST>(
        2, 2, 480, 360, 480, 360, 480, 360, FMT_U8, make_float4(0.0f, 0.0f, 0.0f, 1.0f), REMAP_ABSOLUTE, false, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<uchar1, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_NEAREST,
                               eInterpolationType::INTERP_TYPE_NEAREST>(
        2, 1, 480, 360, 480, 360, 480, 360, FMT_U8, make_float4(0.0f, 0.0f, 0.0f, 1.0f), REMAP_ABSOLUTE, true, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<uchar1, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_NEAREST,
                               eInterpolationType::INTERP_TYPE_NEAREST>(
        2, 2, 480, 360, 480, 360, 480, 360, FMT_U8, make_float4(0.0f, 0.0f, 0.0f, 1.0f), REMAP_ABSOLUTE, true, eDeviceType::GPU)));

    TEST_CASE((TestCorrectness<uchar1, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_NEAREST,
                               eInterpolationType::INTERP_TYPE_NEAREST>(
        1, 1, 480, 360, 480, 360, 480, 360, FMT_U8, make_float4(0.0f, 0.0f, 0.0f, 1.0f), REMAP_ABSOLUTE, false, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<uchar1, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_NEAREST,
                               eInterpolationType::INTERP_TYPE_NEAREST>(
        1, 1, 480, 360, 480, 360, 480, 360, FMT_U8, make_float4(0.0f, 0.0f, 0.0f, 1.0f), REMAP_ABSOLUTE_NORMALIZED, false, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<uchar1, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_NEAREST,
                               eInterpolationType::INTERP_TYPE_NEAREST>(
        1, 1, 480, 360, 480, 360, 480, 360, FMT_U8, make_float4(0.0f, 0.0f, 0.0f, 1.0f), REMAP_RELATIVE_NORMALIZED, false, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<uchar3, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_NEAREST,
                               eInterpolationType::INTERP_TYPE_NEAREST>(
        1, 1, 480, 360, 480, 360, 480, 360, FMT_RGB8, make_float4(0.0f, 0.0f, 0.0f, 1.0f), REMAP_ABSOLUTE, false, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<uchar3, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_NEAREST,
                               eInterpolationType::INTERP_TYPE_NEAREST>(
        1, 1, 480, 360, 480, 360, 480, 360, FMT_RGB8, make_float4(0.0f, 0.0f, 0.0f, 1.0f), REMAP_ABSOLUTE_NORMALIZED, false, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<uchar3, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_NEAREST,
                               eInterpolationType::INTERP_TYPE_NEAREST>(
        1, 1, 480, 360, 480, 360, 480, 360, FMT_RGB8, make_float4(0.0f, 0.0f, 0.0f, 1.0f), REMAP_RELATIVE_NORMALIZED, false, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<uchar4, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_NEAREST,
                               eInterpolationType::INTERP_TYPE_NEAREST>(
        1, 1, 480, 360, 480, 360, 480, 360, FMT_RGBA8, make_float4(0.0f, 0.0f, 0.0f, 1.0f), REMAP_ABSOLUTE, false, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<uchar4, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_NEAREST,
                               eInterpolationType::INTERP_TYPE_NEAREST>(
        1, 1, 480, 360, 480, 360, 480, 360, FMT_RGBA8, make_float4(0.0f, 0.0f, 0.0f, 1.0f), REMAP_ABSOLUTE_NORMALIZED, false, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<uchar4, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_NEAREST,
                               eInterpolationType::INTERP_TYPE_NEAREST>(
        1, 1, 480, 360, 480, 360, 480, 360, FMT_RGBA8, make_float4(0.0f, 0.0f, 0.0f, 1.0f), REMAP_RELATIVE_NORMALIZED, false, eDeviceType::CPU)));

    TEST_CASE((TestCorrectness<uchar1, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_NEAREST,
                               eInterpolationType::INTERP_TYPE_NEAREST>(
        1, 1, 480, 360, 480, 360, 480, 360, FMT_U8, make_float4(0.0f, 0.0f, 0.0f, 1.0f), REMAP_ABSOLUTE, true, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<uchar1, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_NEAREST,
                               eInterpolationType::INTERP_TYPE_NEAREST>(
        1, 1, 480, 360, 480, 360, 480, 360, FMT_U8, make_float4(0.0f, 0.0f, 0.0f, 1.0f), REMAP_ABSOLUTE_NORMALIZED, true, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<uchar1, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_NEAREST,
                               eInterpolationType::INTERP_TYPE_NEAREST>(
        1, 1, 480, 360, 480, 360, 480, 360, FMT_U8, make_float4(0.0f, 0.0f, 0.0f, 1.0f), REMAP_RELATIVE_NORMALIZED, true, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<uchar3, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_NEAREST,
                               eInterpolationType::INTERP_TYPE_NEAREST>(
        1, 1, 480, 360, 480, 360, 480, 360, FMT_RGB8, make_float4(0.0f, 0.0f, 0.0f, 1.0f), REMAP_ABSOLUTE, true, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<uchar3, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_NEAREST,
                               eInterpolationType::INTERP_TYPE_NEAREST>(
        1, 1, 480, 360, 480, 360, 480, 360, FMT_RGB8, make_float4(0.0f, 0.0f, 0.0f, 1.0f), REMAP_ABSOLUTE_NORMALIZED, true, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<uchar3, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_NEAREST,
                               eInterpolationType::INTERP_TYPE_NEAREST>(
        1, 1, 480, 360, 480, 360, 480, 360, FMT_RGB8, make_float4(0.0f, 0.0f, 0.0f, 1.0f), REMAP_RELATIVE_NORMALIZED, true, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<uchar4, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_NEAREST,
                               eInterpolationType::INTERP_TYPE_NEAREST>(
        1, 1, 480, 360, 480, 360, 480, 360, FMT_RGBA8, make_float4(0.0f, 0.0f, 0.0f, 1.0f), REMAP_ABSOLUTE, true, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<uchar4, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_NEAREST,
                               eInterpolationType::INTERP_TYPE_NEAREST>(
        1, 1, 480, 360, 480, 360, 480, 360, FMT_RGBA8, make_float4(0.0f, 0.0f, 0.0f, 1.0f), REMAP_ABSOLUTE_NORMALIZED, true, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<uchar4, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_NEAREST,
                               eInterpolationType::INTERP_TYPE_NEAREST>(
        1, 1, 480, 360, 480, 360, 480, 360, FMT_RGBA8, make_float4(0.0f, 0.0f, 0.0f, 1.0f), REMAP_RELATIVE_NORMALIZED, true, eDeviceType::CPU)));

    TEST_CASE((TestCorrectness<uchar1, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_NEAREST,
                               eInterpolationType::INTERP_TYPE_NEAREST>(
        2, 1, 480, 360, 480, 360, 480, 360, FMT_U8, make_float4(0.0f, 0.0f, 0.0f, 1.0f), REMAP_ABSOLUTE, false, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<uchar1, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_NEAREST,
                               eInterpolationType::INTERP_TYPE_NEAREST>(
        2, 2, 480, 360, 480, 360, 480, 360, FMT_U8, make_float4(0.0f, 0.0f, 0.0f, 1.0f), REMAP_ABSOLUTE, false, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<uchar1, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_NEAREST,
                               eInterpolationType::INTERP_TYPE_NEAREST>(
        2, 1, 480, 360, 480, 360, 480, 360, FMT_U8, make_float4(0.0f, 0.0f, 0.0f, 1.0f), REMAP_ABSOLUTE, true, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<uchar1, eBorderType::BORDER_TYPE_CONSTANT, eInterpolationType::INTERP_TYPE_NEAREST,
                               eInterpolationType::INTERP_TYPE_NEAREST>(
        2, 2, 480, 360, 480, 360, 480, 360, FMT_U8, make_float4(0.0f, 0.0f, 0.0f, 1.0f), REMAP_ABSOLUTE, true, eDeviceType::CPU)));



    TEST_CASES_END();
}