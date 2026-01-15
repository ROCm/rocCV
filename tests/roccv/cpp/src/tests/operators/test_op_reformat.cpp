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

#include <core/detail/type_traits.hpp>
#include <core/tensor.hpp>
#include <core/wrappers/generic_tensor_wrapper.hpp>

#include "op_reformat.hpp"
#include "test_helpers.hpp"

using namespace roccv;
using namespace roccv::tests;
using namespace roccv::detail;

namespace {

/**
 * @brief Calculates the shape of a tensor from a layout, batch size, width, height, and channels.
 *
 * @param[in] layout The layout of the tensor.
 * @param[in] batchSize The batch size of the tensor.
 * @param[in] width The width of the tensor.
 * @param[in] height The height of the tensor.
 * @param[in] channels The channels of the tensor.
 * @return The shape of the tensor.
 */
static std::array<int64_t, ROCCV_TENSOR_MAX_RANK> GetShapeFromLayout(const TensorLayout& layout, int32_t batchSize,
                                                                     int32_t width, int32_t height, int32_t channels) {
    std::array<int64_t, ROCCV_TENSOR_MAX_RANK> shape;
    if (layout.batch_index() != -1) {
        shape[layout.batch_index()] = batchSize;
    }

    shape[layout.height_index()] = height;
    shape[layout.width_index()] = width;
    shape[layout.channels_index()] = channels;
    return shape;
}

/**
 * @brief Gets an element from a generic tensor wrapper based on a layout.
 *
 * @tparam T The datatype of the tensor.
 * @param[in] wrapper The generic tensor wrapper.
 * @param[in] layout The layout of the tensor.
 * @param[in] batchIndex The batch index of the element.
 * @param[in] heightIndex The height index of the element.
 * @param[in] widthIndex The width index of the element.
 * @param[in] channelsIndex The channels index of the element.
 * @return The element from the generic tensor wrapper based on the layout.
 */
template <typename T>
static T& GetElement(GenericTensorWrapper<T>& wrapper, const TensorLayout& layout, int32_t batchIndex,
                     int32_t heightIndex, int32_t widthIndex, int32_t channelsIndex) {
    switch (layout.elayout()) {
        case eTensorLayout::TENSOR_LAYOUT_NHWC:
            return wrapper.at(batchIndex, heightIndex, widthIndex, channelsIndex);
        case eTensorLayout::TENSOR_LAYOUT_NCHW:
            return wrapper.at(batchIndex, channelsIndex, heightIndex, widthIndex);
        case eTensorLayout::TENSOR_LAYOUT_HWC:
            return wrapper.at(heightIndex, widthIndex, channelsIndex);
        case eTensorLayout::TENSOR_LAYOUT_CHW:
            return wrapper.at(channelsIndex, heightIndex, widthIndex);
        default:
            throw Exception("Unsupported layout for Reformat", eStatusType::INVALID_VALUE);
    }
}

/**
 * @brief Golden model for reformatting a tensor from one layout to another.
 *
 * @tparam T The datatype of the tensor.
 * @param[in] input The input tensor data.
 * @param[in] batchSize The batch size of the tensor.
 * @param[in] width The width of the tensor.
 * @param[in] height The height of the tensor.
 * @param[in] channels The channels of the tensor.
 * @param[in] dtype The datatype of the tensor.
 * @param[in] inLayout The input layout of the tensor.
 * @param[in] outLayout The output layout of the tensor.
 * @return The output tensor data.
 */
template <typename T>
static std::vector<T> GoldenReformat(std::vector<T>& input, int32_t batchSize, int32_t width, int32_t height,
                                     int32_t channels, const DataType& dtype, const TensorLayout& inLayout,
                                     const TensorLayout& outLayout) {
    // Calculate the shape and strides for the input and output tensors.
    std::array<int64_t, ROCCV_TENSOR_MAX_RANK> inShape =
        GetShapeFromLayout(inLayout, batchSize, width, height, channels);
    std::array<int64_t, ROCCV_TENSOR_MAX_RANK> inStrides = ComputePackedStrides(inShape, dtype, inLayout.rank());
    std::array<int64_t, ROCCV_TENSOR_MAX_RANK> outShape =
        GetShapeFromLayout(outLayout, batchSize, width, height, channels);
    std::array<int64_t, ROCCV_TENSOR_MAX_RANK> outStrides = ComputePackedStrides(outShape, dtype, outLayout.rank());

    // Create the output data and wrappers.
    std::vector<T> outputData(batchSize * width * height * channels);
    GenericTensorWrapper<T> outputWrapper(outputData.data(), outShape, outStrides, outLayout.rank());
    GenericTensorWrapper<T> inputWrapper(input.data(), inShape, inStrides, inLayout.rank());

    for (int32_t b = 0; b < batchSize; b++) {
        for (int32_t y = 0; y < height; y++) {
            for (int32_t x = 0; x < width; x++) {
                for (int32_t c = 0; c < channels; c++) {
                    GetElement(outputWrapper, outLayout, b, y, x, c) = GetElement(inputWrapper, inLayout, b, y, x, c);
                }
            }
        }
    }

    return outputData;
}

/**
 * @brief Tests correctness of the Reformat operator, comparing it against a generated golden result.
 *
 * @tparam T The datatype of the tensor.
 * @param[in] batchSize The batch size of the tensor.
 * @param[in] width The width of the tensor.
 * @param[in] height The height of the tensor.
 * @param[in] channels The channels of the tensor.
 * @param[in] inLayout The input layout of the tensor.
 * @param[in] outLayout The output layout of the tensor.
 * @param[in] dtype The datatype of the tensor.
 * @param[in] device The device to run the roccv::Reformat operator on.
 * @throws std::runtime_error on test failure.
 */
template <typename T>
static void TestCorrectness(int batchSize, int width, int height, int channels, const TensorLayout& inLayout,
                            const TensorLayout& outLayout, const DataType& dtype, eDeviceType device) {
    // Create input and output tensors based on test parameters
    std::array<int64_t, ROCCV_TENSOR_MAX_RANK> inputShapeData =
        GetShapeFromLayout(inLayout, batchSize, width, height, channels);
    std::array<int64_t, ROCCV_TENSOR_MAX_RANK> outputShapeData =
        GetShapeFromLayout(outLayout, batchSize, width, height, channels);

    TensorShape inputShape(inputShapeData, inLayout.rank(), inLayout.elayout());
    TensorShape outputShape(outputShapeData, outLayout.rank(), outLayout.elayout());

    Tensor input(inputShape, dtype, device);
    Tensor output(outputShape, dtype, device);

    // Create a vector and fill it with random data.
    std::vector<T> inputData(input.shape().size());
    FillVector(inputData);

    CopyVectorIntoTensor(input, inputData);

    // Obtain golden results
    std::vector<T> goldenOutput =
        GoldenReformat<T>(inputData, batchSize, width, height, channels, dtype, inLayout, outLayout);

    // Call roccv::Reformat operator to obtain actual results
    hipStream_t stream;
    HIP_VALIDATE_NO_ERRORS(hipStreamCreate(&stream));

    Reformat op;
    op(stream, input, output, device);
    HIP_VALIDATE_NO_ERRORS(hipStreamSynchronize(stream));
    HIP_VALIDATE_NO_ERRORS(hipStreamDestroy(stream));

    // Copy actual results into host allocated vector
    std::vector<T> actualOutput(output.shape().size());
    CopyTensorIntoVector(actualOutput, output);

    // Compare actual results with golden results
    CompareVectors(actualOutput, goldenOutput);
}

static void TestNegativeReformat() {
    TensorShape validLayoutShape(TensorLayout(eTensorLayout::TENSOR_LAYOUT_NHWC), {1, 1, 1, 1});
    Tensor validGPUTensor(validLayoutShape, DataType(eDataType::DATA_TYPE_U8), eDeviceType::GPU);
    Tensor validCPUTensor(validLayoutShape, DataType(eDataType::DATA_TYPE_U8), eDeviceType::CPU);

    // Test unsupported layout
    {
        TensorShape invalidLayoutShape(TensorLayout(eTensorLayout::TENSOR_LAYOUT_NC), {1, 1});
        Tensor invalidTensor(invalidLayoutShape, DataType(eDataType::DATA_TYPE_U8), eDeviceType::GPU);
        Reformat op;
        EXPECT_EXCEPTION(op(nullptr, invalidTensor, validGPUTensor, eDeviceType::GPU),
                         eStatusType::INVALID_COMBINATION);
    }
}

}  // namespace

int main(int argc, char** argv) {
    TEST_CASES_BEGIN();

    // clang-format off

    // GPU Tests - NHWC to NCHW
    TEST_CASE(TestCorrectness<unsigned char >(1, 32, 24, 3, TensorLayout(eTensorLayout::TENSOR_LAYOUT_NHWC), TensorLayout(eTensorLayout::TENSOR_LAYOUT_NCHW), DataType(eDataType::DATA_TYPE_U8 ),  eDeviceType::GPU));
    TEST_CASE(TestCorrectness<unsigned short>(3, 64, 48, 4, TensorLayout(eTensorLayout::TENSOR_LAYOUT_NHWC), TensorLayout(eTensorLayout::TENSOR_LAYOUT_NCHW), DataType(eDataType::DATA_TYPE_U16),  eDeviceType::GPU));
    TEST_CASE(TestCorrectness<float         >(7, 16,  8, 1, TensorLayout(eTensorLayout::TENSOR_LAYOUT_NHWC), TensorLayout(eTensorLayout::TENSOR_LAYOUT_NCHW), DataType(eDataType::DATA_TYPE_F32),  eDeviceType::GPU));

    // GPU Tests - NCHW to NHWC
    TEST_CASE(TestCorrectness<unsigned char >(1, 32, 24, 3, TensorLayout(eTensorLayout::TENSOR_LAYOUT_NCHW), TensorLayout(eTensorLayout::TENSOR_LAYOUT_NHWC), DataType(eDataType::DATA_TYPE_U8 ),  eDeviceType::GPU));
    TEST_CASE(TestCorrectness<unsigned short>(3, 64, 48, 4, TensorLayout(eTensorLayout::TENSOR_LAYOUT_NCHW), TensorLayout(eTensorLayout::TENSOR_LAYOUT_NHWC), DataType(eDataType::DATA_TYPE_U16),  eDeviceType::GPU));
    TEST_CASE(TestCorrectness<float         >(7, 16,  8, 1, TensorLayout(eTensorLayout::TENSOR_LAYOUT_NCHW), TensorLayout(eTensorLayout::TENSOR_LAYOUT_NHWC), DataType(eDataType::DATA_TYPE_F32),  eDeviceType::GPU));

    // GPU Tests - HWC to NHWC
    TEST_CASE(TestCorrectness<unsigned char >(1, 32, 24, 3, TensorLayout(eTensorLayout::TENSOR_LAYOUT_HWC), TensorLayout(eTensorLayout::TENSOR_LAYOUT_NHWC), DataType(eDataType::DATA_TYPE_U8 ),  eDeviceType::GPU));
    TEST_CASE(TestCorrectness<unsigned short>(1, 64, 48, 4, TensorLayout(eTensorLayout::TENSOR_LAYOUT_HWC), TensorLayout(eTensorLayout::TENSOR_LAYOUT_NHWC), DataType(eDataType::DATA_TYPE_U16),  eDeviceType::GPU));
    TEST_CASE(TestCorrectness<float         >(1, 16,  8, 1, TensorLayout(eTensorLayout::TENSOR_LAYOUT_HWC), TensorLayout(eTensorLayout::TENSOR_LAYOUT_NHWC), DataType(eDataType::DATA_TYPE_F32),  eDeviceType::GPU));

    // GPU Tests - NHWC to HWC
    TEST_CASE(TestCorrectness<unsigned char >(1, 32, 24, 3, TensorLayout(eTensorLayout::TENSOR_LAYOUT_NHWC), TensorLayout(eTensorLayout::TENSOR_LAYOUT_HWC), DataType(eDataType::DATA_TYPE_U8 ),  eDeviceType::GPU));
    TEST_CASE(TestCorrectness<unsigned short>(1, 64, 48, 4, TensorLayout(eTensorLayout::TENSOR_LAYOUT_NHWC), TensorLayout(eTensorLayout::TENSOR_LAYOUT_HWC), DataType(eDataType::DATA_TYPE_U16),  eDeviceType::GPU));
    TEST_CASE(TestCorrectness<float         >(1, 16,  8, 1, TensorLayout(eTensorLayout::TENSOR_LAYOUT_NHWC), TensorLayout(eTensorLayout::TENSOR_LAYOUT_HWC), DataType(eDataType::DATA_TYPE_F32),  eDeviceType::GPU));

    // GPU Tests - HWC to NCHW
    TEST_CASE(TestCorrectness<unsigned char >(1, 32, 24, 3, TensorLayout(eTensorLayout::TENSOR_LAYOUT_HWC), TensorLayout(eTensorLayout::TENSOR_LAYOUT_NCHW), DataType(eDataType::DATA_TYPE_U8 ),  eDeviceType::GPU));
    TEST_CASE(TestCorrectness<unsigned short>(1, 64, 48, 4, TensorLayout(eTensorLayout::TENSOR_LAYOUT_HWC), TensorLayout(eTensorLayout::TENSOR_LAYOUT_NCHW), DataType(eDataType::DATA_TYPE_U16),  eDeviceType::GPU));
    TEST_CASE(TestCorrectness<float         >(1, 16,  8, 1, TensorLayout(eTensorLayout::TENSOR_LAYOUT_HWC), TensorLayout(eTensorLayout::TENSOR_LAYOUT_NCHW), DataType(eDataType::DATA_TYPE_F32),  eDeviceType::GPU));
    
    // CPU Tests - NHWC to NCHW
    TEST_CASE(TestCorrectness<unsigned char >(1, 32, 24, 3, TensorLayout(eTensorLayout::TENSOR_LAYOUT_NHWC), TensorLayout(eTensorLayout::TENSOR_LAYOUT_NCHW), DataType(eDataType::DATA_TYPE_U8 ),  eDeviceType::CPU));
    TEST_CASE(TestCorrectness<signed  short >(3, 64, 48, 4, TensorLayout(eTensorLayout::TENSOR_LAYOUT_NHWC), TensorLayout(eTensorLayout::TENSOR_LAYOUT_NCHW), DataType(eDataType::DATA_TYPE_S16),  eDeviceType::CPU));
    TEST_CASE(TestCorrectness<double        >(7, 16,  8, 1, TensorLayout(eTensorLayout::TENSOR_LAYOUT_NHWC), TensorLayout(eTensorLayout::TENSOR_LAYOUT_NCHW), DataType(eDataType::DATA_TYPE_F64),  eDeviceType::CPU));

    // CPU Tests - NCHW to NHWC
    TEST_CASE(TestCorrectness<unsigned char >(1, 32, 24, 3, TensorLayout(eTensorLayout::TENSOR_LAYOUT_NCHW), TensorLayout(eTensorLayout::TENSOR_LAYOUT_NHWC), DataType(eDataType::DATA_TYPE_U8 ),  eDeviceType::CPU));
    TEST_CASE(TestCorrectness<signed  short >(3, 64, 48, 4, TensorLayout(eTensorLayout::TENSOR_LAYOUT_NCHW), TensorLayout(eTensorLayout::TENSOR_LAYOUT_NHWC), DataType(eDataType::DATA_TYPE_S16),  eDeviceType::CPU));
    TEST_CASE(TestCorrectness<double        >(7, 16,  8, 1, TensorLayout(eTensorLayout::TENSOR_LAYOUT_NCHW), TensorLayout(eTensorLayout::TENSOR_LAYOUT_NHWC), DataType(eDataType::DATA_TYPE_F64),  eDeviceType::CPU));

    // clang-format on

    // Negative Tests
    TEST_CASE(TestNegativeReformat());

    TEST_CASES_END();
}