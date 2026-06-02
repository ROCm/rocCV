/**
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

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

#include <core/detail/casting.hpp>
#include <core/detail/type_traits.hpp>
#include <core/wrappers/image_wrapper.hpp>
#include <op_brightness_contrast.hpp>

#include "test_helpers.hpp"

using namespace roccv;
using namespace roccv::tests;

// Keep all non-entrypoint functions in an anonymous namespace to prevent redefinition errors across translation units.
namespace {

/**
 * @brief Golden model for the Brightness Contrast operation.
 *
 * @tparam SRC_DT Vectorized datatype of the source image's pixels.
 * @tparam DEST_DT Vectorized datatype of the destination image's pixels.
 * @tparam BC_DT Datatype of the brightness/contrast parameters.
 * @tparam BT_SRC Base type of the source image's data.
 * @tparam BT_DEST Base type of the destination image's data.
 * @param[in] input An input vector containing image data.
 * @param[in] batchSize The number of images in the batch.
 * @param[in] width Image width.
 * @param[in] height Image height.
 * @param[in] brightness Vector of brightness multiplier values (one per image in the batch).
 * @param[in] contrast Vector of contrast multiplier values (one per image in the batch).
 * @param[in] brightnessShift Vector of brightness offset values (one per image in the batch).
 * @param[in] contrastCenter Vector of contrast center points (one per image in the batch).
 * @return Vector containing the results of the operation.
 */
template <typename SRC_DT, typename DEST_DT, typename BC_DT, typename BT_SRC = detail::BaseType<SRC_DT>,
          typename BT_DEST = detail::BaseType<DEST_DT>>
std::vector<BT_DEST> GoldenBrightnessContrast(std::vector<BT_SRC>& input, int32_t batchSize, int32_t width,
                                              int32_t height, std::vector<BC_DT>& brightness,
                                              std::vector<BC_DT>& contrast, std::vector<BC_DT>& brightnessShift,
                                              std::vector<BC_DT>& contrastCenter) {
    // Create an output vector the same size as the input vector
    std::vector<BT_DEST> output(input.size());

    // Wrap input/output vectors for simplified data access
    ImageWrapper<SRC_DT> src(input, batchSize, width, height);
    ImageWrapper<DEST_DT> dst(output, batchSize, width, height);

    using work_type = detail::MakeType<BC_DT, detail::NumElements<DEST_DT>>;

    for (int b = 0; b < batchSize; ++b) {
        BC_DT brt = brightness[b];
        BC_DT contr = contrast[b];
        BC_DT brtShift = brightnessShift[b];
        BC_DT contrCenter = contrastCenter[b];

        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                work_type src_val = detail::StaticCast<work_type>(src.at(b, y, x, 0));
                work_type result = brtShift + brt * (contrCenter + contr * (src_val - contrCenter));
                dst.at(b, y, x, 0) = detail::SaturateCast<DEST_DT>(result);
            }
        }
    }
    return output;
}

template <typename SRC_DT, typename DEST_DT, typename BC_DT, typename BT_SRC = detail::BaseType<SRC_DT>,
          typename BT_DEST = detail::BaseType<DEST_DT>>
void TestCorrectness(int batchSize, int width, int height, ImageFormat inFormat, ImageFormat outFormat,
                     BC_DT brightness, BC_DT contrast, BC_DT brightnessShift, BC_DT contrastCenter, eDataType bc_edt,
                     eDeviceType device) {
    // Create input and output tensor based on test parameters
    Tensor input(batchSize, {width, height}, inFormat, device);
    Tensor output(batchSize, {width, height}, outFormat, device);

    // Create a vector and fill it with random data.
    std::vector<BT_SRC> inputData(input.shape().size());
    FillVector(inputData);

    // Copy generated input data into input tensor
    CopyVectorIntoTensor(input, inputData);

    // Create brightness/contrast tensors
    TensorShape shape(TensorLayout(eTensorLayout::TENSOR_LAYOUT_N), {batchSize});
    Tensor brightnessTensor(shape, DataType(bc_edt), device);
    Tensor contrastTensor(shape, DataType(bc_edt), device);
    Tensor brightnessShiftTensor(shape, DataType(bc_edt), device);
    Tensor contrastCenterTensor(shape, DataType(bc_edt), device);

    // Create BC vectors filled with passed in values, use to fill in tensors
    std::vector<BC_DT> brightnessData(batchSize, brightness);
    std::vector<BC_DT> contrastData(batchSize, contrast);
    std::vector<BC_DT> brightnessShiftData(batchSize, brightnessShift);
    std::vector<BC_DT> contrastCenterData(batchSize, contrastCenter);

    CopyVectorIntoTensor(brightnessTensor, brightnessData);
    CopyVectorIntoTensor(contrastTensor, contrastData);
    CopyVectorIntoTensor(brightnessShiftTensor, brightnessShiftData);
    CopyVectorIntoTensor(contrastCenterTensor, contrastCenterData);

    // Calculate golden output reference
    std::vector<BT_DEST> ref = GoldenBrightnessContrast<SRC_DT, DEST_DT, BC_DT>(
        inputData, batchSize, width, height, brightnessData, contrastData, brightnessShiftData, contrastCenterData);

    // Run roccv::Brightness Contrast operator to obtain actual results
    hipStream_t stream;
    HIP_VALIDATE_NO_ERRORS(hipStreamCreate(&stream));
    BrightnessContrast op;
    op(stream, input, output, brightnessTensor, contrastTensor, brightnessShiftTensor, contrastCenterTensor, device);
    HIP_VALIDATE_NO_ERRORS(hipStreamSynchronize(stream));
    HIP_VALIDATE_NO_ERRORS(hipStreamDestroy(stream));

    // Copy data from output tensor into a host allocated vector
    std::vector<BT_DEST> result(output.shape().size());
    CopyTensorIntoVector(result, output);

    // Compare data in actual output versus the generated golden reference image
    // Using 1.0E-4 as the error threshold to account for FMA/non-FMA float divergence between CPU and GPU.
    CompareVectorsNear(result, ref, 1.0E-4);  // tightest threshold passing
}

template <typename SRC_DT, typename DEST_DT, typename BC_DT, typename BT_SRC = detail::BaseType<SRC_DT>,
          typename BT_DEST = detail::BaseType<DEST_DT>>
void TestCorrectnessDefaultsAndPer(int batchSize, int width, int height, ImageFormat inFormat, ImageFormat outFormat,
                                   eDeviceType device) {
    // Create input and output tensors based on test parameters
    Tensor input(batchSize, {width, height}, inFormat, device);
    Tensor defaultOutput(batchSize, {width, height}, outFormat, device);
    Tensor randOutput(batchSize, {width, height}, outFormat, device);

    // Create a vector and fill it with random data.
    std::vector<BT_SRC> inputData(input.shape().size());
    FillVector(inputData);

    // Copy generated input data into input tensor
    CopyVectorIntoTensor(input, inputData);

    // Create BC vectors according to default vals
    eDataType input_dtype = input.dtype().etype();
    // Note: this same lambda is used in the actual op (dependency?)
    auto compute_cc_default = [&]() -> double {
        switch (input_dtype) {
            case eDataType::DATA_TYPE_U8:
                return 1u << (8 - 1);
            case eDataType::DATA_TYPE_U16:
                return 1u << (16 - 1);
            case eDataType::DATA_TYPE_S16:
                return 1u << (16 - 2);
            case eDataType::DATA_TYPE_S32:
                return 1u << (32 - 2);
            case eDataType::DATA_TYPE_F32:
                return 0.5;
            default:
                return 0.5;
        }
    };
    BC_DT trueContrastCenter = static_cast<BC_DT>(compute_cc_default());
    std::vector<BC_DT> brightnessDataDefault(batchSize, static_cast<BC_DT>(1.0));
    std::vector<BC_DT> contrastDataDefault(batchSize, static_cast<BC_DT>(1.0));
    std::vector<BC_DT> brightnessShiftDataDefault(batchSize, static_cast<BC_DT>(0.0));
    std::vector<BC_DT> contrastCenterDataDefault(batchSize, trueContrastCenter);

    // Create BC vectors with rand values different per sample, use to fill in tensors
    std::vector<BC_DT> brightnessDataRand(batchSize);
    std::vector<BC_DT> contrastDataRand(batchSize);
    std::vector<BC_DT> brightnessShiftDataRand(batchSize);
    std::vector<BC_DT> contrastCenterDataRand(batchSize);
    FillVectorRange(brightnessDataRand, static_cast<BC_DT>(0.5), static_cast<BC_DT>(2.0));
    FillVectorRange(contrastDataRand, static_cast<BC_DT>(0.5), static_cast<BC_DT>(2.0));
    FillVectorRange(brightnessShiftDataRand, static_cast<BC_DT>(-0.25), static_cast<BC_DT>(0.25));
    BC_DT centerOffset = trueContrastCenter * static_cast<BC_DT>(0.1);
    FillVectorRange(contrastCenterDataRand, trueContrastCenter - centerOffset, trueContrastCenter + centerOffset);

    constexpr eDataType bc_edt = std::is_same_v<BC_DT, double> ? eDataType::DATA_TYPE_F64 : eDataType::DATA_TYPE_F32;
    TensorShape shape(TensorLayout(eTensorLayout::TENSOR_LAYOUT_N), {batchSize});
    Tensor brightnessTensor(shape, DataType(bc_edt), device);
    Tensor contrastTensor(shape, DataType(bc_edt), device);
    Tensor brightnessShiftTensor(shape, DataType(bc_edt), device);
    Tensor contrastCenterTensor(shape, DataType(bc_edt), device);
    CopyVectorIntoTensor(brightnessTensor, brightnessDataRand);
    CopyVectorIntoTensor(contrastTensor, contrastDataRand);
    CopyVectorIntoTensor(brightnessShiftTensor, brightnessShiftDataRand);
    CopyVectorIntoTensor(contrastCenterTensor, contrastCenterDataRand);

    // Calculate golden output reference for default vals
    std::vector<BT_DEST> defaultRef = GoldenBrightnessContrast<SRC_DT, DEST_DT, BC_DT>(
        inputData, batchSize, width, height, brightnessDataDefault, contrastDataDefault, brightnessShiftDataDefault,
        contrastCenterDataDefault);
    // Calculate golden output reference for random per sample vals
    std::vector<BT_DEST> randRef = GoldenBrightnessContrast<SRC_DT, DEST_DT, BC_DT>(
        inputData, batchSize, width, height, brightnessDataRand, contrastDataRand, brightnessShiftDataRand,
        contrastCenterDataRand);

    // Run roccv::Brightness Contrast operator to obtain actual results using default vals
    hipStream_t stream;
    HIP_VALIDATE_NO_ERRORS(hipStreamCreate(&stream));
    BrightnessContrast op;
    op(stream, input, defaultOutput, std::nullopt, std::nullopt, std::nullopt, std::nullopt, device);
    HIP_VALIDATE_NO_ERRORS(hipStreamSynchronize(stream));

    // Copy data from output tensor into a host allocated vector
    std::vector<BT_DEST> defaultResult(defaultOutput.shape().size());
    CopyTensorIntoVector(defaultResult, defaultOutput);

    // Run roccv::Brightness Contrast operator to obtain actual results using rand per sample vals
    op(stream, input, randOutput, brightnessTensor, contrastTensor, brightnessShiftTensor, contrastCenterTensor,
       device);
    HIP_VALIDATE_NO_ERRORS(hipStreamSynchronize(stream));
    HIP_VALIDATE_NO_ERRORS(hipStreamDestroy(stream));

    // Copy data from output tensor into a host allocated vector
    std::vector<BT_DEST> randResult(randOutput.shape().size());
    CopyTensorIntoVector(randResult, randOutput);

    // Compare data in actual output versus the generated golden reference image
    // Using 1.0E-4 as the error threshold to account for FMA/non-FMA float divergence between CPU and GPU.
    CompareVectorsNear(defaultResult, defaultRef, 1.0E-4);
    CompareVectorsNear(randResult, randRef, 1.0E-4);
}

void TestNegativeBrightnessContrast() {
    TensorShape validShape(TensorLayout(eTensorLayout::TENSOR_LAYOUT_NHWC), {1, 1, 1, 1});
    Tensor validGPUTensor(validShape, DataType(eDataType::DATA_TYPE_U8), eDeviceType::GPU);
    Tensor validCPUTensor(validShape, DataType(eDataType::DATA_TYPE_U8), eDeviceType::CPU);

    TensorShape bc_validOneShape(TensorLayout(eTensorLayout::TENSOR_LAYOUT_N), {1});

    BrightnessContrast op;

    {
        // Test wrong device
        Tensor bc_validCPUTensor(bc_validOneShape, DataType(eDataType::DATA_TYPE_F32), eDeviceType::CPU);

        EXPECT_EXCEPTION(op(nullptr, validCPUTensor, validGPUTensor, std::nullopt, std::nullopt, std::nullopt,
                            std::nullopt, eDeviceType::GPU),
                         eStatusType::INVALID_OPERATION);
        EXPECT_EXCEPTION(op(nullptr, validGPUTensor, validCPUTensor, std::nullopt, std::nullopt, std::nullopt,
                            std::nullopt, eDeviceType::GPU),
                         eStatusType::INVALID_COMBINATION);
        EXPECT_EXCEPTION(op(nullptr, validGPUTensor, validGPUTensor, bc_validCPUTensor, bc_validCPUTensor,
                            bc_validCPUTensor, bc_validCPUTensor, eDeviceType::GPU),
                         eStatusType::INVALID_COMBINATION);
    }

    {
        // Test unsupported input/output data type
        Tensor invalidTensor(validGPUTensor.shape(), DataType(eDataType::DATA_TYPE_U32), eDeviceType::GPU);
        EXPECT_EXCEPTION(op(nullptr, invalidTensor, validGPUTensor, std::nullopt, std::nullopt, std::nullopt,
                            std::nullopt, eDeviceType::GPU),
                         eStatusType::NOT_IMPLEMENTED);
        EXPECT_EXCEPTION(op(nullptr, validGPUTensor, invalidTensor, std::nullopt, std::nullopt, std::nullopt,
                            std::nullopt, eDeviceType::GPU),
                         eStatusType::NOT_IMPLEMENTED);
    }

    {
        // Test unsupported input/output layout
        TensorShape invalidLayoutShape(TensorLayout(eTensorLayout::TENSOR_LAYOUT_NC), {1, 1});
        Tensor invalidTensor(invalidLayoutShape, DataType(eDataType::DATA_TYPE_U8), eDeviceType::GPU);
        EXPECT_EXCEPTION(op(nullptr, invalidTensor, validGPUTensor, std::nullopt, std::nullopt, std::nullopt,
                            std::nullopt, eDeviceType::GPU),
                         eStatusType::INVALID_COMBINATION);
        EXPECT_EXCEPTION(op(nullptr, validGPUTensor, invalidTensor, std::nullopt, std::nullopt, std::nullopt,
                            std::nullopt, eDeviceType::GPU),
                         eStatusType::INVALID_COMBINATION);
    }

    {
        // Test input/output shape mismatch
        Tensor invalidTensor(TensorShape(validGPUTensor.layout(), {2, 2, 2, 2}), DataType(eDataType::DATA_TYPE_U8),
                             eDeviceType::GPU);
        EXPECT_EXCEPTION(op(nullptr, invalidTensor, validGPUTensor, std::nullopt, std::nullopt, std::nullopt,
                            std::nullopt, eDeviceType::GPU),
                         eStatusType::INVALID_COMBINATION);
    }

    {
        // Test brightness/contrast datatype mismatch with input/output
        Tensor bc_validFloatTensor(bc_validOneShape, DataType(eDataType::DATA_TYPE_F32), eDeviceType::GPU);
        Tensor bc_validDoubleTensor(bc_validOneShape, DataType(eDataType::DATA_TYPE_F64), eDeviceType::GPU);
        Tensor bc_invalidCharTensor(bc_validOneShape, DataType(eDataType::DATA_TYPE_U8), eDeviceType::GPU);

        Tensor validIntTensor(validShape, DataType(eDataType::DATA_TYPE_S32), eDeviceType::GPU);
        Tensor validCharTensor(validShape, DataType(eDataType::DATA_TYPE_U8), eDeviceType::GPU);

        EXPECT_EXCEPTION(op(nullptr, validCharTensor, validCharTensor, bc_invalidCharTensor, std::nullopt, std::nullopt,
                            std::nullopt, eDeviceType::GPU),
                         eStatusType::INVALID_COMBINATION);

        EXPECT_EXCEPTION(op(nullptr, validCharTensor, validCharTensor, std::nullopt, bc_validDoubleTensor, std::nullopt,
                            std::nullopt, eDeviceType::GPU),
                         eStatusType::INVALID_COMBINATION);

        EXPECT_EXCEPTION(op(nullptr, validIntTensor, validCharTensor, std::nullopt, std::nullopt, bc_validFloatTensor,
                            std::nullopt, eDeviceType::GPU),
                         eStatusType::INVALID_COMBINATION);

        EXPECT_EXCEPTION(op(nullptr, validCharTensor, validIntTensor, std::nullopt, std::nullopt, std::nullopt,
                            bc_validFloatTensor, eDeviceType::GPU),
                         eStatusType::INVALID_COMBINATION);
    }

    {
        // Test brightness/contrast bad shape
        TensorShape bc_mismatchBatchShape(TensorLayout(eTensorLayout::TENSOR_LAYOUT_N), {5});
        TensorShape bc_invalidLayoutShape(TensorLayout(eTensorLayout::TENSOR_LAYOUT_NC), {1, 1});
        Tensor bc_invalidBatchTensor(bc_mismatchBatchShape, DataType(eDataType::DATA_TYPE_F32), eDeviceType::GPU);
        Tensor bc_invalidLayoutTensor(bc_invalidLayoutShape, DataType(eDataType::DATA_TYPE_F32), eDeviceType::GPU);

        EXPECT_EXCEPTION(op(nullptr, validGPUTensor, validGPUTensor, bc_invalidBatchTensor, std::nullopt, std::nullopt,
                            std::nullopt, eDeviceType::GPU),
                         eStatusType::INVALID_COMBINATION);

        EXPECT_EXCEPTION(op(nullptr, validGPUTensor, validGPUTensor, std::nullopt, bc_invalidBatchTensor, std::nullopt,
                            std::nullopt, eDeviceType::GPU),
                         eStatusType::INVALID_COMBINATION);

        EXPECT_EXCEPTION(op(nullptr, validGPUTensor, validGPUTensor, std::nullopt, std::nullopt, bc_invalidLayoutTensor,
                            std::nullopt, eDeviceType::GPU),
                         eStatusType::INVALID_COMBINATION);

        EXPECT_EXCEPTION(op(nullptr, validGPUTensor, validGPUTensor, std::nullopt, std::nullopt, std::nullopt,
                            bc_invalidLayoutTensor, eDeviceType::GPU),
                         eStatusType::INVALID_COMBINATION);
    }
}

}  // namespace

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;
    TEST_CASES_BEGIN();

    // Test negative operator cases
    TEST_CASE(TestNegativeBrightnessContrast());

    // CPU correctness tests
    // 1 Channel
    TEST_CASE((TestCorrectness<uchar1, uchar1, float>(1, 360, 480, FMT_U8, FMT_U8, 1.5f, 1.2f, 0.1f, 100.0f,
                                                      eDataType::DATA_TYPE_F32, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<ushort1, ushort1, float>(1, 360, 480, FMT_U16, FMT_U16, 1.5f, 1.2f, 0.1f, 30000.0f,
                                                        eDataType::DATA_TYPE_F32, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<short1, short1, float>(1, 360, 480, FMT_S16, FMT_S16, 1.5f, 1.2f, 0.1f, 10000.0f,
                                                      eDataType::DATA_TYPE_F32, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<int1, int1, double>(1, 360, 480, FMT_S32, FMT_S32, 1.5, 1.2, 0.1, 1000000000.0,
                                                   eDataType::DATA_TYPE_F64, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<float1, float1, float>(1, 360, 480, FMT_F32, FMT_F32, 1.5f, 1.2f, 0.1f, 0.6f,
                                                      eDataType::DATA_TYPE_F32, eDeviceType::CPU)));

    TEST_CASE((TestCorrectness<uchar1, ushort1, float>(1, 480, 360, FMT_U8, FMT_U16, 1.5f, 1.2f, 0.1f, 100.0f,
                                                       eDataType::DATA_TYPE_F32, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<ushort1, uchar1, float>(1, 480, 360, FMT_U16, FMT_U8, 1.5f, 1.2f, 0.1f, 30000.0f,
                                                       eDataType::DATA_TYPE_F32, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<short1, uchar1, float>(1, 480, 360, FMT_S16, FMT_U8, 1.5f, 1.2f, 0.1f, 10000.0f,
                                                      eDataType::DATA_TYPE_F32, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<ushort1, short1, float>(1, 480, 360, FMT_U16, FMT_S16, 1.5f, 1.2f, 0.1f, 30000.0f,
                                                       eDataType::DATA_TYPE_F32, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<short1, ushort1, float>(1, 480, 360, FMT_S16, FMT_U16, 1.5f, 1.2f, 0.1f, 10000.0f,
                                                       eDataType::DATA_TYPE_F32, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<uchar1, int1, double>(1, 480, 360, FMT_U8, FMT_S32, 1.5, 1.2, 0.1, 100.0,
                                                     eDataType::DATA_TYPE_F64, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<int1, uchar1, double>(1, 480, 360, FMT_S32, FMT_U8, 1.5, 1.2, 0.1, 1000000000.0,
                                                     eDataType::DATA_TYPE_F64, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<uchar1, float1, float>(1, 480, 360, FMT_U8, FMT_F32, 1.5f, 1.2f, 0.1f, 100.0f,
                                                      eDataType::DATA_TYPE_F32, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<float1, uchar1, float>(1, 480, 360, FMT_F32, FMT_U8, 1.5f, 1.2f, 0.1f, 0.6f,
                                                      eDataType::DATA_TYPE_F32, eDeviceType::CPU)));

    TEST_CASE((TestCorrectnessDefaultsAndPer<float1, float1, float>(1, 480, 360, FMT_F32, FMT_F32, eDeviceType::CPU)));
    TEST_CASE((TestCorrectnessDefaultsAndPer<uchar1, int1, double>(1, 480, 360, FMT_U8, FMT_S32, eDeviceType::CPU)));

    // 3 Channels
    TEST_CASE((TestCorrectness<uchar3, uchar3, float>(1, 640, 480, FMT_RGB8, FMT_RGB8, 1.5f, 1.2f, 0.1f, 100.0f,
                                                      eDataType::DATA_TYPE_F32, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<ushort3, ushort3, float>(1, 640, 480, FMT_RGB16, FMT_RGB16, 1.5f, 1.2f, 0.1f, 30000.0f,
                                                        eDataType::DATA_TYPE_F32, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<short3, short3, float>(1, 640, 480, FMT_RGBs16, FMT_RGBs16, 1.5f, 1.2f, 0.1f, 10000.0f,
                                                      eDataType::DATA_TYPE_F32, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<int3, int3, double>(1, 640, 480, FMT_RGBs32, FMT_RGBs32, 1.5, 1.2, 0.1, 1000000000.0,
                                                   eDataType::DATA_TYPE_F64, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<float3, float3, float>(1, 640, 480, FMT_RGBf32, FMT_RGBf32, 1.5f, 1.2f, 0.1f, 0.6f,
                                                      eDataType::DATA_TYPE_F32, eDeviceType::CPU)));

    TEST_CASE((TestCorrectness<uchar3, ushort3, float>(1, 480, 640, FMT_RGB8, FMT_RGB16, 1.5f, 1.2f, 0.1f, 100.0f,
                                                       eDataType::DATA_TYPE_F32, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<ushort3, uchar3, float>(1, 480, 640, FMT_RGB16, FMT_RGB8, 1.5f, 1.2f, 0.1f, 30000.0f,
                                                       eDataType::DATA_TYPE_F32, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<short3, uchar3, float>(1, 480, 640, FMT_RGBs16, FMT_RGB8, 1.5f, 1.2f, 0.1f, 10000.0f,
                                                      eDataType::DATA_TYPE_F32, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<ushort3, short3, float>(1, 480, 640, FMT_RGB16, FMT_RGBs16, 1.5f, 1.2f, 0.1f, 30000.0f,
                                                       eDataType::DATA_TYPE_F32, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<short3, ushort3, float>(1, 480, 640, FMT_RGBs16, FMT_RGB16, 1.5f, 1.2f, 0.1f, 10000.0f,
                                                       eDataType::DATA_TYPE_F32, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<uchar3, int3, double>(1, 480, 640, FMT_RGB8, FMT_RGBs32, 1.5, 1.2, 0.1, 100.0,
                                                     eDataType::DATA_TYPE_F64, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<int3, uchar3, double>(1, 480, 640, FMT_RGBs32, FMT_RGB8, 1.5, 1.2, 0.1, 1000000000.0,
                                                     eDataType::DATA_TYPE_F64, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<uchar3, float3, float>(1, 480, 640, FMT_RGB8, FMT_RGBf32, 1.5f, 1.2f, 0.1f, 100.0f,
                                                      eDataType::DATA_TYPE_F32, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<float3, uchar3, float>(1, 480, 640, FMT_RGBf32, FMT_RGB8, 1.5f, 1.2f, 0.1f, 0.6f,
                                                      eDataType::DATA_TYPE_F32, eDeviceType::CPU)));

    TEST_CASE(
        (TestCorrectnessDefaultsAndPer<float3, float3, float>(1, 480, 640, FMT_RGBf32, FMT_RGBf32, eDeviceType::CPU)));
    TEST_CASE(
        (TestCorrectnessDefaultsAndPer<uchar3, int3, double>(1, 480, 640, FMT_RGB8, FMT_RGBs32, eDeviceType::CPU)));

    // 4 Channels
    TEST_CASE((TestCorrectness<uchar4, uchar4, float>(1, 360, 480, FMT_RGBA8, FMT_RGBA8, 1.5f, 1.2f, 0.1f, 100.0f,
                                                      eDataType::DATA_TYPE_F32, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<ushort4, ushort4, float>(1, 360, 480, FMT_RGBA16, FMT_RGBA16, 1.5f, 1.2f, 0.1f, 30000.0f,
                                                        eDataType::DATA_TYPE_F32, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<short4, short4, float>(1, 360, 480, FMT_RGBAs16, FMT_RGBAs16, 1.5f, 1.2f, 0.1f, 10000.0f,
                                                      eDataType::DATA_TYPE_F32, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<int4, int4, double>(1, 360, 480, FMT_RGBAs32, FMT_RGBAs32, 1.5, 1.2, 0.1, 1000000000.0,
                                                   eDataType::DATA_TYPE_F64, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<float4, float4, float>(1, 360, 480, FMT_RGBAf32, FMT_RGBAf32, 1.5f, 1.2f, 0.1f, 0.6f,
                                                      eDataType::DATA_TYPE_F32, eDeviceType::CPU)));

    TEST_CASE((TestCorrectness<uchar4, ushort4, float>(1, 480, 360, FMT_RGBA8, FMT_RGBA16, 1.5f, 1.2f, 0.1f, 100.0f,
                                                       eDataType::DATA_TYPE_F32, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<ushort4, uchar4, float>(1, 480, 360, FMT_RGBA16, FMT_RGBA8, 1.5f, 1.2f, 0.1f, 30000.0f,
                                                       eDataType::DATA_TYPE_F32, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<short4, uchar4, float>(1, 480, 360, FMT_RGBAs16, FMT_RGBA8, 1.5f, 1.2f, 0.1f, 10000.0f,
                                                      eDataType::DATA_TYPE_F32, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<ushort4, short4, float>(1, 480, 360, FMT_RGBA16, FMT_RGBAs16, 1.5f, 1.2f, 0.1f, 30000.0f,
                                                       eDataType::DATA_TYPE_F32, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<short4, ushort4, float>(1, 480, 360, FMT_RGBAs16, FMT_RGBA16, 1.5f, 1.2f, 0.1f, 10000.0f,
                                                       eDataType::DATA_TYPE_F32, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<uchar4, int4, double>(1, 480, 360, FMT_RGBA8, FMT_RGBAs32, 1.5, 1.2, 0.1, 100.0,
                                                     eDataType::DATA_TYPE_F64, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<int4, uchar4, double>(1, 480, 360, FMT_RGBAs32, FMT_RGBA8, 1.5, 1.2, 0.1, 1000000000.0,
                                                     eDataType::DATA_TYPE_F64, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<uchar4, float4, float>(1, 480, 360, FMT_RGBA8, FMT_RGBAf32, 1.5f, 1.2f, 0.1f, 100.0f,
                                                      eDataType::DATA_TYPE_F32, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<float4, uchar4, float>(1, 480, 360, FMT_RGBAf32, FMT_RGBA8, 1.5f, 1.2f, 0.1f, 0.6f,
                                                      eDataType::DATA_TYPE_F32, eDeviceType::CPU)));

    TEST_CASE((
        TestCorrectnessDefaultsAndPer<float4, float4, float>(1, 480, 360, FMT_RGBAf32, FMT_RGBAf32, eDeviceType::CPU)));
    TEST_CASE(
        (TestCorrectnessDefaultsAndPer<uchar4, int4, double>(1, 480, 360, FMT_RGBA8, FMT_RGBAs32, eDeviceType::CPU)));

    // GPU Correctness Tests
    // 1 Channel
    TEST_CASE((TestCorrectness<uchar1, uchar1, float>(1, 360, 480, FMT_U8, FMT_U8, 1.5f, 1.2f, 0.1f, 100.0f,
                                                      eDataType::DATA_TYPE_F32, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<ushort1, ushort1, float>(1, 360, 480, FMT_U16, FMT_U16, 1.5f, 1.2f, 0.1f, 30000.0f,
                                                        eDataType::DATA_TYPE_F32, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<short1, short1, float>(1, 360, 480, FMT_S16, FMT_S16, 1.5f, 1.2f, 0.1f, 10000.0f,
                                                      eDataType::DATA_TYPE_F32, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<int1, int1, double>(1, 360, 480, FMT_S32, FMT_S32, 1.5, 1.2, 0.1, 1000000000.0,
                                                   eDataType::DATA_TYPE_F64, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<float1, float1, float>(1, 360, 480, FMT_F32, FMT_F32, 1.5f, 1.2f, 0.1f, 0.6f,
                                                      eDataType::DATA_TYPE_F32, eDeviceType::GPU)));

    TEST_CASE((TestCorrectness<uchar1, ushort1, float>(1, 480, 360, FMT_U8, FMT_U16, 1.5f, 1.2f, 0.1f, 100.0f,
                                                       eDataType::DATA_TYPE_F32, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<ushort1, uchar1, float>(1, 480, 360, FMT_U16, FMT_U8, 1.5f, 1.2f, 0.1f, 30000.0f,
                                                       eDataType::DATA_TYPE_F32, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<short1, uchar1, float>(1, 480, 360, FMT_S16, FMT_U8, 1.5f, 1.2f, 0.1f, 10000.0f,
                                                      eDataType::DATA_TYPE_F32, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<ushort1, short1, float>(1, 480, 360, FMT_U16, FMT_S16, 1.5f, 1.2f, 0.1f, 30000.0f,
                                                       eDataType::DATA_TYPE_F32, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<short1, ushort1, float>(1, 480, 360, FMT_S16, FMT_U16, 1.5f, 1.2f, 0.1f, 10000.0f,
                                                       eDataType::DATA_TYPE_F32, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<uchar1, int1, double>(1, 480, 360, FMT_U8, FMT_S32, 1.5, 1.2, 0.1, 100.0,
                                                     eDataType::DATA_TYPE_F64, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<int1, uchar1, double>(1, 480, 360, FMT_S32, FMT_U8, 1.5, 1.2, 0.1, 1000000000.0,
                                                     eDataType::DATA_TYPE_F64, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<uchar1, float1, float>(1, 480, 360, FMT_U8, FMT_F32, 1.5f, 1.2f, 0.1f, 100.0f,
                                                      eDataType::DATA_TYPE_F32, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<float1, uchar1, float>(1, 480, 360, FMT_F32, FMT_U8, 1.5f, 1.2f, 0.1f, 0.6f,
                                                      eDataType::DATA_TYPE_F32, eDeviceType::GPU)));

    TEST_CASE((TestCorrectnessDefaultsAndPer<float1, float1, float>(1, 480, 360, FMT_F32, FMT_F32, eDeviceType::GPU)));
    TEST_CASE((TestCorrectnessDefaultsAndPer<uchar1, int1, double>(1, 480, 360, FMT_U8, FMT_S32, eDeviceType::GPU)));

    // 3 Channels
    TEST_CASE((TestCorrectness<uchar3, uchar3, float>(1, 360, 480, FMT_RGB8, FMT_RGB8, 1.5f, 1.2f, 0.1f, 100.0f,
                                                      eDataType::DATA_TYPE_F32, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<ushort3, ushort3, float>(1, 360, 480, FMT_RGB16, FMT_RGB16, 1.5f, 1.2f, 0.1f, 30000.0f,
                                                        eDataType::DATA_TYPE_F32, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<short3, short3, float>(1, 360, 480, FMT_RGBs16, FMT_RGBs16, 1.5f, 1.2f, 0.1f, 10000.0f,
                                                      eDataType::DATA_TYPE_F32, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<int3, int3, double>(1, 360, 480, FMT_RGBs32, FMT_RGBs32, 1.5, 1.2, 0.1, 1000000000.0,
                                                   eDataType::DATA_TYPE_F64, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<float3, float3, float>(1, 360, 480, FMT_RGBf32, FMT_RGBf32, 1.5f, 1.2f, 0.1f, 0.6f,
                                                      eDataType::DATA_TYPE_F32, eDeviceType::GPU)));

    TEST_CASE((TestCorrectness<uchar3, ushort3, float>(1, 480, 360, FMT_RGB8, FMT_RGB16, 1.5f, 1.2f, 0.1f, 100.0f,
                                                       eDataType::DATA_TYPE_F32, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<ushort3, uchar3, float>(1, 480, 360, FMT_RGB16, FMT_RGB8, 1.5f, 1.2f, 0.1f, 30000.0f,
                                                       eDataType::DATA_TYPE_F32, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<short3, uchar3, float>(1, 480, 360, FMT_RGBs16, FMT_RGB8, 1.5f, 1.2f, 0.1f, 10000.0f,
                                                      eDataType::DATA_TYPE_F32, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<ushort3, short3, float>(1, 480, 360, FMT_RGB16, FMT_RGBs16, 1.5f, 1.2f, 0.1f, 30000.0f,
                                                       eDataType::DATA_TYPE_F32, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<short3, ushort3, float>(1, 480, 360, FMT_RGBs16, FMT_RGB16, 1.5f, 1.2f, 0.1f, 10000.0f,
                                                       eDataType::DATA_TYPE_F32, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<uchar3, int3, double>(1, 480, 360, FMT_RGB8, FMT_RGBs32, 1.5, 1.2, 0.1, 100.0,
                                                     eDataType::DATA_TYPE_F64, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<int3, uchar3, double>(1, 480, 360, FMT_RGBs32, FMT_RGB8, 1.5, 1.2, 0.1, 1000000000.0,
                                                     eDataType::DATA_TYPE_F64, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<uchar3, float3, float>(1, 480, 360, FMT_RGB8, FMT_RGBf32, 1.5f, 1.2f, 0.1f, 100.0f,
                                                      eDataType::DATA_TYPE_F32, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<float3, uchar3, float>(1, 480, 360, FMT_RGBf32, FMT_RGB8, 1.5f, 1.2f, 0.1f, 0.6f,
                                                      eDataType::DATA_TYPE_F32, eDeviceType::GPU)));

    TEST_CASE(
        (TestCorrectnessDefaultsAndPer<float3, float3, float>(1, 480, 360, FMT_RGBf32, FMT_RGBf32, eDeviceType::GPU)));
    TEST_CASE(
        (TestCorrectnessDefaultsAndPer<uchar3, int3, double>(1, 480, 360, FMT_RGB8, FMT_RGBs32, eDeviceType::GPU)));

    // 4 Channels
    TEST_CASE((TestCorrectness<uchar4, uchar4, float>(1, 1920, 1080, FMT_RGBA8, FMT_RGBA8, 1.5f, 1.2f, 0.1f, 100.0f,
                                                      eDataType::DATA_TYPE_F32, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<ushort4, ushort4, float>(1, 1920, 1080, FMT_RGBA16, FMT_RGBA16, 1.5f, 1.2f, 0.1f,
                                                        30000.0f, eDataType::DATA_TYPE_F32, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<short4, short4, float>(1, 1920, 1080, FMT_RGBAs16, FMT_RGBAs16, 1.5f, 1.2f, 0.1f,
                                                      10000.0f, eDataType::DATA_TYPE_F32, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<int4, int4, double>(1, 1920, 1080, FMT_RGBAs32, FMT_RGBAs32, 1.5, 1.2, 0.1, 1000000000.0,
                                                   eDataType::DATA_TYPE_F64, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<float4, float4, float>(1, 1920, 1080, FMT_RGBAf32, FMT_RGBAf32, 1.5f, 1.2f, 0.1f, 0.6f,
                                                      eDataType::DATA_TYPE_F32, eDeviceType::GPU)));

    TEST_CASE((TestCorrectness<uchar4, ushort4, float>(1, 1080, 1920, FMT_RGBA8, FMT_RGBA16, 1.5f, 1.2f, 0.1f, 100.0f,
                                                       eDataType::DATA_TYPE_F32, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<ushort4, uchar4, float>(1, 1080, 1920, FMT_RGBA16, FMT_RGBA8, 1.5f, 1.2f, 0.1f, 30000.0f,
                                                       eDataType::DATA_TYPE_F32, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<short4, uchar4, float>(1, 1080, 1920, FMT_RGBAs16, FMT_RGBA8, 1.5f, 1.2f, 0.1f, 10000.0f,
                                                      eDataType::DATA_TYPE_F32, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<ushort4, short4, float>(1, 1080, 1920, FMT_RGBA16, FMT_RGBAs16, 1.5f, 1.2f, 0.1f,
                                                       30000.0f, eDataType::DATA_TYPE_F32, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<short4, ushort4, float>(1, 1080, 1920, FMT_RGBAs16, FMT_RGBA16, 1.5f, 1.2f, 0.1f,
                                                       10000.0f, eDataType::DATA_TYPE_F32, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<uchar4, int4, double>(1, 1080, 1920, FMT_RGBA8, FMT_RGBAs32, 1.5, 1.2, 0.1, 100.0,
                                                     eDataType::DATA_TYPE_F64, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<int4, uchar4, double>(1, 1080, 1920, FMT_RGBAs32, FMT_RGBA8, 1.5, 1.2, 0.1, 1000000000.0,
                                                     eDataType::DATA_TYPE_F64, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<uchar4, float4, float>(1, 1080, 1920, FMT_RGBA8, FMT_RGBAf32, 1.5f, 1.2f, 0.1f, 100.0f,
                                                      eDataType::DATA_TYPE_F32, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<float4, uchar4, float>(1, 1080, 1920, FMT_RGBAf32, FMT_RGBA8, 1.5f, 1.2f, 0.1f, 0.6f,
                                                      eDataType::DATA_TYPE_F32, eDeviceType::GPU)));

    TEST_CASE((TestCorrectnessDefaultsAndPer<float4, float4, float>(1, 3840, 2160, FMT_RGBAf32, FMT_RGBAf32,
                                                                    eDeviceType::GPU)));
    TEST_CASE(
        (TestCorrectnessDefaultsAndPer<uchar4, int4, double>(1, 2160, 3840, FMT_RGBA8, FMT_RGBAs32, eDeviceType::GPU)));

    TEST_CASES_END();
}