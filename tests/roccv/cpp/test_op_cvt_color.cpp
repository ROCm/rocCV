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

// #include <hip/hip_runtime.h>

// #include <iostream>

// #include "common/array_wrapper.hpp"
// #include "common/conversion_helpers.hpp"
// #include "common/math_vector.hpp"
// #include "common/strided_data_wrap.hpp"
// #include "common/validation_helpers.hpp"
// #include "core/tensor.hpp"
// #include "core/wrappers/image_wrapper.hpp"
// #include "kernels/device/cvt_color_device.hpp"
// #include "kernels/host/cvt_color_host.hpp"
// #include "op_cvt_color.hpp"
#include "test_helpers.hpp"

// using namespace roccv;
// using namespace roccv::tests;

// namespace fs = std::filesystem;

// // Keep all non-entrypoint functions in an anonymous namespace to prevent redefinition errors across translation
// units. namespace {

// /**
//  * @brief Verified golden C++ model for the Regular Color Convert operation.
//  *
//  * @tparam T Vectorized datatype of the image's pixels.
//  * @tparam BT Base type of the image's data.
//  * @param[in] input An input vector containing image data.
//  * @param[in] output An output vector containing result image data, to be filled.
//  * @param[in] conversionCode the conversion code, from enum eColorConversionCode.
//  * @return Vector containing the results of the operation.
//  */
// template <typename T, typename BT = detail::BaseType<T>>
// std::vector<BT> GoldenGammaColorCvt(const Tensor &input, const Tensor &output, int batch, int width, int height,
//                                     const eColorConversionCode conversionCode) {
//     ImageWrapper<T> inW(input);
//     ImageWrapper<T> outW(output);

//     using PairNdxDlta = std::tuple<int, float>;
//     static const std::unordered_map<eColorConversionCode, PairNdxDlta> ndx_dlt = {
//         {COLOR_RGB2YUV, {0, 128.0f}}, {COLOR_BGR2YUV, {2, 128.0f}}, {COLOR_YUV2RGB, {0, 128.0f}},
//         {COLOR_YUV2BGR, {2, 128.0f}}, {COLOR_RGB2BGR, {0, 2.0f}},   {COLOR_BGR2RGB, {2, 0.0f}},
//         {COLOR_RGB2GRAY, {0, 0.0f}},  {COLOR_BGR2GRAY, {2, 0.0f}}};
//     auto [orderIdx, delta] = ndx_dlt.at(conversionCode);

//     switch (conversionCode) {
//         case COLOR_RGB2YUV:
//         case COLOR_BGR2YUV: {
//             using namespace roccv;
//             using namespace roccv::detail;

//             // Working type will always be a 3-channel floating point since input/output is always RGB/BGR
//             using work_type_t = MakeType<float, 3>;

//             for (int z_idx = 0; z_idx < output.batches(); z_idx++) {
//                 for (int y_idx = 0; y_idx < output.height(); y_idx++) {
//                     for (int x_idx = 0; x_idx < output.width(); x_idx++) {
//                         T val = input.at(z_idx, y_idx, x_idx, 0);
//                         work_type_t valF = StaticCast<work_type_t>(val);

//                         float r = GetElement(valF, orderIdx);
//                         float g = GetElement(valF, 1);
//                         float b = GetElement(valF, orderIdx ^ 2);

//                         float y = r * 0.299f + g * 0.587f + b * 0.114f;
//                         float cr = (r - y) * 0.877f + delta;
//                         float cb = (b - y) * 0.492f + delta;

//                         T out = make_uchar3(RoundImplementationsToYUV<float>(y),
//                         RoundImplementationsToYUV<float>(cb),
//                                             RoundImplementationsToYUV<float>(cr));

//                         output.at(z_idx, y_idx, x_idx, 0) = out;
//                     }
//                 }
//             }
//             break;
//         }

//         case COLOR_YUV2RGB:
//         case COLOR_YUV2BGR: {
//             using namespace roccv;
//             using namespace roccv::detail;
//             using work_type_t = MakeType<float, 3>;

//             for (int z_idx = 0; z_idx < output.batches(); z_idx++) {
//                 for (int y_idx = 0; y_idx < output.height(); y_idx++) {
//                     for (int x_idx = 0; x_idx < output.width(); x_idx++) {
//                         T val = input.at(z_idx, y_idx, x_idx, 0);
//                         work_type_t valF = StaticCast<work_type_t>(val);

//                         // Y = valF.x
//                         // Cr = valF.y
//                         // Cb = valF.z

//                         // Convert from YUV to RGB
//                         work_type_t rgb =
//                             make_float3(RoundImplementationsFromYUV<float>(valF.x + (valF.z - delta) * 1.140f),  // R
//                                         RoundImplementationsFromYUV<float>(valF.x + (valF.y - delta) * -0.395f +
//                                                                            (valF.z - delta) * -0.581f),           //
//                                                                            G
//                                         RoundImplementationsFromYUV<float>(valF.x + (valF.y - delta) * 2.032f));  //
//                                         B

//                         // Reorder to proper layout (RGB or BGR, depending on orderIdx)
//                         work_type_t out;
//                         out.x = GetElement(rgb, orderIdx);
//                         out.y = GetElement(rgb, 1);
//                         out.z = GetElement(rgb, orderIdx ^ 2);

//                         // Saturate cast to type T (this clamps to proper ranges)
//                         output.at(z_idx, y_idx, x_idx, 0) = SaturateCast<T>(out);
//                     }
//                 }
//             }
//             break;
//         }

//         case COLOR_RGB2BGR:
//         case COLOR_BGR2RGB: {
//             using namespace roccv::detail;

//             for (int z_idx = 0; z_idx < output.batches(); z_idx++) {
//                 for (int y_idx = 0; y_idx < output.height(); y_idx++) {
//                     for (int x_idx = 0; x_idx < output.width(); x_idx++) {
//                         T inVal = input.at(z_idx, y_idx, x_idx, 0);

//                         // Convert inVal into RGB format, depends on orderIdxInput
//                         T inValRGB{GetElement(inVal, orderIdxInput), GetElement(inVal, 1),
//                                    GetElement(inVal, orderIdxInput ^ 2)};

//                         // Convert from inValRGB to requested final format, depends on orderIdxOutput
//                         T outVal{GetElement(inValRGB, orderIdxOutput), GetElement(inValRGB, 1),
//                                  GetElement(inValRGB, orderIdxOutput ^ 2)};

//                         output.at(z_idx, y_idx, x_idx, 0) = outVal;
//                     }
//                 }
//             }
//             break;
//         }

//         case COLOR_RGB2GRAY:
//         case COLOR_BGR2GRAY: {
//             using namespace roccv::detail;
//             using work_type_t = MakeType<float, 3>;

//             for (int z_idx = 0; z_idx < output.batches(); z_idx++) {
//                 for (int y_idx = 0; y_idx < output.height(); y_idx++) {
//                     for (int x_idx = 0; x_idx < output.width(); x_idx++) {
//                         T inVal = input.at(z_idx, y_idx, x_idx, 0);
//                         work_type_t inValF = StaticCast<work_type_t>(inVal);

//                         // Get RGB elements (input order defined by orderIdxInput)
//                         float r = GetElement(inValF, orderIdxInput);
//                         float g = GetElement(inValF, 1);
//                         float b = GetElement(inValF, orderIdxInput ^ 2);

//                         // Calculate luminance
//                         float y = RoundImplementationsToYUV<float>(r * 0.299f + g * 0.587f + b * 0.114f);

//                         output.at(z_idx, y_idx, x_idx, 0) = SaturateCast<uchar1>(y);
//                     }
//                 }
//             }
//             break;
//         }
//         default:
//             break;
//     }
//     std::vector<BT> result;
//     result.resize(output.shape().size());
//     CopyTensorIntoVector(result, output);
//     return result;
// }

// /**
//  * @brief Tests correctness of the CVT Color operator, comparing it against a generated golden result.
//  *
//  * @tparam T Underlying datatype of the image's pixels.
//  * @tparam BT Base type of the image data.
//  * @param[in] width The width of each image in the batch.
//  * @param[in] height The height of each image in the batch.
//  * @param[in] format The image format.
//  * @param[in] conversionCode the conversion code, from enum eColorConversionCode.
//  * @param[in] device The device this correctness test should be run on.
//  */
// template <typename T, typename BT = detail::BaseType<T>>
// void TestCorrectness(int batch, int width, int height, ImageFormat format, eColorConversionCode conversionCode,
//                      eDeviceType device) {
//     TensorShape shape_clr(TensorLayout(eTensorLayout::TENSOR_LAYOUT_NHWC), {batch, height, width, 3});
//     TensorShape shape_gry(TensorLayout(eTensorLayout::TENSOR_LAYOUT_NHWC), {batch, height, width, 1});
//     DataType dtype(eDataType::DATA_TYPE_U8);
//     Tensor input_clr(shape_clr, dtype, device);
//     Tensor output_clr(shape_clr, input_clr.dtype(), input_clr.device());
//     Tensor output_gry(shape_gry, input_clr.dtype(), input_clr.device());
//     size_t image_size = input_clr.shape().size() * input_clr.dtype().size();

//     // Create a vector and fill it with random data.
//     std::vector<BT> inputData(image_size);
//     FillVector(inputData);

//     // Copy generated input data into input tensor
//     CopyVectorIntoTensor(input_clr, inputData);

//     // Calculate golden output reference
//     std::vector<BT> ref;
//     CvtColor op;

//     hipStream_t stream = static_cast<hipStream_t>(nullptr);
//     if (device == eDeviceType::GPU) HIP_VALIDATE_NO_ERRORS(hipStreamCreate(&stream));

//     size_t output_image_size = 0;
//     switch (conversionCode) {
//         case eColorConversionCode::COLOR_RGB2GRAY:
//         case eColorConversionCode::COLOR_BGR2GRAY: {
//             ref = GoldenGammaColorCvt<T>(input_clr, output_gry, batch, width, height, conversionCode);
//             op(stream, input_clr, output_gry, conversionCode, device);
//             output_image_size = output_gry.shape().size() * output_clr.dtype().size();
//         } break;
//         default:  // COLOR = any other than GRAY
//         {
//             ref = GoldenGammaColorCvt<T>(input_clr, output_clr, batch, width, height, conversionCode);
//             op(stream, input_clr, output_clr, conversionCode, device);
//             output_image_size = output_clr.shape().size() * output_clr.dtype().size();
//         } break;
//     }
//     // results vector
//     std::vector<BT> result;
//     result.resize(output_image_size);
//     // Copy data from output tensor into a host allocated results vector
//     switch (conversionCode) {
//         case eColorConversionCode::COLOR_RGB2GRAY:
//         case eColorConversionCode::COLOR_BGR2GRAY:
//             CopyTensorIntoVector(result, output_gry);
//             break;
//         default:
//             CopyTensorIntoVector(result, output_clr);
//             break;
//     }
//     // Compare data in actual output versus the generated golden reference image
//     CompareVectorsNear(result, ref);
// }
// }  // namespace

using namespace roccv;

eTestStatusType test_op_cvt_color(int argc, char **argv) {
    TEST_CASES_BEGIN();

    // // GPU correctness tests
    // TEST_CASE(TestCorrectness<uchar3>(1, 480, 360, FMT_RGB8, COLOR_RGB2YUV, eDeviceType::GPU));
    // TEST_CASE(TestCorrectness<uchar3>(2, 480, 120, FMT_RGB8, COLOR_BGR2YUV, eDeviceType::GPU));
    // TEST_CASE(TestCorrectness<uchar3>(5, 360, 360, FMT_RGB8, COLOR_YUV2RGB, eDeviceType::GPU));
    // TEST_CASE(TestCorrectness<uchar3>(3, 134, 360, FMT_RGB8, COLOR_YUV2BGR, eDeviceType::GPU));
    // TEST_CASE(TestCorrectness<uchar3>(7, 134, 360, FMT_RGB8, COLOR_RGB2BGR, eDeviceType::GPU));
    // TEST_CASE(TestCorrectness<uchar3>(6, 134, 360, FMT_RGB8, COLOR_BGR2RGB, eDeviceType::GPU));
    // TEST_CASE(TestCorrectness<uchar3>(9, 480, 120, FMT_RGB8, COLOR_RGB2GRAY, eDeviceType::GPU));
    // TEST_CASE(TestCorrectness<uchar3>(8, 134, 360, FMT_RGB8, COLOR_BGR2GRAY, eDeviceType::GPU));

    // // CPU correctness tests
    // TEST_CASE(TestCorrectness<uchar3>(8, 480, 360, FMT_RGB8, COLOR_RGB2YUV, eDeviceType::CPU));
    // TEST_CASE(TestCorrectness<uchar3>(2, 480, 120, FMT_RGB8, COLOR_BGR2YUV, eDeviceType::CPU));
    // TEST_CASE(TestCorrectness<uchar3>(5, 360, 360, FMT_RGB8, COLOR_YUV2RGB, eDeviceType::CPU));
    // TEST_CASE(TestCorrectness<uchar3>(3, 134, 360, FMT_RGB8, COLOR_YUV2BGR, eDeviceType::CPU));
    // TEST_CASE(TestCorrectness<uchar3>(7, 134, 360, FMT_RGB8, COLOR_RGB2BGR, eDeviceType::CPU));
    // TEST_CASE(TestCorrectness<uchar3>(6, 134, 360, FMT_RGB8, COLOR_BGR2RGB, eDeviceType::CPU));
    // TEST_CASE(TestCorrectness<uchar3>(9, 480, 120, FMT_RGB8, COLOR_RGB2GRAY, eDeviceType::CPU));
    // TEST_CASE(TestCorrectness<uchar3>(3, 134, 360, FMT_RGB8, COLOR_BGR2GRAY, eDeviceType::CPU));

    TEST_CASES_END();
}
