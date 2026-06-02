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
#include <op_convert_to.hpp>

#include "test_helpers.hpp"

using namespace roccv;
using namespace roccv::tests;

// Keep all non-entrypoint functions in an anonymous namespace to prevent redefinition errors across translation units.
namespace {

/**
 * @brief Verified golden C++ model for the ConvertTo operation.
 *
 * @tparam SRC_DT Vectorized datatype of the input image's pixels.
 * @tparam DEST_DT Vectorized datatype of the output image's pixels.
 * @tparam BT_SRC Base type of the input image's data.
 * @tparam BT_DEST Base type of the output image's data.
 * @param[in] input An input vector containing image data.
 * @param[in] batchSize The number of images in the batch.
 * @param[in] width Image width.
 * @param[in] height Image height.
 * @param[in] channels Number of channels in the image.
 * @param[in] alpha Scalar for output data.
 * @param[in] beta Offset for the data.
 * @return Vector containing the results of the operation.
 */
template <typename SRC_DT, typename DEST_DT, typename BT_SRC = detail::BaseType<SRC_DT>,
          typename BT_DEST = detail::BaseType<DEST_DT>>
std::vector<BT_DEST> GoldenConvertTo(std::vector<BT_SRC>& input, int32_t batchSize, int32_t width, int32_t height,
                                     double alpha, double beta) {
    // Create an output vector the same size as the input vector
    std::vector<BT_DEST> output(input.size());

    // Wrap input/output vectors for simplified data access
    ImageWrapper<SRC_DT> src(input, batchSize, width, height);
    ImageWrapper<DEST_DT> dst(output, batchSize, width, height);

    using AB_DT = decltype(float() * BT_SRC() * BT_DEST());
    using work_type = detail::MakeType<AB_DT, detail::NumElements<DEST_DT>>;

    AB_DT alpha_dt = detail::SaturateCast<AB_DT>(alpha);
    AB_DT beta_dt = detail::SaturateCast<AB_DT>(beta);

    for (int b = 0; b < batchSize; ++b) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                work_type src_val = detail::StaticCast<work_type>(src.at(b, y, x, 0));
                work_type result = alpha_dt * src_val + beta_dt;
                dst.at(b, y, x, 0) = detail::SaturateCast<DEST_DT>(result);
            }
        }
    }
    return output;
}

template <typename SRC_DT, typename DEST_DT, typename BT_SRC = detail::BaseType<SRC_DT>,
          typename BT_DEST = detail::BaseType<DEST_DT>>
void TestCorrectness(int batchSize, int width, int height, ImageFormat inFormat, ImageFormat outFormat, double alpha,
                     double beta, eDeviceType device) {
    // Create input and output tensor based on test parameters
    Tensor input(batchSize, {width, height}, inFormat, device);
    Tensor output(batchSize, {width, height}, outFormat, device);

    // Create a vector and fill it with random data.
    std::vector<BT_SRC> inputData(input.shape().size());
    FillVector(inputData);

    // Copy generated input data into input tensor
    CopyVectorIntoTensor(input, inputData);

    // Calculate golden output reference
    std::vector<BT_DEST> ref = GoldenConvertTo<SRC_DT, DEST_DT>(inputData, batchSize, width, height, alpha, beta);

    // Run roccv::Convert To operator to obtain actual results
    hipStream_t stream;
    HIP_VALIDATE_NO_ERRORS(hipStreamCreate(&stream));

    ConvertTo op;
    op(stream, input, output, alpha, beta, device);
    HIP_VALIDATE_NO_ERRORS(hipStreamSynchronize(stream));
    HIP_VALIDATE_NO_ERRORS(hipStreamDestroy(stream));

    // Copy data from output tensor into a host allocated vector
    std::vector<BT_DEST> result(output.shape().size());
    CopyTensorIntoVector(result, output);

    // Compare data in actual output versus the generated golden reference image
    // Using 1.0E-4 as the error threshold to account for FMA/non-FMA float divergence between CPU and GPU.
    CompareVectorsNear(result, ref, 1.0E-4);
}

}  // namespace

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;
    TEST_CASES_BEGIN();

    // CPU correctness tests
    // 1 Channel
    TEST_CASE((TestCorrectness<uchar1, uchar1>(1, 480, 360, FMT_U8, FMT_U8, 1.2, 10.2, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<char1, char1>(1, 480, 360, FMT_S8, FMT_S8, 1.2, 10.2, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<uchar1, char1>(1, 480, 360, FMT_U8, FMT_S8, 1.2, 10.2, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<char1, uchar1>(1, 480, 360, FMT_S8, FMT_U8, 1.2, 10.2, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<uchar1, ushort1>(1, 480, 360, FMT_U8, FMT_U16, 1.2, 10.2, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<char1, short1>(1, 480, 360, FMT_S8, FMT_S16, 1.2, 10.2, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<ushort1, uchar1>(1, 480, 360, FMT_U16, FMT_U8, 1.2, 10.2, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<short1, char1>(1, 480, 360, FMT_S16, FMT_S8, 1.2, 10.2, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<ushort1, char1>(1, 480, 360, FMT_U16, FMT_S8, 1.2, 10.2, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<short1, uchar1>(1, 480, 360, FMT_S16, FMT_U8, 1.2, 10.2, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<ushort1, ushort1>(1, 480, 360, FMT_U16, FMT_U16, 1.2, 10.2, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<short1, short1>(1, 480, 360, FMT_S16, FMT_S16, 1.2, 10.2, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<ushort1, short1>(1, 480, 360, FMT_U16, FMT_S16, 1.2, 10.2, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<short1, ushort1>(1, 480, 360, FMT_S16, FMT_U16, 1.2, 10.2, eDeviceType::CPU)));

    TEST_CASE((TestCorrectness<uchar1, int1>(1, 480, 360, FMT_U8, FMT_S32, 1.2, 10.2, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<char1, int1>(1, 480, 360, FMT_S8, FMT_S32, 1.2, 10.2, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<int1, uchar1>(1, 480, 360, FMT_S32, FMT_U8, 1.2, 10.2, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<int1, char1>(1, 480, 360, FMT_S32, FMT_S8, 1.2, 10.2, eDeviceType::CPU)));

    TEST_CASE((TestCorrectness<uchar1, float1>(1, 480, 360, FMT_U8, FMT_F32, 1.2, 10.2, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<char1, float1>(1, 480, 360, FMT_S8, FMT_F32, 1.2, 10.2, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<float1, uchar1>(1, 480, 360, FMT_F32, FMT_U8, 1.2, 10.2, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<float1, char1>(1, 480, 360, FMT_F32, FMT_S8, 1.2, 10.2, eDeviceType::CPU)));

    // 3 Channels
    TEST_CASE((TestCorrectness<uchar3, uchar3>(1, 480, 360, FMT_RGB8, FMT_RGB8, 1.2, 10.2, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<char3, char3>(1, 480, 360, FMT_RGBs8, FMT_RGBs8, 1.2, 10.2, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<uchar3, char3>(1, 480, 360, FMT_RGB8, FMT_RGBs8, 1.2, 10.2, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<char3, uchar3>(1, 480, 360, FMT_RGBs8, FMT_RGB8, 1.2, 10.2, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<uchar3, ushort3>(1, 480, 360, FMT_RGB8, FMT_RGB16, 1.2, 10.2, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<char3, short3>(1, 480, 360, FMT_RGBs8, FMT_RGBs16, 1.2, 10.2, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<ushort3, uchar3>(1, 480, 360, FMT_RGB16, FMT_RGB8, 1.2, 10.2, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<short3, char3>(1, 480, 360, FMT_RGBs16, FMT_RGBs8, 1.2, 10.2, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<ushort3, char3>(1, 480, 360, FMT_RGB16, FMT_RGBs8, 1.2, 10.2, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<short3, uchar3>(1, 480, 360, FMT_RGBs16, FMT_RGB8, 1.2, 10.2, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<ushort3, ushort3>(1, 480, 360, FMT_RGB16, FMT_RGB16, 1.2, 10.2, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<short3, short3>(1, 480, 360, FMT_RGBs16, FMT_RGBs16, 1.2, 10.2, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<ushort3, short3>(1, 480, 360, FMT_RGB16, FMT_RGBs16, 1.2, 10.2, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<short3, ushort3>(1, 480, 360, FMT_RGBs16, FMT_RGB16, 1.2, 10.2, eDeviceType::CPU)));

    TEST_CASE((TestCorrectness<uchar3, int3>(1, 480, 360, FMT_RGB8, FMT_RGBs32, 1.2, 10.2, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<char3, int3>(1, 480, 360, FMT_RGBs8, FMT_RGBs32, 1.2, 10.2, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<int3, uchar3>(1, 480, 360, FMT_RGBs32, FMT_RGB8, 1.2, 10.2, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<int3, char3>(1, 480, 360, FMT_RGBs32, FMT_RGBs8, 1.2, 10.2, eDeviceType::CPU)));

    TEST_CASE((TestCorrectness<uchar3, float3>(1, 480, 360, FMT_RGB8, FMT_RGBf32, 1.2, 10.2, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<char3, float3>(1, 480, 360, FMT_RGBs8, FMT_RGBf32, 1.2, 10.2, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<float3, uchar3>(1, 480, 360, FMT_RGBf32, FMT_RGB8, 1.2, 10.2, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<float3, char3>(1, 480, 360, FMT_RGBf32, FMT_RGBs8, 1.2, 10.2, eDeviceType::CPU)));

    // 4 Channels
    TEST_CASE((TestCorrectness<uchar4, uchar4>(1, 480, 360, FMT_RGBA8, FMT_RGBA8, 1.2, 10.2, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<char4, char4>(1, 480, 360, FMT_RGBAs8, FMT_RGBAs8, 1.2, 10.2, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<uchar4, char4>(1, 480, 360, FMT_RGBA8, FMT_RGBAs8, 1.2, 10.2, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<char4, uchar4>(1, 480, 360, FMT_RGBAs8, FMT_RGBA8, 1.2, 10.2, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<uchar4, ushort4>(1, 480, 360, FMT_RGBA8, FMT_RGBA16, 1.2, 10.2, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<char4, short4>(1, 480, 360, FMT_RGBAs8, FMT_RGBAs16, 1.2, 10.2, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<ushort4, uchar4>(1, 480, 360, FMT_RGBA16, FMT_RGBA8, 1.2, 10.2, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<short4, char4>(1, 480, 360, FMT_RGBAs16, FMT_RGBAs8, 1.2, 10.2, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<ushort4, char4>(1, 480, 360, FMT_RGBA16, FMT_RGBAs8, 1.2, 10.2, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<short4, uchar4>(1, 480, 360, FMT_RGBAs16, FMT_RGBA8, 1.2, 10.2, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<ushort4, ushort4>(1, 480, 360, FMT_RGBA16, FMT_RGBA16, 1.2, 10.2, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<short4, short4>(1, 480, 360, FMT_RGBAs16, FMT_RGBAs16, 1.2, 10.2, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<ushort4, short4>(1, 480, 360, FMT_RGBA16, FMT_RGBAs16, 1.2, 10.2, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<short4, ushort4>(1, 480, 360, FMT_RGBAs16, FMT_RGBA16, 1.2, 10.2, eDeviceType::CPU)));

    TEST_CASE((TestCorrectness<uchar4, int4>(1, 480, 360, FMT_RGBA8, FMT_RGBAs32, 1.2, 10.2, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<char4, int4>(1, 480, 360, FMT_RGBAs8, FMT_RGBAs32, 1.2, 10.2, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<int4, uchar4>(1, 480, 360, FMT_RGBAs32, FMT_RGBA8, 1.2, 10.2, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<int4, char4>(1, 480, 360, FMT_RGBAs32, FMT_RGBAs8, 1.2, 10.2, eDeviceType::CPU)));

    TEST_CASE((TestCorrectness<uchar4, float4>(1, 480, 360, FMT_RGBA8, FMT_RGBAf32, 1.2, 10.2, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<char4, float4>(1, 480, 360, FMT_RGBAs8, FMT_RGBAf32, 1.2, 10.2, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<float4, uchar4>(1, 480, 360, FMT_RGBAf32, FMT_RGBA8, 1.2, 10.2, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<float4, char4>(1, 480, 360, FMT_RGBAf32, FMT_RGBAs8, 1.2, 10.2, eDeviceType::CPU)));

    // GPU Correctness Tests
    // 1 Channels
    TEST_CASE((TestCorrectness<uchar1, uchar1>(1, 480, 360, FMT_U8, FMT_U8, 1.2, 10.2, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<char1, char1>(1, 480, 360, FMT_S8, FMT_S8, 1.2, 10.2, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<uchar1, char1>(1, 480, 360, FMT_U8, FMT_S8, 1.2, 10.2, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<char1, uchar1>(1, 480, 360, FMT_S8, FMT_U8, 1.2, 10.2, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<uchar1, ushort1>(1, 480, 360, FMT_U8, FMT_U16, 1.2, 10.2, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<char1, short1>(1, 480, 360, FMT_S8, FMT_S16, 1.2, 10.2, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<ushort1, uchar1>(1, 480, 360, FMT_U16, FMT_U8, 1.2, 10.2, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<short1, char1>(1, 480, 360, FMT_S16, FMT_S8, 1.2, 10.2, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<ushort1, char1>(1, 480, 360, FMT_U16, FMT_S8, 1.2, 10.2, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<short1, uchar1>(1, 480, 360, FMT_S16, FMT_U8, 1.2, 10.2, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<ushort1, ushort1>(1, 480, 360, FMT_U16, FMT_U16, 1.2, 10.2, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<short1, short1>(1, 480, 360, FMT_S16, FMT_S16, 1.2, 10.2, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<ushort1, short1>(1, 480, 360, FMT_U16, FMT_S16, 1.2, 10.2, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<short1, ushort1>(1, 480, 360, FMT_S16, FMT_U16, 1.2, 10.2, eDeviceType::GPU)));

    TEST_CASE((TestCorrectness<uchar1, int1>(1, 480, 360, FMT_U8, FMT_S32, 1.2, 10.2, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<char1, int1>(1, 480, 360, FMT_S8, FMT_S32, 1.2, 10.2, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<int1, uchar1>(1, 480, 360, FMT_S32, FMT_U8, 1.2, 10.2, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<int1, char1>(1, 480, 360, FMT_S32, FMT_S8, 1.2, 10.2, eDeviceType::GPU)));

    TEST_CASE((TestCorrectness<uchar1, float1>(1, 480, 360, FMT_U8, FMT_F32, 1.2, 10.2, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<char1, float1>(1, 480, 360, FMT_S8, FMT_F32, 1.2, 10.2, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<float1, uchar1>(1, 480, 360, FMT_F32, FMT_U8, 1.2, 10.2, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<float1, char1>(1, 480, 360, FMT_F32, FMT_S8, 1.2, 10.2, eDeviceType::GPU)));

    // 3 Channels
    TEST_CASE((TestCorrectness<uchar3, uchar3>(1, 480, 360, FMT_RGB8, FMT_RGB8, 1.2, 10.2, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<char3, char3>(1, 480, 360, FMT_RGBs8, FMT_RGBs8, 1.2, 10.2, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<uchar3, char3>(1, 480, 360, FMT_RGB8, FMT_RGBs8, 1.2, 10.2, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<char3, uchar3>(1, 480, 360, FMT_RGBs8, FMT_RGB8, 1.2, 10.2, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<uchar3, ushort3>(1, 480, 360, FMT_RGB8, FMT_RGB16, 1.2, 10.2, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<char3, short3>(1, 480, 360, FMT_RGBs8, FMT_RGBs16, 1.2, 10.2, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<ushort3, uchar3>(1, 480, 360, FMT_RGB16, FMT_RGB8, 1.2, 10.2, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<short3, char3>(1, 480, 360, FMT_RGBs16, FMT_RGBs8, 1.2, 10.2, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<ushort3, char3>(1, 480, 360, FMT_RGB16, FMT_RGBs8, 1.2, 10.2, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<short3, uchar3>(1, 480, 360, FMT_RGBs16, FMT_RGB8, 1.2, 10.2, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<ushort3, ushort3>(1, 480, 360, FMT_RGB16, FMT_RGB16, 1.2, 10.2, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<short3, short3>(1, 480, 360, FMT_RGBs16, FMT_RGBs16, 1.2, 10.2, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<ushort3, short3>(1, 480, 360, FMT_RGB16, FMT_RGBs16, 1.2, 10.2, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<short3, ushort3>(1, 480, 360, FMT_RGBs16, FMT_RGB16, 1.2, 10.2, eDeviceType::GPU)));

    TEST_CASE((TestCorrectness<uchar3, int3>(1, 480, 360, FMT_RGB8, FMT_RGBs32, 1.2, 10.2, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<char3, int3>(1, 480, 360, FMT_RGBs8, FMT_RGBs32, 1.2, 10.2, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<int3, uchar3>(1, 480, 360, FMT_RGBs32, FMT_RGB8, 1.2, 10.2, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<int3, char3>(1, 480, 360, FMT_RGBs32, FMT_RGBs8, 1.2, 10.2, eDeviceType::GPU)));

    TEST_CASE((TestCorrectness<uchar3, float3>(1, 480, 360, FMT_RGB8, FMT_RGBf32, 1.2, 10.2, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<char3, float3>(1, 480, 360, FMT_RGBs8, FMT_RGBf32, 1.2, 10.2, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<float3, uchar3>(1, 480, 360, FMT_RGBf32, FMT_RGB8, 1.2, 10.2, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<float3, char3>(1, 480, 360, FMT_RGBf32, FMT_RGBs8, 1.2, 10.2, eDeviceType::GPU)));

    // 4 Channels
    TEST_CASE((TestCorrectness<uchar4, uchar4>(1, 480, 360, FMT_RGBA8, FMT_RGBA8, 1.2, 10.2, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<char4, char4>(1, 480, 360, FMT_RGBAs8, FMT_RGBAs8, 1.2, 10.2, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<uchar4, char4>(1, 480, 360, FMT_RGBA8, FMT_RGBAs8, 1.2, 10.2, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<char4, uchar4>(1, 480, 360, FMT_RGBAs8, FMT_RGBA8, 1.2, 10.2, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<uchar4, ushort4>(1, 480, 360, FMT_RGBA8, FMT_RGBA16, 1.2, 10.2, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<char4, short4>(1, 480, 360, FMT_RGBAs8, FMT_RGBAs16, 1.2, 10.2, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<ushort4, uchar4>(1, 480, 360, FMT_RGBA16, FMT_RGBA8, 1.2, 10.2, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<short4, char4>(1, 480, 360, FMT_RGBAs16, FMT_RGBAs8, 1.2, 10.2, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<ushort4, char4>(1, 480, 360, FMT_RGBA16, FMT_RGBAs8, 1.2, 10.2, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<short4, uchar4>(1, 480, 360, FMT_RGBAs16, FMT_RGBA8, 1.2, 10.2, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<ushort4, ushort4>(1, 480, 360, FMT_RGBA16, FMT_RGBA16, 1.2, 10.2, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<short4, short4>(1, 480, 360, FMT_RGBAs16, FMT_RGBAs16, 1.2, 10.2, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<ushort4, short4>(1, 480, 360, FMT_RGBA16, FMT_RGBAs16, 1.2, 10.2, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<short4, ushort4>(1, 480, 360, FMT_RGBAs16, FMT_RGBA16, 1.2, 10.2, eDeviceType::GPU)));

    TEST_CASE((TestCorrectness<uchar4, int4>(1, 480, 360, FMT_RGBA8, FMT_RGBAs32, 1.2, 10.2, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<char4, int4>(1, 480, 360, FMT_RGBAs8, FMT_RGBAs32, 1.2, 10.2, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<int4, uchar4>(1, 480, 360, FMT_RGBAs32, FMT_RGBA8, 1.2, 10.2, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<int4, char4>(1, 480, 360, FMT_RGBAs32, FMT_RGBAs8, 1.2, 10.2, eDeviceType::GPU)));

    TEST_CASE((TestCorrectness<uchar4, float4>(1, 480, 360, FMT_RGBA8, FMT_RGBAf32, 1.2, 10.2, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<char4, float4>(1, 480, 360, FMT_RGBAs8, FMT_RGBAf32, 1.2, 10.2, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<float4, uchar4>(1, 480, 360, FMT_RGBAf32, FMT_RGBA8, 1.2, 10.2, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<float4, char4>(1, 480, 360, FMT_RGBAf32, FMT_RGBAs8, 1.2, 10.2, eDeviceType::GPU)));

    TEST_CASES_END();
}