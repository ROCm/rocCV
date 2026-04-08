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
#include <core/detail/vector_utils.hpp>
#include <core/wrappers/border_wrapper.hpp>
#include <core/wrappers/image_wrapper.hpp>
#include <op_avg_blur.hpp>

#include "test_helpers.hpp"

using namespace roccv;
using namespace roccv::detail;
using namespace roccv::tests;

namespace {

/**
 * @brief Verified golden C++ model for the average blur operation on one image.
 *
 * @tparam T Vectorized datatype of the image's pixels.
 * @param[in] input Input tensor containing image data.
 * @param[out] output Output tensor containing blurred image data.
 * @param[in] kernelWidth Width of the averaging kernel
 * @param[in] kernelHeight Height of the averaging kernel
 * @param[in] kernelAnchorX X-coordinate of kernel anchor point
 * @param[in] kernelAnchorY Y-coordinate of kernel anchor point
 * @param[in] borderMode Border pixel extrapolation method
 * @param[in] borderValue Color for constant border mode
 * @return None.
 */
template <typename T, eBorderType borderMode, typename BT = detail::BaseType<T>>
void GenerateGoldenAvgBlur(std::vector<BT>& input, std::vector<BT>& output, int32_t batchSize, Size2D imageSize,
                           int kernelWidth, int kernelHeight, int kernelAnchorX, int kernelAnchorY, T borderValue) {
    BorderWrapper<T, borderMode> src(ImageWrapper<T>(input, batchSize, imageSize.w, imageSize.h), borderValue);
    ImageWrapper<T> dst(output, batchSize, imageSize.w, imageSize.h);
    using namespace roccv::detail;
    using WorkType = MakeType<float, NumElements<T>>;

    // Compute kernel area for averaging
    float kernelArea = static_cast<float>(kernelWidth * kernelHeight);

    // Iterate over all batches
    for (int b = 0; b < dst.batches(); b++) {
        // Iterate over all output pixels
        for (int j = 0; j < dst.height(); j++) {
            for (int i = 0; i < dst.width(); i++) {
                // Initialize accumulator
                WorkType sum = SetAll<WorkType>(0.0f);

                // Compute the sum over the kernel window
                for (int ky = 0; ky < kernelHeight; ++ky) {
                    int srcY = j - kernelAnchorY + ky;

                    for (int kx = 0; kx < kernelWidth; ++kx) {
                        int srcX = i - kernelAnchorX + kx;

                        T pixel = src.at(b, srcY, srcX, 0);

                        sum = sum + StaticCast<WorkType>(pixel);
                    }
                }

                WorkType average = sum / kernelArea;

                dst.at(b, j, i, 0) = SaturateCast<T>(average);
            }
        }
    }
}

/**
 * @brief Tests correctness of the average blur operator, comparing it against a generated golden result.
 *
 * @tparam T Underlying datatype of the image's pixels.
 * @tparam BT Base type of the image data.
 * @param[in] batchSize Number of images in the batch.
 * @param[in] width Width of each image in the batch.
 * @param[in] height Height of each image in the batch.
 * @param[in] format Image format.
 * @param[in] kernelWidth Width of the averaging kernel
 * @param[in] kernelHeight Height of the averaging kernel
 * @param[in] borderColor Color for constant border mode
 * @param[in] device Device this correctness test should be run on.
 */
template <typename T, eBorderType BorderMode, typename BT = detail::BaseType<T>>
void TestCorrectness(int batchSize, int width, int height, ImageFormat format, int kernelWidth, int kernelHeight,
                     float4 borderColor, eDeviceType device) {
    // Create input and output tensor based on test parameters
    Tensor input(batchSize, {width, height}, format, device);
    Tensor output(batchSize, {width, height}, format, device);

    // Create a vector and fill it with random data.
    std::vector<BT> inputData(input.shape().size());
    FillVector(inputData);
    if constexpr (std::is_floating_point_v<BT>) {
        for (int i = 0; i < inputData.size(); i++) {
            inputData[i] *= static_cast<BT>(std::numeric_limits<ushort>::max());
        }
    }

    // Copy generated input data into input tensor
    CopyVectorIntoTensor(input, inputData);

    // Calculate kernel anchor (center of kernel)
    int kernelAnchorX = kernelWidth / 2;
    int kernelAnchorY = kernelHeight / 2;

    hipStream_t stream;
    HIP_VALIDATE_NO_ERRORS(hipStreamCreate(&stream));
    AvgBlur op;
    op(stream, input, output, kernelWidth, kernelHeight, kernelAnchorX, kernelAnchorY, BorderMode, borderColor,
       device);
    HIP_VALIDATE_NO_ERRORS(hipStreamSynchronize(stream));
    HIP_VALIDATE_NO_ERRORS(hipStreamDestroy(stream));

    // Copy data from output tensor into a host allocated vector
    std::vector<BT> outputData(output.shape().size());
    CopyTensorIntoVector(outputData, output);

    // Calculate golden reference
    std::vector<BT> refData(output.shape().size());
    GenerateGoldenAvgBlur<T, BorderMode>(inputData, refData, batchSize, {width, height}, kernelWidth, kernelHeight,
                                         kernelAnchorX, kernelAnchorY, detail::SaturateCast<T>(borderColor));

    // Compare data in actual output versus the generated golden reference image
    CompareVectorsNear(outputData, refData, 1);
}

}  // namespace

int main(int argc, char** argv) {
    TEST_CASES_BEGIN();

    // GPU correctness tests - U8 (1 channel)
    TEST_CASE((TestCorrectness<uchar, BORDER_TYPE_CONSTANT>(1, 20, 20, FMT_U8, 3, 3, {0.0, 0.0, 0.0, 0.0},
                                                            eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<uchar, BORDER_TYPE_REPLICATE>(4, 32, 32, FMT_U8, 5, 5, {0.0, 0.0, 0.0, 0.0},
                                                             eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<uchar, BORDER_TYPE_REFLECT>(2, 24, 24, FMT_U8, 7, 7, {0.0, 0.0, 0.0, 0.0},
                                                           eDeviceType::GPU)));

    // GPU correctness tests - RGB8 (3 channels)
    TEST_CASE((TestCorrectness<uchar3, BORDER_TYPE_CONSTANT>(1, 20, 20, FMT_RGB8, 3, 3, {0.0, 0.0, 0.0, 0.0},
                                                             eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<uchar3, BORDER_TYPE_REFLECT>(2, 32, 32, FMT_RGB8, 5, 5, {100.0, 100.0, 0.0, 0.0},
                                                            eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<uchar3, BORDER_TYPE_WRAP>(1, 16, 16, FMT_RGB8, 3, 3, {50.0, 50.0, 50.0, 0.0},
                                                         eDeviceType::GPU)));

    // GPU correctness tests - RGBA8 (4 channels)
    TEST_CASE((TestCorrectness<uchar4, BORDER_TYPE_WRAP>(1, 10, 10, FMT_RGBA8, 3, 3, {0.0, 0.0, 0.0, 0.0},
                                                         eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<uchar4, BORDER_TYPE_REPLICATE>(5, 64, 64, FMT_RGBA8, 5, 5, {0.0, 0.0, 0.0, 0.0},
                                                              eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<uchar4, BORDER_TYPE_REFLECT101>(2, 20, 20, FMT_RGBA8, 7, 7, {128.0, 128.0, 128.0, 255.0},
                                                               eDeviceType::GPU)));

    // GPU correctness tests - S16 (signed 16-bit)
    TEST_CASE((TestCorrectness<short1, BORDER_TYPE_CONSTANT>(1, 20, 20, FMT_S16, 3, 3, {500.0, 500.0, 0.0, 0.0},
                                                             eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<short1, BORDER_TYPE_REFLECT>(3, 20, 20, FMT_S16, 5, 5, {500.0, 500.0, 0.0, 0.0},
                                                            eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<short3, BORDER_TYPE_REPLICATE>(2, 16, 16, FMT_RGBs16, 3, 3, {100.0, 100.0, 100.0, 0.0},
                                                              eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<short4, BORDER_TYPE_WRAP>(1, 24, 24, FMT_RGBAs16, 5, 5, {0.0, 0.0, 0.0, 0.0},
                                                         eDeviceType::GPU)));

    // GPU correctness tests - U16 (unsigned 16-bit)
    TEST_CASE((TestCorrectness<ushort1, BORDER_TYPE_CONSTANT>(1, 20, 20, FMT_U16, 3, 3, {0.0, 0.0, 0.0, 0.0},
                                                              eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<ushort1, BORDER_TYPE_REFLECT>(2, 20, 20, FMT_U16, 5, 5, {0.0, 0.0, 0.0, 0.0},
                                                             eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<ushort3, BORDER_TYPE_CONSTANT>(1, 20, 20, FMT_RGB16, 3, 3, {500.0, 600.0, 0.0, 0.0},
                                                              eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<ushort4, BORDER_TYPE_REFLECT>(1, 20, 20, FMT_RGBA16, 5, 5, {500.0, 600.0, 0.0, 0.0},
                                                             eDeviceType::GPU)));

    // GPU correctness tests - S32 (signed 32-bit)
    TEST_CASE((TestCorrectness<int1, BORDER_TYPE_CONSTANT>(1, 32, 32, FMT_S32, 3, 3, {500.0, 500.0, 0.0, 0.0},
                                                           eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<int1, BORDER_TYPE_WRAP>(2, 32, 32, FMT_S32, 5, 5, {500.0, 500.0, 0.0, 0.0},
                                                       eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<int3, BORDER_TYPE_REPLICATE>(1, 16, 16, FMT_RGBs32, 3, 3, {100.0, 100.0, 100.0, 0.0},
                                                            eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<int4, BORDER_TYPE_REFLECT>(2, 24, 24, FMT_RGBAs32, 5, 5, {0.0, 0.0, 0.0, 0.0},
                                                          eDeviceType::GPU)));

    // GPU correctness tests - F32 (float)
    TEST_CASE((TestCorrectness<float1, BORDER_TYPE_REPLICATE>(1, 24, 24, FMT_F32, 3, 3, {500.0, 500.0, 0.0, 0.0},
                                                              eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<float1, BORDER_TYPE_WRAP>(2, 24, 24, FMT_F32, 5, 5, {600.0, 500.0, 0.0, 0.0},
                                                         eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<float3, BORDER_TYPE_REPLICATE>(1, 24, 24, FMT_RGBf32, 3, 3, {500.0, 500.0, 0.0, 0.0},
                                                              eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<float3, BORDER_TYPE_WRAP>(2, 24, 24, FMT_RGBf32, 7, 7, {600.0, 500.0, 0.0, 0.0},
                                                         eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<float4, BORDER_TYPE_REPLICATE>(1, 24, 24, FMT_RGBAf32, 3, 3, {500.0, 500.0, 0.0, 0.0},
                                                              eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<float4, BORDER_TYPE_WRAP>(2, 24, 24, FMT_RGBAf32, 5, 5, {600.0, 500.0, 0.0, 0.0},
                                                         eDeviceType::GPU)));

    // CPU correctness tests - U8 (1 channel)
    TEST_CASE((TestCorrectness<uchar, BORDER_TYPE_CONSTANT>(1, 20, 20, FMT_U8, 3, 3, {0.0, 0.0, 0.0, 0.0},
                                                            eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<uchar, BORDER_TYPE_REPLICATE>(4, 32, 32, FMT_U8, 5, 5, {0.0, 0.0, 0.0, 0.0},
                                                             eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<uchar, BORDER_TYPE_REFLECT>(2, 24, 24, FMT_U8, 7, 7, {0.0, 0.0, 0.0, 0.0},
                                                           eDeviceType::CPU)));

    // CPU correctness tests - RGB8 (3 channels)
    TEST_CASE((TestCorrectness<uchar3, BORDER_TYPE_CONSTANT>(1, 20, 20, FMT_RGB8, 3, 3, {0.0, 0.0, 0.0, 0.0},
                                                             eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<uchar3, BORDER_TYPE_REFLECT>(2, 32, 32, FMT_RGB8, 5, 5, {100.0, 100.0, 0.0, 0.0},
                                                            eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<uchar3, BORDER_TYPE_WRAP>(1, 16, 16, FMT_RGB8, 3, 3, {50.0, 50.0, 50.0, 0.0},
                                                         eDeviceType::CPU)));

    // CPU correctness tests - RGBA8 (4 channels)
    TEST_CASE((TestCorrectness<uchar4, BORDER_TYPE_WRAP>(1, 10, 10, FMT_RGBA8, 3, 3, {0.0, 0.0, 0.0, 0.0},
                                                         eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<uchar4, BORDER_TYPE_REPLICATE>(5, 64, 64, FMT_RGBA8, 5, 5, {0.0, 0.0, 0.0, 0.0},
                                                              eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<uchar4, BORDER_TYPE_REFLECT101>(2, 20, 20, FMT_RGBA8, 7, 7, {128.0, 128.0, 128.0, 255.0},
                                                               eDeviceType::CPU)));

    // CPU correctness tests - S16 (signed 16-bit)
    TEST_CASE((TestCorrectness<short1, BORDER_TYPE_CONSTANT>(1, 20, 20, FMT_S16, 3, 3, {500.0, 500.0, 0.0, 0.0},
                                                             eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<short1, BORDER_TYPE_REFLECT>(3, 20, 20, FMT_S16, 5, 5, {500.0, 500.0, 0.0, 0.0},
                                                            eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<short3, BORDER_TYPE_REPLICATE>(2, 16, 16, FMT_RGBs16, 3, 3, {100.0, 100.0, 100.0, 0.0},
                                                              eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<short4, BORDER_TYPE_WRAP>(1, 24, 24, FMT_RGBAs16, 5, 5, {0.0, 0.0, 0.0, 0.0},
                                                         eDeviceType::CPU)));

    // CPU correctness tests - U16 (unsigned 16-bit)
    TEST_CASE((TestCorrectness<ushort1, BORDER_TYPE_CONSTANT>(1, 20, 20, FMT_U16, 3, 3, {0.0, 0.0, 0.0, 0.0},
                                                              eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<ushort1, BORDER_TYPE_REFLECT>(2, 20, 20, FMT_U16, 5, 5, {0.0, 0.0, 0.0, 0.0},
                                                             eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<ushort3, BORDER_TYPE_CONSTANT>(1, 20, 20, FMT_RGB16, 3, 3, {500.0, 600.0, 0.0, 0.0},
                                                              eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<ushort4, BORDER_TYPE_REFLECT>(1, 20, 20, FMT_RGBA16, 5, 5, {500.0, 600.0, 0.0, 0.0},
                                                             eDeviceType::CPU)));

    // CPU correctness tests - S32 (signed 32-bit)
    TEST_CASE((TestCorrectness<int1, BORDER_TYPE_CONSTANT>(1, 32, 32, FMT_S32, 3, 3, {500.0, 500.0, 0.0, 0.0},
                                                           eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<int1, BORDER_TYPE_WRAP>(2, 32, 32, FMT_S32, 5, 5, {500.0, 500.0, 0.0, 0.0},
                                                       eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<int3, BORDER_TYPE_REPLICATE>(1, 16, 16, FMT_RGBs32, 3, 3, {100.0, 100.0, 100.0, 0.0},
                                                            eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<int4, BORDER_TYPE_REFLECT>(2, 24, 24, FMT_RGBAs32, 5, 5, {0.0, 0.0, 0.0, 0.0},
                                                          eDeviceType::CPU)));

    // CPU correctness tests - F32 (float)
    TEST_CASE((TestCorrectness<float1, BORDER_TYPE_REPLICATE>(1, 24, 24, FMT_F32, 3, 3, {500.0, 500.0, 0.0, 0.0},
                                                              eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<float1, BORDER_TYPE_WRAP>(2, 24, 24, FMT_F32, 5, 5, {600.0, 500.0, 0.0, 0.0},
                                                         eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<float3, BORDER_TYPE_REPLICATE>(1, 24, 24, FMT_RGBf32, 3, 3, {500.0, 500.0, 0.0, 0.0},
                                                              eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<float3, BORDER_TYPE_WRAP>(2, 24, 24, FMT_RGBf32, 7, 7, {600.0, 500.0, 0.0, 0.0},
                                                         eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<float4, BORDER_TYPE_REPLICATE>(1, 24, 24, FMT_RGBAf32, 3, 3, {500.0, 500.0, 0.0, 0.0},
                                                              eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<float4, BORDER_TYPE_WRAP>(2, 24, 24, FMT_RGBAf32, 5, 5, {600.0, 500.0, 0.0, 0.0},
                                                         eDeviceType::CPU)));

    // Additional edge cases - various kernel sizes
    TEST_CASE((TestCorrectness<uchar3, BORDER_TYPE_REPLICATE>(1, 40, 40, FMT_RGB8, 9, 9, {0.0, 0.0, 0.0, 0.0},
                                                              eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<float4, BORDER_TYPE_CONSTANT>(1, 50, 50, FMT_RGBAf32, 11, 11, {0.0, 0.0, 0.0, 0.0},
                                                             eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<uchar3, BORDER_TYPE_REPLICATE>(1, 40, 40, FMT_RGB8, 9, 9, {0.0, 0.0, 0.0, 0.0},
                                                              eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<float4, BORDER_TYPE_CONSTANT>(1, 50, 50, FMT_RGBAf32, 11, 11, {0.0, 0.0, 0.0, 0.0},
                                                             eDeviceType::CPU)));

    TEST_CASES_END();
}
