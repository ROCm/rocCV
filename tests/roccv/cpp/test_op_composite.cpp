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
#include <core/wrappers/image_wrapper.hpp>
#include <op_composite.hpp>

#include "test_helpers.hpp"

using namespace roccv;
using namespace roccv::tests;

namespace {
/**
 * @brief Computes the Golden model for the Composite operation, which blends foreground and background images together
 * using an alpha mask.
 *
 * @tparam InputType The type for the foreground and background images.
 * @tparam OutputType The type for the final output images.
 * @tparam MaskType The type for the alpha mask images.
 * @param[in] foreground Foreground images to blend.
 * @param[in] background Background images to blend.
 * @param[in] mask Alpha mask images.
 * @param[in] batchSize The number of images within the batch.
 * @param[in] width The width of each image in the batch.
 * @param[in] height The height of each image in the batch.
 * @param[in] outChannels The number of output channels for the output images (adds an optional alpha channel if 4
 * channels are desired)
 * @return Golden output for the Composite operator as a vector of the base type of the input images.
 */
template <typename InputType, typename OutputType, typename MaskType>
std::vector<detail::BaseType<OutputType>> GoldenComposite(std::vector<detail::BaseType<InputType>> foreground,
                                                          std::vector<detail::BaseType<InputType>> background,
                                                          std::vector<detail::BaseType<MaskType>> mask, int batchSize,
                                                          int width, int height) {
    // Defines the working type for intermediate math. The working type is always a base type of float with the same
    // number of elements as the InputType. For example, if input type is uchar3, working type is float3.
    using WorkType = detail::MakeType<float, detail::NumElements<InputType>>;

    // Wrap input data into ImageWrappers for easy data access
    ImageWrapper<InputType> fgWrap(foreground, batchSize, width, height);
    ImageWrapper<InputType> bgWrap(background, batchSize, width, height);
    ImageWrapper<MaskType> maskWrap(mask, batchSize, width, height);

    // Size of the output depends on the requested number of output channels. If it is 3, then the output images will
    // have 3 channels. If it is 4, then an additional alpha channel is added to the output. This alpha channel is
    // always fully on.
    int numOutElements = batchSize * width * height * detail::NumElements<OutputType>;
    std::vector<detail::BaseType<OutputType>> output(numOutElements);
    ImageWrapper<OutputType> outWrap(output, batchSize, width, height);

    for (int b = 0; b < batchSize; b++) {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                // Mask will always only have a single channel. We use the RangeCast to normalize it into a float of
                // range 0-1.0f.
                float1 maskVal = detail::RangeCast<float1>(maskWrap.at(b, y, x, 0));

                // We cast our foreground and background values into the appropriate WorkType to maintain consistency
                // and precision during math.
                WorkType fgVal = detail::RangeCast<WorkType>(fgWrap.at(b, y, x, 0));
                WorkType bgVal = detail::RangeCast<WorkType>(bgWrap.at(b, y, x, 0));

                // Blend the foreground and background values with the corresponding mask value.
                WorkType outVal = bgVal + maskVal.x * (fgVal - bgVal);

                // Cast the outVal (which is still in the float working type) back to the range of the output type using
                // a RangeCast.
                if constexpr (detail::NumElements<OutputType> == 4) {
                    // If the number of channels in the output type is 4, we need to add an extra alpha channel. This
                    // assumes that the input types have at least three channels (which they do according to the
                    // operator's spec).
                    outWrap.at(b, y, x, 0) =
                        detail::RangeCast<OutputType>((detail::MakeType<float, 4>){outVal.x, outVal.y, outVal.z, 1.0f});
                } else {
                    // Otherwise, there is no more work to be done and we cast the working type back to the output type
                    outWrap.at(b, y, x, 0) = detail::RangeCast<OutputType>(outVal);
                }
            }
        }
    }

    return output;
}

/**
 * @brief Performs a correctness test for the Composite operator by comparing actual results to the results of the
 * golden model.
 *
 * @tparam InputType The type for the foreground and background images.
 * @tparam MaskType Type for the alpha mask images.
 * @tparam OutputType Type for the output images.
 * @param[in] batchSize Number of images in the batch.
 * @param[in] width Width of images in the batch.
 * @param[in] height Height of images in the batch.
 * @param[in] fmt Format for foreground and background images.
 * @param[in] mskFmt Format for mask images.
 * @param[in] outFormat Format for output images.
 * @param[in] device Device for this correctness test.
 */
template <typename InputType, typename MaskType, typename OutputType>
void TestCorrectness(int batchSize, int width, int height, ImageFormat fmt, ImageFormat mskFmt, ImageFormat outFormat,
                     eDeviceType device) {
    // Create required input tensors for actual output
    Tensor foreground(batchSize, {width, height}, fmt, device);
    Tensor background(batchSize, {width, height}, fmt, device);
    Tensor mask(batchSize, {width, height}, mskFmt, device);

    // Fill input vectors with random data. They should be using different seeds to make sure the generated images
    // differ.
    std::vector<detail::BaseType<InputType>> foregroundData(foreground.shape().size());
    FillVector(foregroundData);
    std::vector<detail::BaseType<InputType>> backgroundData(background.shape().size());
    FillVector(backgroundData, 54321);
    std::vector<detail::BaseType<MaskType>> maskData(mask.shape().size());
    FillVector(maskData, 1331);

    // Copy random image data into input tensors
    CopyVectorIntoTensor(foreground, foregroundData);
    CopyVectorIntoTensor(background, backgroundData);
    CopyVectorIntoTensor(mask, maskData);

    // Generate golden output
    auto goldenOutput = GoldenComposite<InputType, OutputType, MaskType>(foregroundData, backgroundData, maskData,
                                                                         batchSize, width, height);

    // Run actual operator to gain actual output
    Tensor output(batchSize, {width, height}, outFormat, device);

    hipStream_t stream;
    HIP_VALIDATE_NO_ERRORS(hipStreamCreate(&stream));

    Composite op;
    op(stream, foreground, background, mask, output, device);
    HIP_VALIDATE_NO_ERRORS(hipStreamSynchronize(stream));
    HIP_VALIDATE_NO_ERRORS(hipStreamDestroy(stream));

    // Copy actual results into a host allocated vector for comparison.
    std::vector<detail::BaseType<OutputType>> actualOutput(output.shape().size());
    CopyTensorIntoVector(actualOutput, output);

    // Compare final vectors
    CompareVectorsNear(actualOutput, goldenOutput);
}
}  // namespace

eTestStatusType test_op_composite(int argc, char **argv) {
    TEST_CASES_BEGIN();

    // Test RGB8 input/output
    TEST_CASE((TestCorrectness<uchar3, uchar1, uchar3>(1, 80, 40, FMT_RGB8, FMT_U8, FMT_RGB8, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<uchar3, uchar1, uchar3>(5, 56, 32, FMT_RGB8, FMT_U8, FMT_RGB8, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<uchar3, uchar1, uchar3>(1, 80, 40, FMT_RGB8, FMT_U8, FMT_RGB8, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<uchar3, uchar1, uchar3>(5, 56, 32, FMT_RGB8, FMT_U8, FMT_RGB8, eDeviceType::CPU)));

    // Test RGB8 input, RGBA8 output (to ensure alpha-channel is being set properly)
    TEST_CASE((TestCorrectness<uchar3, uchar1, uchar4>(1, 80, 40, FMT_RGB8, FMT_U8, FMT_RGBA8, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<uchar3, uchar1, uchar4>(5, 56, 32, FMT_RGB8, FMT_U8, FMT_RGBA8, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<uchar3, uchar1, uchar4>(1, 80, 40, FMT_RGB8, FMT_U8, FMT_RGBA8, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<uchar3, uchar1, uchar4>(5, 56, 32, FMT_RGB8, FMT_U8, FMT_RGBA8, eDeviceType::CPU)));

    // Test RGBf32 input/output
    TEST_CASE((TestCorrectness<float3, uchar1, float3>(1, 80, 40, FMT_RGBf32, FMT_U8, FMT_RGBf32, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<float3, uchar1, float3>(5, 56, 32, FMT_RGBf32, FMT_U8, FMT_RGBf32, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<float3, uchar1, float3>(1, 80, 40, FMT_RGBf32, FMT_U8, FMT_RGBf32, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<float3, uchar1, float3>(5, 56, 32, FMT_RGBf32, FMT_U8, FMT_RGBf32, eDeviceType::CPU)));

    // Test RGBf32 input, RGBAf32 output (alpha channel)
    TEST_CASE((TestCorrectness<float3, uchar1, float4>(1, 80, 40, FMT_RGBf32, FMT_U8, FMT_RGBAf32, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<float3, uchar1, float4>(5, 56, 32, FMT_RGBf32, FMT_U8, FMT_RGBAf32, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<float3, uchar1, float4>(1, 80, 40, FMT_RGBf32, FMT_U8, FMT_RGBAf32, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<float3, uchar1, float4>(5, 56, 32, FMT_RGBf32, FMT_U8, FMT_RGBAf32, eDeviceType::CPU)));

    // Test RGB8 input/output, F32 alpha mask
    TEST_CASE((TestCorrectness<uchar3, float1, uchar3>(1, 80, 40, FMT_RGB8, FMT_F32, FMT_RGB8, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<uchar3, float1, uchar3>(5, 56, 32, FMT_RGB8, FMT_F32, FMT_RGB8, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<uchar3, float1, uchar3>(1, 80, 40, FMT_RGB8, FMT_F32, FMT_RGB8, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<uchar3, float1, uchar3>(5, 56, 32, FMT_RGB8, FMT_F32, FMT_RGB8, eDeviceType::CPU)));

    TEST_CASES_END();
}