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
#include <core/wrappers/interpolation_wrapper.hpp>
#include <op_rotate.hpp>
#include <opencv2/opencv.hpp>

#include "math_utils.hpp"
#include "test_helpers.hpp"

using namespace roccv;
using namespace roccv::tests;

namespace {

/**
 * @brief Computes the shift required to move the resulting rotated image back to the center of the image.
 *
 * @param centerX The x coordinate for the center of the image.
 * @param centerY The y coordinate for the center of the image.
 * @param angle The angle in degrees the resulting image will be rotated.
 * @return A double2 with the shift required to translate the image back to its center after a rotation.
 */
double2 ComputeCenterShift(const double centerX, const double centerY, const double angle) {
    double xShift = (1 - cos(angle * M_PI / 180)) * centerX - sin(angle * M_PI / 180) * centerY;
    double yShift = sin(angle * M_PI / 180) * centerX + (1 - cos(angle * M_PI / 180)) * centerY;
    return {xShift, yShift};
}

/**
 * @brief Golden model to rotate an image clockwise. This will always use a constant border mode with all values set to
 * 0.
 *
 * @tparam T The image datatype.
 * @tparam InterpType The interpolation type to use.
 * @param input Input images.
 * @param batchSize The number of images in the batch.
 * @param imageSize The size of each image in the batch.
 * @param angle The angle in degrees to rotate counter-clockwise.
 * @param shift The translation required to shift the image back to its center.
 * @return A batch of the output images.
 */
template <typename T, eInterpolationType InterpType>
std::vector<detail::BaseType<T>> GoldenRotate(std::vector<detail::BaseType<T>>& input, int batchSize, Size2D imageSize,
                                              double angle, double2 shift) {
    size_t numElements = batchSize * imageSize.w * imageSize.h * detail::NumElements<T>;
    std::vector<detail::BaseType<T>> output(numElements);

    T borderVal = detail::RangeCast<T>(make_float4(0.0f, 0.0f, 0.0f, 0.0f));

    ImageWrapper<T> outputWrapper(output, batchSize, imageSize.w, imageSize.h);
    InterpolationWrapper<T, eBorderType::BORDER_TYPE_CONSTANT, InterpType> inputWrapper(
        BorderWrapper<T, eBorderType::BORDER_TYPE_CONSTANT>(ImageWrapper<T>(input, batchSize, imageSize.w, imageSize.h),
                                                            borderVal));

    /**
     * Affine warp for a combined rotation and translate looks like the following when in its inverse representation:
     * [[cos(angle), sin(angle), shiftX],
     *  [-sin(angle), cos(angle) , shiftY]]
     *
     * To perform a rotation, we must map from our output space to our input space by multiplying each point by the
     * inverted form of the matrix (which is displayed above).
     */

    double angleRad = angle * (M_PI / 180.0f);
    // clang-format off
    std::array<double, 6> mat = {
        cosf(angleRad), sinf(angleRad), static_cast<float>(shift.x),
        -sin(angleRad),  cos(angleRad),   static_cast<float>(shift.y)
    };

    for (int b = 0; b < batchSize; b++) {
        for (int y = 0; y < imageSize.h; y++) {
            for (int x = 0; x < imageSize.w; x++) {
                const double shiftX = x - mat[2];
                const double shiftY = y - mat[5];
                
                const float srcX = static_cast<float>(shiftX * mat[0] + shiftY * -mat[1]);
                const float srcY = static_cast<float>(shiftX * -mat[3] + shiftY * mat[4]);

                outputWrapper.at(b, y, x, 0) = inputWrapper.at(b, srcY, srcX, 0);
            }
        }
    }

    return output;
}

/**
 * @brief Tests correctness for the roccv::Rotate operator by comparing results against a golden implementation.
 * 
 * @tparam T The image datatype.
 * @tparam InterpType The interpolation type to use.
 * @param batchSize The number of images in the batch.
 * @param imageSize The size of each image in the batch.
 * @param format The format of the images. This must be compatible with the provided image datatype T.
 * @param angle The angle in degrees to rotate counter clockwise.
 * @param device The device to run this correctness test on.
 */
template <typename T, eInterpolationType InterpType>
void TestCorrectness(int batchSize, Size2D imageSize, ImageFormat format, double angle, eDeviceType device) {
    std::vector<detail::BaseType<T>> input(batchSize * imageSize.w * imageSize.h * format.channels());
    FillVector(input);

    Tensor inputTensor(batchSize, imageSize, format, device);
    Tensor outputTensor(batchSize, imageSize, format, device);

    CopyVectorIntoTensor(inputTensor, input);

    // Compute center shift required for rotation
    double centerX = (imageSize.w - 1) / 2.0;
    double centerY = (imageSize.h - 1) / 2.0;
    double2 shift = ComputeCenterShift(centerX, centerY, angle);

    // Compute actual results using roccv::Rotate
    hipStream_t stream;
    HIP_VALIDATE_NO_ERRORS(hipStreamCreate(&stream));
    Rotate op;
    op(stream, inputTensor, outputTensor, angle, shift, InterpType, device);
    HIP_VALIDATE_NO_ERRORS(hipStreamSynchronize(stream));
    HIP_VALIDATE_NO_ERRORS(hipStreamDestroy(stream));

    std::vector<detail::BaseType<T>> actualResults(outputTensor.shape().size());
    CopyTensorIntoVector(actualResults, outputTensor);

    // Compute golden results
    std::vector<detail::BaseType<T>> goldenResults =
        GoldenRotate<T, InterpType>(input, batchSize, imageSize, angle, shift);

    // Compare actual and golden results
    CompareVectorsNear(actualResults, goldenResults, 1.0E-5);
}

}  // namespace

eTestStatusType test_op_rotate(int argc, char** argv) {
    TEST_CASES_BEGIN();

    // clang-format off

    // GPU Tests
    // U8
    TEST_CASE((TestCorrectness<uchar1, eInterpolationType::INTERP_TYPE_LINEAR>(1, {56, 78}, FMT_U8, 270.0, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<uchar3, eInterpolationType::INTERP_TYPE_LINEAR>(3, {34, 50}, FMT_RGB8, 90.0, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<uchar4, eInterpolationType::INTERP_TYPE_LINEAR>(5, {86, 23}, FMT_RGBA8, -180.0, eDeviceType::GPU)));

    // S8
    TEST_CASE((TestCorrectness<char1, eInterpolationType::INTERP_TYPE_LINEAR>(1, {56, 78}, FMT_S8, 270.0, eDeviceType::GPU)));

    // U16
    TEST_CASE((TestCorrectness<ushort1, eInterpolationType::INTERP_TYPE_LINEAR>(1, {56, 78}, FMT_U16, 270.0, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<ushort3, eInterpolationType::INTERP_TYPE_LINEAR>(3, {34, 50}, FMT_RGB16, 90.0, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<ushort4, eInterpolationType::INTERP_TYPE_LINEAR>(5, {86, 23}, FMT_RGBA16, -180.0, eDeviceType::GPU)));
    
    // S16
    TEST_CASE((TestCorrectness<short1, eInterpolationType::INTERP_TYPE_LINEAR>(1, {56, 78}, FMT_S16, 270.0, eDeviceType::GPU)));

    // U32
    TEST_CASE((TestCorrectness<uint1, eInterpolationType::INTERP_TYPE_LINEAR>(1, {56, 78}, FMT_U32, 270.0, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<uint3, eInterpolationType::INTERP_TYPE_LINEAR>(3, {34, 50}, FMT_RGB32, 90.0, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<uint4, eInterpolationType::INTERP_TYPE_LINEAR>(5, {86, 23}, FMT_RGBA32, -180.0, eDeviceType::GPU)));

    // S32
    TEST_CASE((TestCorrectness<int1, eInterpolationType::INTERP_TYPE_LINEAR>(1, {56, 78}, FMT_S32, 270.0, eDeviceType::GPU)));

    // F32
    TEST_CASE((TestCorrectness<float1, eInterpolationType::INTERP_TYPE_LINEAR>(1, {56, 78}, FMT_F32, 270.0, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<float3, eInterpolationType::INTERP_TYPE_LINEAR>(3, {34, 50}, FMT_RGBf32, 90.0, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<float4, eInterpolationType::INTERP_TYPE_LINEAR>(5, {86, 23}, FMT_RGBAf32, -180.0, eDeviceType::GPU)));

    // F64
    TEST_CASE((TestCorrectness<double1, eInterpolationType::INTERP_TYPE_LINEAR>(1, {56, 78}, FMT_F64, 270.0, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<double3, eInterpolationType::INTERP_TYPE_LINEAR>(3, {34, 50}, FMT_RGBf64, 90.0, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<double4, eInterpolationType::INTERP_TYPE_LINEAR>(5, {86, 23}, FMT_RGBAf64, -180.0, eDeviceType::GPU)));

    // CPU Tests
    // U8
    TEST_CASE((TestCorrectness<uchar1, eInterpolationType::INTERP_TYPE_LINEAR>(1, {56, 78}, FMT_U8, 270.0, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<uchar3, eInterpolationType::INTERP_TYPE_LINEAR>(3, {34, 50}, FMT_RGB8, 90.0, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<uchar4, eInterpolationType::INTERP_TYPE_LINEAR>(5, {86, 23}, FMT_RGBA8, -180.0, eDeviceType::CPU)));

    // S8
    TEST_CASE((TestCorrectness<char1, eInterpolationType::INTERP_TYPE_LINEAR>(1, {56, 78}, FMT_S8, 270.0, eDeviceType::CPU)));

    // U16
    TEST_CASE((TestCorrectness<ushort1, eInterpolationType::INTERP_TYPE_LINEAR>(1, {56, 78}, FMT_U16, 270.0, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<ushort3, eInterpolationType::INTERP_TYPE_LINEAR>(3, {34, 50}, FMT_RGB16, 90.0, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<ushort4, eInterpolationType::INTERP_TYPE_LINEAR>(5, {86, 23}, FMT_RGBA16, -180.0, eDeviceType::CPU)));
    
    // S16
    TEST_CASE((TestCorrectness<short1, eInterpolationType::INTERP_TYPE_LINEAR>(1, {56, 78}, FMT_S16, 270.0, eDeviceType::CPU)));

    // U32
    TEST_CASE((TestCorrectness<uint1, eInterpolationType::INTERP_TYPE_LINEAR>(1, {56, 78}, FMT_U32, 270.0, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<uint3, eInterpolationType::INTERP_TYPE_LINEAR>(3, {34, 50}, FMT_RGB32, 90.0, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<uint4, eInterpolationType::INTERP_TYPE_LINEAR>(5, {86, 23}, FMT_RGBA32, -180.0, eDeviceType::CPU)));

    // S32
    TEST_CASE((TestCorrectness<int1, eInterpolationType::INTERP_TYPE_LINEAR>(1, {56, 78}, FMT_S32, 270.0, eDeviceType::CPU)));

    // F32
    TEST_CASE((TestCorrectness<float1, eInterpolationType::INTERP_TYPE_LINEAR>(1, {56, 78}, FMT_F32, 270.0, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<float3, eInterpolationType::INTERP_TYPE_LINEAR>(3, {34, 50}, FMT_RGBf32, 90.0, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<float4, eInterpolationType::INTERP_TYPE_LINEAR>(5, {86, 23}, FMT_RGBAf32, -180.0, eDeviceType::CPU)));

    // F64
    TEST_CASE((TestCorrectness<double1, eInterpolationType::INTERP_TYPE_LINEAR>(1, {56, 78}, FMT_F64, 270.0, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<double3, eInterpolationType::INTERP_TYPE_LINEAR>(3, {34, 50}, FMT_RGBf64, 90.0, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<double4, eInterpolationType::INTERP_TYPE_LINEAR>(5, {86, 23}, FMT_RGBAf64, -180.0, eDeviceType::CPU)));

    // clang-format on

    TEST_CASES_END();
}