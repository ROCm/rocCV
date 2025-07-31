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
double2 ComputeCenterShift(const double centerX, const double centerY, const double angle) {
    double xShift = (1 - cos(angle * M_PI / 180)) * centerX - sin(angle * M_PI / 180) * centerY;
    double yShift = sin(angle * M_PI / 180) * centerX + (1 - cos(angle * M_PI / 180)) * centerY;
    return {xShift, yShift};
}

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
     * inverted form of the matrix (which is displayed above). We will represent our affine transformation as a 3x3
     * perspective transformation matrix with the bottom row set to [0, 0, 1].
     */

    float angleRad = static_cast<float>(angle) * (M_PI / 180.0f);
    // clang-format off
    std::array<float, 9> mat = {
        cosf(angleRad), sinf(angleRad), static_cast<float>(shift.x),
        -sin(angleRad),  cos(angleRad),   static_cast<float>(shift.y),
        0,              0,               1
    };

    for (int b = 0; b < batchSize; b++) {
        for (int y = 0; y < imageSize.h; y++) {
            for (int x = 0; x < imageSize.w; x++) {
                Point2D inputCoord = MatTransform({static_cast<float>(x), static_cast<float>(y)}, mat);
                outputWrapper.at(b, y, x, 0) = inputWrapper.at(b, inputCoord.y, inputCoord.x, 0);
            }
        }
    }

    return output;
}

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

    // printf("Got here.\n");
    cv::Mat actualMat(imageSize.h, imageSize.w, CV_8UC4, actualResults.data());
    cv::imwrite("actual.png", actualMat);

    cv::Mat expectedMat(imageSize.h, imageSize.w, CV_8UC4, goldenResults.data());
    cv::imwrite("expected.png", expectedMat);

    // Compare actual and golden results
    CompareVectorsNear(actualResults, goldenResults);
}

}  // namespace

eTestStatusType test_op_rotate(int argc, char** argv) {
    TEST_CASES_BEGIN();

    // clang-format off
    // TEST_CASE((TestCorrectness<uchar1, eInterpolationType::INTERP_TYPE_LINEAR>(1, {56, 78}, FMT_U8, 54.0, eDeviceType::GPU)));
    // TEST_CASE((TestCorrectness<uchar3, eInterpolationType::INTERP_TYPE_LINEAR>(3, {34, 50}, FMT_RGB8, 90.0, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<uchar4, eInterpolationType::INTERP_TYPE_LINEAR>(1, {86, 23}, FMT_RGBA8, -180.0, eDeviceType::GPU)));

    // TEST_CASE((TestCorrectness<uchar1, eInterpolationType::INTERP_TYPE_LINEAR>(1, {56, 78}, FMT_U8, 54.0, eDeviceType::CPU)));
    // TEST_CASE((TestCorrectness<uchar3, eInterpolationType::INTERP_TYPE_LINEAR>(3, {34, 50}, FMT_RGB8, 90.0, eDeviceType::CPU)));
    // TEST_CASE((TestCorrectness<uchar4, eInterpolationType::INTERP_TYPE_LINEAR>(5, {86, 23}, FMT_RGBA8, -180.0, eDeviceType::CPU)));

    // clang-format on

    TEST_CASES_END();
}