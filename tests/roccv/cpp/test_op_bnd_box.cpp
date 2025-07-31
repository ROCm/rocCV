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
#include <iostream>
#include <core/detail/casting.hpp>
#include <core/detail/type_traits.hpp>
#include <core/wrappers/image_wrapper.hpp>
#include <op_bnd_box.hpp>
#include "common/math_vector.hpp"
#include "test_helpers.hpp"

using namespace roccv;
using namespace roccv::tests;

namespace {

bool isPixelInBox(float ix, float iy, float left, float right, float top, float bottom) {
    return (ix > left) && (ix < right) && (iy > top) && (iy < bottom);
}

template <typename T>
void alphaBlend(T &color, const T &color_in) {
    // Working type for internal pixel format, which is float and has 4 channels.
    using WorkType = float4;
    WorkType fgColor = detail::RangeCast<WorkType>(color_in);
    WorkType bgColor = detail::RangeCast<WorkType>(color);
    float fgAlpha = fgColor.w;
    float bgAlpha = bgColor.w;
    float blendAlpha = fgAlpha + bgAlpha * (1 - fgAlpha);

    WorkType blendColor = (fgColor * fgAlpha + bgColor * bgAlpha * (1 - fgAlpha)) / blendAlpha;
    blendColor.w = blendAlpha;
    color = detail::RangeCast<T>(blendColor);
}

template <typename T>
void shadeBBRect(const Rect_t &rect, int ix, int iy, T *out_color) {
    if (rect.bordered) {
        if (!isPixelInBox(ix, iy, rect.i_left, rect.i_right, rect.i_top, rect.i_bottom) &&
            isPixelInBox(ix, iy, rect.o_left, rect.o_right, rect.o_top, rect.o_bottom)) {
            alphaBlend<T>(out_color[0], detail::RangeCast<T>(rect.color));
        }
    } else {
        if (isPixelInBox(ix, iy, rect.o_left, rect.o_right, rect.o_top, rect.o_bottom)) {
            alphaBlend<T>(out_color[0], detail::RangeCast<T>(rect.color));
        }
    }
}

/**
 * @brief Verified golden C++ model for the bounding box operation.
 *
 * @tparam T Vectorized datatype of the image's pixels.
 * @tparam BT Base type of the image's data.
 * @param[in] input Input vector containing image data.
 * @param[out] output Output vector containing image data with drawn bounding boxes.
 * @param[in] batchSize The number of images in the batch.
 * @param[in] width Image width.
 * @param[in] height Image height.
 * @param[in] bboxes Bounding box array.
 * @return None.
 */
template <typename T, typename BT = detail::BaseType<T>>
void GenerateGoldenBndBox(std::vector<BT>& input, std::vector<BT>& output, int32_t batchSize, int32_t width, int32_t height, BndBoxes_t bboxes) {
    // Wrap input/output vectors for simplified data access
    ImageWrapper<T> src(input, batchSize, width, height);
    ImageWrapper<T> dst(output, batchSize, width, height);

    // Working type for internal pixel format, which has 4 channels.
    using WorkType = detail::MakeType<BT, 4>;

    std::vector<Rect_t> rects;
    BndBox op;
    op.generateRects(rects, bboxes, height, width);

    bool has_alpha;
    if constexpr (detail::NumElements<T> == 4) {
        has_alpha = true;
    } else {
        has_alpha = false;
    }

    for (int b_idx = 0; b_idx < batchSize; b_idx++) {
        for (int y_idx = 0; y_idx < height; y_idx++) {
            for (int x_idx = 0; x_idx < width; x_idx++) {
                WorkType shaded_pixel{0, 0, 0, 0};

                for (size_t i = 0; i < rects.size(); i++) {
                    Rect_t curr_rect = rects[i];
                    if (curr_rect.batch <= b_idx)
                        shadeBBRect<WorkType>(curr_rect, x_idx, y_idx, &shaded_pixel);
                }

                WorkType out_color = MathVector::fill(src.at(b_idx, y_idx, x_idx, 0));
                out_color.w = has_alpha ? out_color.w : (std::numeric_limits<BT>::max());

                if (shaded_pixel.w != 0)
                    alphaBlend<WorkType>(out_color, shaded_pixel);

                MathVector::trunc(out_color, &dst.at(b_idx, y_idx, x_idx, 0));
            }
        }
    }
}

/**
 * @brief Tests correctness of the bounding box operator, comparing it against a generated golden result.
 *
 * @tparam T Underlying datatype of the image's pixels.
 * @tparam BT Base type of the image data.
 * @param[in] batchSize Number of images in the batch.
 * @param[in] width Width of each image in the batch.
 * @param[in] height Height of each image in the batch.
 * @param[in] format Image format.
 * @param[in] device Device this correctness test should be run on.
 */
template <typename T, typename BT = detail::BaseType<T>>
void TestCorrectness(int batchSize, int width, int height, ImageFormat format, eDeviceType device) {
    // Create input and output tensor based on test parameters
    Tensor input(batchSize, {width, height}, format, device);
    Tensor output(batchSize, {width, height}, format, device);

    // Create a vector and fill it with random data.
    std::vector<BT> inputData(input.shape().size());
    FillVector(inputData);
    // Copy generated input data into input tensor
    CopyVectorIntoTensor(input, inputData);

    std::vector<int32_t> bboxes_size_vector(1, 3);
    std::vector<BndBox_t> bbox_vector(3);
    bbox_vector[0].box.x = width / 4;
    bbox_vector[0].box.y = height / 4;
    bbox_vector[0].box.width = width / 2;
    bbox_vector[0].box.height = height / 2;
    bbox_vector[0].thickness = 5;
    bbox_vector[0].borderColor = {0, 0, 255, 200};
    bbox_vector[0].fillColor = {0, 255, 0, 100};
    bbox_vector[1].box.x = width / 3;
    bbox_vector[1].box.y = height / 3;
    bbox_vector[1].box.width = width / 3 * 2;
    bbox_vector[1].box.height = height / 4;
    bbox_vector[1].thickness = -1;
    bbox_vector[1].borderColor = {90, 16, 181, 50};
    bbox_vector[2].box.x = -50;
    bbox_vector[2].box.y = (2 * height) / 3;
    bbox_vector[2].box.width = width + 50;
    bbox_vector[2].box.height = height / 3 + 50;
    bbox_vector[2].thickness = 0;
    bbox_vector[2].borderColor = {0, 0, 0, 50};
    bbox_vector[2].fillColor = {111, 159, 232, 150};
    BndBoxes_t bboxes{1, bboxes_size_vector, bbox_vector};

    hipStream_t stream;
    HIP_VALIDATE_NO_ERRORS(hipStreamCreate(&stream));
    BndBox op;
    op(stream, input, output, bboxes, device);
    HIP_VALIDATE_NO_ERRORS(hipStreamSynchronize(stream));
    HIP_VALIDATE_NO_ERRORS(hipStreamDestroy(stream));

    // Copy data from output tensor into a host allocated vector
    std::vector<BT> result(output.shape().size());
    CopyTensorIntoVector(result, output);

    // Calculate golden output reference
    std::vector<BT> ref(output.shape().size());
    GenerateGoldenBndBox<T>(inputData, ref, batchSize, width, height, bboxes);

    // Compare data in actual output versus the generated golden reference image
    CompareVectorsNear(result, ref);
}

}  // namespace

eTestStatusType test_op_bnd_box(int argc, char **argv) {
    TEST_CASES_BEGIN();

    // GPU correctness tests
    TEST_CASE(TestCorrectness<uchar3>(1, 360, 240, FMT_RGB8, eDeviceType::GPU));
    TEST_CASE(TestCorrectness<uchar3>(2, 256, 256, FMT_RGB8, eDeviceType::GPU));

    TEST_CASE(TestCorrectness<uchar4>(1, 360, 240, FMT_RGBA8, eDeviceType::GPU));
    TEST_CASE(TestCorrectness<uchar4>(3, 200, 100, FMT_RGBA8, eDeviceType::GPU));

    TEST_CASE(TestCorrectness<char3>(1, 360, 240, FMT_RGBs8, eDeviceType::GPU));
    TEST_CASE(TestCorrectness<char3>(4, 100, 80, FMT_RGBs8, eDeviceType::GPU));

    TEST_CASE(TestCorrectness<char4>(1, 360, 240, FMT_RGBAs8, eDeviceType::GPU));
    TEST_CASE(TestCorrectness<char4>(5, 80, 40, FMT_RGBAs8, eDeviceType::GPU));

    // CPU correctness tests
    TEST_CASE(TestCorrectness<uchar3>(1, 360, 240, FMT_RGB8, eDeviceType::CPU));
    TEST_CASE(TestCorrectness<uchar3>(2, 256, 256, FMT_RGB8, eDeviceType::CPU));

    TEST_CASE(TestCorrectness<uchar4>(1, 360, 240, FMT_RGBA8, eDeviceType::CPU));
    TEST_CASE(TestCorrectness<uchar4>(3, 200, 100, FMT_RGBA8, eDeviceType::CPU));

    TEST_CASE(TestCorrectness<char3>(1, 360, 240, FMT_RGBs8, eDeviceType::CPU));
    TEST_CASE(TestCorrectness<char3>(4, 100, 80, FMT_RGBs8, eDeviceType::CPU));

    TEST_CASE(TestCorrectness<char4>(1, 360, 240, FMT_RGBAs8, eDeviceType::CPU));
    TEST_CASE(TestCorrectness<char4>(5, 80, 40, FMT_RGBAs8, eDeviceType::CPU));

    TEST_CASES_END();
}
