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
    shape[layout.batch_index()] = batchSize;
    shape[layout.height_index()] = height;
    shape[layout.width_index()] = width;
    shape[layout.channels_index()] = channels;
    return shape;
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

    // TODO: Support HWC <-> CHW conversions.

    for (int32_t b = 0; b < batchSize; b++) {
        for (int32_t y = 0; y < height; y++) {
            for (int32_t x = 0; x < width; x++) {
                for (int32_t c = 0; c < channels; c++) {
                    if (inLayout.elayout() == eTensorLayout::TENSOR_LAYOUT_NHWC &&
                        outLayout.elayout() == eTensorLayout::TENSOR_LAYOUT_NCHW) {
                        // NHWC to NCHW
                        outputWrapper.at(b, c, y, x) = inputWrapper.at(b, y, x, c);
                    }

                    else if (inLayout.elayout() == eTensorLayout::TENSOR_LAYOUT_NCHW &&
                             outLayout.elayout() == eTensorLayout::TENSOR_LAYOUT_NHWC) {
                        // NCHW to NHWC
                        outputWrapper.at(b, y, x, c) = inputWrapper.at(b, c, y, x);
                    }

                    else {
                        throw Exception("Invalid layout conversion requested in GoldenReformat",
                                        eStatusType::INVALID_VALUE);
                    }
                }
            }
        }
    }

    return outputData;
}
}  // namespace

int main(int argc, char** argv) {
    TEST_CASES_BEGIN();
    TEST_CASES_END();
}