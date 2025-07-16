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

#include <core/tensor.hpp>

#include "test_helpers.hpp"

using namespace roccv;
using namespace roccv::tests;

namespace {

/**
 * @brief Golden model for calculating strides given a TensorShape and a datatype.
 *
 * @param shape The tensor's shape.
 * @param dtype The datatype of the tensor.
 * @return A list of strides for each dimension of the given shape.
 */
std::vector<int64_t> CalculateStrides(const TensorShape& shape, const DataType& dtype) {
    std::vector<int64_t> strides(shape.layout().rank());

    // Strides are calculated byte-wise. Therefore, the highest dimension will refer to the stride between singular
    // elements (which, in turn, is the number of bytes per said element).
    strides[shape.layout().rank() - 1] = dtype.size();
    for (int i = shape.layout().rank() - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    return strides;
}

/**
 * @brief Negative tests related to TensorShape.
 *
 */
void TestNegativeTensorShape() {
    // Ensure TensorShape errors are thrown
    {
        EXPECT_EXCEPTION(TensorShape(TensorLayout(TENSOR_LAYOUT_N), {1, 10}), eStatusType::OUT_OF_BOUNDS);
        TensorShape shape(TensorLayout(TENSOR_LAYOUT_N), {10});
        EXPECT_EXCEPTION(shape[1], eStatusType::OUT_OF_BOUNDS);
    }
}

/**
 * @brief Negative tests related to the Tensor object.
 *
 */
void TestNegativeTensor() {
    {
        // Should not be able to reshape tensor into another view with a differing number of elements
        Tensor tensor(TensorShape(TensorLayout(TENSOR_LAYOUT_HWC), {1, 2, 3}), DataType(DATA_TYPE_U8));
        EXPECT_EXCEPTION(tensor.reshape(TensorShape(TensorLayout(TENSOR_LAYOUT_NHWC), {1, 1, 2, 2})),
                         eStatusType::INVALID_VALUE);

        // Should not be able to reshape tensor into another view which would result in a different number of bytes in
        // the underlying memory.
        EXPECT_EXCEPTION(
            tensor.reshape(TensorShape(TensorLayout(TENSOR_LAYOUT_NHWC), {1, 1, 2, 3}), DataType(DATA_TYPE_S16)),
            eStatusType::INVALID_VALUE);
    }
}

/**
 * @brief General correctness tests related to Tensor construction and manipulation.
 *
 */
void TestTensorCorrectness() {
    // Regular tensor construction
    {
        Tensor tensor(TensorShape(TensorLayout(TENSOR_LAYOUT_NHWC), {1, 720, 480, 3}), DataType(DATA_TYPE_U8));
        EXPECT_EQ(tensor.shape().size(), 1 * 720 * 480 * 3);
        EXPECT_EQ(tensor.dtype().size(), 1);
    }

    // Image-based tensor construction
    {
        Tensor tensor(4, {720, 480}, FMT_RGB8);
        EXPECT_EQ(tensor.shape().size(), 4 * 720 * 480 * 3);
        EXPECT_EQ(tensor.dtype().size(), 1);
    }

    // Tensor reshape: Change layout
    {
        // Reshape tensor from NHWC -> HWC layout
        Tensor tensor(1, {720, 480}, FMT_RGB8);
        Tensor reshapedTensor = tensor.reshape(TensorShape(TensorLayout(TENSOR_LAYOUT_HWC), {720, 480, 3}));
        EXPECT_EQ(reshapedTensor.rank(), 3);
        EXPECT_NE(reshapedTensor.rank(), tensor.rank());
        EXPECT_EQ(reshapedTensor.shape().size(), tensor.shape().size());

        // Ensure they are sharing the same underlying data
        auto data = tensor.exportData<TensorDataStrided>();
        auto dataReshaped = reshapedTensor.exportData<TensorDataStrided>();
        EXPECT_TRUE(data.basePtr() == dataReshaped.basePtr());
    }

    // Tensor reshape: Change layout and datatype
    {
        Tensor tensor(TensorShape(TensorLayout(TENSOR_LAYOUT_NWC), {1, 5, 4}), DataType(DATA_TYPE_S16));
        Tensor reshapedTensor =
            tensor.reshape(TensorShape(TensorLayout(TENSOR_LAYOUT_NW), {1, 5}), DataType(DATA_TYPE_4S16));
        EXPECT_NE(reshapedTensor.shape().size(), tensor.shape().size());
        EXPECT_NE(reshapedTensor.rank(), tensor.rank());
        EXPECT_EQ(reshapedTensor.rank(), 2);

        // Ensure they are sharing the same underlying data
        auto data = tensor.exportData<TensorDataStrided>();
        auto dataReshaped = reshapedTensor.exportData<TensorDataStrided>();
        EXPECT_TRUE(data.basePtr() == dataReshaped.basePtr());
    }
}

/**
 * @brief Tests internal stride calculations on Tensor construction.
 */
void TestTensorStrideCalculation(const TensorShape& shape, const DataType& dtype) {
    Tensor tensor(shape, dtype);
    std::vector<int64_t> expectedStrides = CalculateStrides(shape, dtype);
    std::vector<int64_t> actualStrides(tensor.rank());
    auto data = tensor.exportData<TensorDataStrided>();

    for (int i = 0; i < tensor.rank(); i++) {
        actualStrides[i] = data.stride(i);
    }

    EXPECT_VECTOR_EQ(actualStrides, expectedStrides);
}

}  // namespace

eTestStatusType test_tensor(int argc, char** argv) {
    TEST_CASES_BEGIN();

    // Negative tests
    TEST_CASE(TestNegativeTensorShape());
    TEST_CASE(TestNegativeTensor());

    // Correctness tests
    TEST_CASE(TestTensorCorrectness());

    // Stride calculation tests
    // clang-format off
    TEST_CASE(TestTensorStrideCalculation(TensorShape(TensorLayout(TENSOR_LAYOUT_NHWC), {1, 2, 4, 4}), DataType(DATA_TYPE_U8)));
    TEST_CASE(TestTensorStrideCalculation(TensorShape(TensorLayout(TENSOR_LAYOUT_NHWC), {2, 16, 4, 1}), DataType(DATA_TYPE_F32)));
    TEST_CASE(TestTensorStrideCalculation(TensorShape(TensorLayout(TENSOR_LAYOUT_NHWC), {3, 54, 4, 3}), DataType(DATA_TYPE_S16)));
    TEST_CASE(TestTensorStrideCalculation(TensorShape(TensorLayout(TENSOR_LAYOUT_NHWC), {4, 3, 4, 4}), DataType(DATA_TYPE_S8)));
    TEST_CASE(TestTensorStrideCalculation(TensorShape(TensorLayout(TENSOR_LAYOUT_NHWC), {6, 12, 4, 3}), DataType(DATA_TYPE_S32)));
    TEST_CASE(TestTensorStrideCalculation(TensorShape(TensorLayout(TENSOR_LAYOUT_NHWC), {8, 45, 4, 1}), DataType(DATA_TYPE_U32)));

    TEST_CASE(TestTensorStrideCalculation(TensorShape(TensorLayout(TENSOR_LAYOUT_HWC), {2, 4, 4}), DataType(DATA_TYPE_U8)));
    TEST_CASE(TestTensorStrideCalculation(TensorShape(TensorLayout(TENSOR_LAYOUT_HWC), {16, 4, 1}), DataType(DATA_TYPE_F32)));
    TEST_CASE(TestTensorStrideCalculation(TensorShape(TensorLayout(TENSOR_LAYOUT_HWC), {54, 4, 3}), DataType(DATA_TYPE_S16)));
    TEST_CASE(TestTensorStrideCalculation(TensorShape(TensorLayout(TENSOR_LAYOUT_HWC), {3, 4, 4}), DataType(DATA_TYPE_S8)));
    TEST_CASE(TestTensorStrideCalculation(TensorShape(TensorLayout(TENSOR_LAYOUT_HWC), {12, 4, 3}), DataType(DATA_TYPE_S32)));
    TEST_CASE(TestTensorStrideCalculation(TensorShape(TensorLayout(TENSOR_LAYOUT_HWC), {45, 4, 1}), DataType(DATA_TYPE_U32)));
    // clang-format on

    TEST_CASES_END();
}