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
        EXPECT_EXCEPTION(TensorShape({1, 10}, "N"), eStatusType::OUT_OF_BOUNDS);
        TensorShape shape({10}, "N");
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
        Tensor tensor(TensorShape({1, 2, 3}, "HWC"), DataType(DATA_TYPE_U8));
        EXPECT_EXCEPTION(tensor.reshape(TensorShape({1, 1, 2, 2}, "NHWC")), eStatusType::INVALID_VALUE);

        // Should not be able to reshape tensor into another view which would result in a different number of bytes in
        // the underlying memory.
        EXPECT_EXCEPTION(tensor.reshape(TensorShape({1, 1, 2, 3}, "NHWC"), DataType(DATA_TYPE_S16)),
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
        Tensor tensor(TensorShape({1, 720, 480, 3}, "NHWC"), DataType(DATA_TYPE_U8));
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
        Tensor reshapedTensor = tensor.reshape(TensorShape({720, 480, 3}, "HWC"));
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
        Tensor tensor(TensorShape({1, 5, 4}, "NWC"), DataType(DATA_TYPE_S16));
        Tensor reshapedTensor = tensor.reshape(TensorShape({1, 5}, "NW"), DataType(DATA_TYPE_4S16));
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
 * @brief Ensures that wrapping external data via TensorWrapData does not take ownership of the underlying buffer.
 *
 * The buffer must remain valid and intact after the wrapping Tensor (and any views derived from it) are destroyed.
 */
void TestTensorWrapNonOwning() {
    TensorShape shape({1, 2, 2, 3}, "NHWC");
    DataType dtype(DATA_TYPE_U8);
    size_t numElems = shape.size();

    // Test owns this buffer for the entire duration of the test.
    auto* buffer = new uint8_t[numElems];
    for (size_t i = 0; i < numElems; i++) {
        buffer[i] = static_cast<uint8_t>(i);
    }

    {
        TensorDataStrided::Buffer buf;
        buf.basePtr = buffer;
        buf.strides = Tensor::CalcStrides(shape, dtype);
        TensorDataStridedHost data(shape, dtype, buf);

        // Wrap with no cleanup function: this must produce a non-owning view.
        Tensor tensor = TensorWrapData(data);
        EXPECT_TRUE(tensor.exportData<TensorDataStrided>().basePtr() == buffer);

        // A reshaped view shares the same underlying storage; destroying both must still not free the buffer.
        Tensor view = tensor.reshape(TensorShape({2, 2, 3}, "HWC"));
        EXPECT_TRUE(view.exportData<TensorDataStrided>().basePtr() == buffer);
    }

    // The buffer must still be intact (untouched by tensor destruction).
    bool intact = true;
    for (size_t i = 0; i < numElems; i++) {
        if (buffer[i] != static_cast<uint8_t>(i)) {
            intact = false;
            break;
        }
    }
    EXPECT_TRUE(intact);

    // The test still owns the buffer and is responsible for freeing it. A double-free here would indicate the tensor
    // incorrectly took ownership.
    delete[] buffer;
}

/**
 * @brief Ensures that a cleanup function provided to TensorWrapData is invoked exactly once, when the last reference to
 * the wrapped data is destroyed.
 */
void TestTensorWrapCleanup() {
    TensorShape shape({1, 2, 2, 3}, "NHWC");
    DataType dtype(DATA_TYPE_U8);

    auto* buffer = new uint8_t[shape.size()];
    int cleanupCalls = 0;

    {
        TensorDataStrided::Buffer buf;
        buf.basePtr = buffer;
        buf.strides = Tensor::CalcStrides(shape, dtype);
        TensorDataStridedHost data(shape, dtype, buf);

        Tensor tensor = TensorWrapData(data, [&cleanupCalls](const TensorData&) { cleanupCalls++; });

        // A view shares the same storage, so the cleanup must not fire until both references are dropped.
        Tensor view = tensor.reshape(TensorShape({2, 2, 3}, "HWC"));
        EXPECT_EQ(cleanupCalls, 0);
    }

    // Both the tensor and its view have gone out of scope: cleanup must have run exactly once.
    EXPECT_EQ(cleanupCalls, 1);

    delete[] buffer;
}

/**
 * @brief Ensures TensorData::cast preserves the concrete device when casting to a non-leaf type.
 *
 * Casting to the base TensorDataStrided (as exportData<TensorDataStrided>() does throughout the codebase) must report
 * the same device as the source tensor, not a hardcoded default.
 */
void TestTensorDataCastDevicePropagation() {
    // Host tensor: exporting/casting to the base strided type must still report CPU.
    {
        Tensor tensor(TensorShape({1, 2, 2, 3}, "NHWC"), DataType(DATA_TYPE_U8), eDeviceType::CPU);
        auto data = tensor.exportData<TensorDataStrided>();
        EXPECT_TRUE(data.device() == eDeviceType::CPU);
    }

    // Device tensor: exporting/casting to the base strided type must still report GPU.
    {
        Tensor tensor(TensorShape({1, 2, 2, 3}, "NHWC"), DataType(DATA_TYPE_U8), eDeviceType::GPU);
        auto data = tensor.exportData<TensorDataStrided>();
        EXPECT_TRUE(data.device() == eDeviceType::GPU);
    }

    // Direct cast: a host strided descriptor must remain CPU when viewed as the base type, and must refuse a cast to an
    // incompatible (device) leaf type.
    {
        TensorShape shape({1, 2, 2, 3}, "NHWC");
        DataType dtype(DATA_TYPE_U8);
        TensorDataStrided::Buffer buf;
        buf.basePtr = nullptr;
        buf.strides = Tensor::CalcStrides(shape, dtype);
        TensorDataStridedHost host(shape, dtype, buf);

        auto asStrided = host.cast<TensorDataStrided>();
        EXPECT_TRUE(asStrided.has_value());
        EXPECT_TRUE(asStrided->device() == eDeviceType::CPU);

        EXPECT_FALSE(host.cast<TensorDataStridedHip>().has_value());
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

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;
    TEST_CASES_BEGIN();

    // Negative tests
    TEST_CASE(TestNegativeTensorShape());
    TEST_CASE(TestNegativeTensor());

    // Correctness tests
    TEST_CASE(TestTensorCorrectness());

    // Wrapped-data ownership tests
    TEST_CASE(TestTensorWrapNonOwning());
    TEST_CASE(TestTensorWrapCleanup());

    // TensorData cast device propagation
    TEST_CASE(TestTensorDataCastDevicePropagation());

    // Stride calculation tests
    // clang-format off
    TEST_CASE(TestTensorStrideCalculation(TensorShape({1, 2, 4, 4}, "NHWC"), DataType(DATA_TYPE_U8)));
    TEST_CASE(TestTensorStrideCalculation(TensorShape({2, 16, 4, 1}, "NHWC"), DataType(DATA_TYPE_F32)));
    TEST_CASE(TestTensorStrideCalculation(TensorShape({3, 54, 4, 3}, "NHWC"), DataType(DATA_TYPE_S16)));
    TEST_CASE(TestTensorStrideCalculation(TensorShape({4, 3, 4, 4}, "NHWC"), DataType(DATA_TYPE_S8)));
    TEST_CASE(TestTensorStrideCalculation(TensorShape({6, 12, 4, 3}, "NHWC"), DataType(DATA_TYPE_S32)));
    TEST_CASE(TestTensorStrideCalculation(TensorShape({8, 45, 4, 1}, "NHWC"), DataType(DATA_TYPE_U32)));

    TEST_CASE(TestTensorStrideCalculation(TensorShape({2, 4, 4}, "HWC"), DataType(DATA_TYPE_U8)));
    TEST_CASE(TestTensorStrideCalculation(TensorShape({16, 4, 1}, "HWC"), DataType(DATA_TYPE_F32)));
    TEST_CASE(TestTensorStrideCalculation(TensorShape({54, 4, 3}, "HWC"), DataType(DATA_TYPE_S16)));
    TEST_CASE(TestTensorStrideCalculation(TensorShape({3, 4, 4}, "HWC"), DataType(DATA_TYPE_S8)));
    TEST_CASE(TestTensorStrideCalculation(TensorShape({12, 4, 3}, "HWC"), DataType(DATA_TYPE_S32)));
    TEST_CASE(TestTensorStrideCalculation(TensorShape({45, 4, 1}, "HWC"), DataType(DATA_TYPE_U32)));
    // clang-format on

    TEST_CASES_END();
}