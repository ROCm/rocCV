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

#include <core/tensor_shape.hpp>

#include "test_helpers.hpp"

using namespace roccv;
using namespace roccv::tests;

namespace {

/**
 * @brief Negative tests related to TensorShape.
 *
 */
void TestNegativeTensorShape() {
    // Test out of bounds index
    {
        TensorShape shape({1, 2, 3}, "NWC");
        EXPECT_EXCEPTION(shape[3], eStatusType::OUT_OF_BOUNDS);
        EXPECT_EXCEPTION(shape[-1], eStatusType::OUT_OF_BOUNDS);
    }

    // Test invalid shape/layout combination
    {
        EXPECT_EXCEPTION(TensorShape shape({1, 2, 3}, "NHWC"), eStatusType::OUT_OF_BOUNDS);
        EXPECT_EXCEPTION(TensorShape shape({1, 2}, "NWC"), eStatusType::OUT_OF_BOUNDS);
    }

    // Test shape creation with negative values
    {
        EXPECT_EXCEPTION(TensorShape shape({1, 2, -3}, "NWC"), eStatusType::OUT_OF_BOUNDS);
    }

    // Test shape creation with invalid layout string
    {
        EXPECT_EXCEPTION(TensorShape shape({1, 2, 3}, "NWCX"), eStatusType::INVALID_VALUE);
    }

    // Test permute operation with invalid layout
    {
        TensorShape shape({1, 2, 3}, "HWC");
        EXPECT_EXCEPTION(shape.permute(TensorLayout(TENSOR_LAYOUT_NCHW)), eStatusType::OUT_OF_BOUNDS);
    }
}

/**
 * @brief Correctness tests related to TensorShape.
 *
 */
void TestTensorShapeCorrectness() {
    // Test TensorShape construction
    {
        TensorShape shape({1, 2, 3}, "NWC");
        EXPECT_EQ(shape.size(), 1 * 2 * 3);
        EXPECT_EQ(shape[0], 1);
        EXPECT_EQ(shape[1], 2);
        EXPECT_EQ(shape[2], 3);
    }

    // Test TensorShape == operator
    {
        TensorShape shape1({1, 2, 3}, "NWC");
        TensorShape shape2({1, 2, 3}, "NWC");
        TensorShape shape3({3, 2, 1}, "NWC");
        EXPECT_TRUE(shape1 != shape3);
        EXPECT_TRUE(shape1 == shape2);
    }

    // Test TensorShape assignment operator
    {
        TensorShape shape1({1, 2, 3}, "NWC");
        TensorShape shape2({1, 2, 3}, "NWC");
        shape2 = shape1;
        EXPECT_TRUE(shape1 == shape2);
    }

    // Test TensorShape index operator
    {
        TensorShape shape({1, 2, 3, 4}, "NHWC");
        EXPECT_EQ(shape["N"], 1);
        EXPECT_EQ(shape["H"], 2);
        EXPECT_EQ(shape["W"], 3);
        EXPECT_EQ(shape["C"], 4);
        EXPECT_EXCEPTION(shape["X"], eStatusType::OUT_OF_BOUNDS);
    }

    // Test TensorShape permute operator
    {
        TensorShape shape({1, 2, 3, 4}, "NHWC");
        TensorShape permutedShape = shape.permute(TensorLayout(TENSOR_LAYOUT_NCHW));
        EXPECT_TRUE(permutedShape.layout() == eTensorLayout::TENSOR_LAYOUT_NCHW);
        EXPECT_EQ(permutedShape["N"], 1);
        EXPECT_EQ(permutedShape["C"], 4);
        EXPECT_EQ(permutedShape["H"], 2);
        EXPECT_EQ(permutedShape["W"], 3);
    }

    // Test TensorShape containsDim operator
    {
        TensorShape shape({1, 2, 3}, "HWC");
        EXPECT_TRUE(shape.containsDim("H"));
        EXPECT_TRUE(shape.containsDim("W"));
        EXPECT_TRUE(shape.containsDim("C"));
        EXPECT_FALSE(shape.containsDim("N"));
    }
}
}  // namespace

int main(int argc, char** argv) {
    TEST_CASES_BEGIN();

    TEST_CASE(TestTensorShapeCorrectness());
    TEST_CASE(TestNegativeTensorShape());

    TEST_CASES_END();
}