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

#include <core/detail/allocators/default_allocator.hpp>

#include "test_helpers.hpp"

using namespace roccv;
using namespace roccv::tests;

namespace {

/**
 * @brief Tests that DefaultAllocator correctly rejects invalid arguments.
 */
void TestNegative() {
    // Test alignment which is not a power of 2
    {
        DefaultAllocator allocator;
        EXPECT_EXCEPTION(allocator.allocHostMem(1024, 3), eStatusType::INVALID_VALUE);
    }
}

/**
 * @brief Tests the correctness of the DefaultAllocator memory allocation and free routines.
 */
void TestCorrectness() {
    // Test basic host memory allocation and free
    {
        DefaultAllocator allocator;
        void* ptr = allocator.allocHostMem(1024);
        EXPECT_TRUE(ptr != nullptr);
        allocator.freeHostMem(ptr);
    }

    // Test basic device memory allocation and free
    {
        DefaultAllocator allocator;
        void* ptr = allocator.allocHipMem(1024);
        EXPECT_TRUE(ptr != nullptr);
        allocator.freeHipMem(ptr);
    }

    // Test basic pinned host memory allocation and free
    {
        DefaultAllocator allocator;
        void* ptr = allocator.allocHostPinnedMem(1024);
        EXPECT_TRUE(ptr != nullptr);
        allocator.freeHostPinnedMem(ptr);
    }

    // Test alignment parameter for host memory allocation
    {
        DefaultAllocator allocator;
        void* ptr = allocator.allocHostMem(1024, 16);
        EXPECT_TRUE(ptr != nullptr);
        EXPECT_TRUE(reinterpret_cast<uintptr_t>(ptr) % 16 == 0);
        allocator.freeHostMem(ptr);
    }
}
}  // namespace

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;
    TEST_CASES_BEGIN();

    TEST_CASE(TestCorrectness());
    TEST_CASE(TestNegative());

    TEST_CASES_END();
}