/*
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include <core/mem_alignment.hpp>

#include "test_helpers.hpp"

using namespace roccv;
using namespace roccv::tests;

namespace {

/**
 * @brief Tests the correctness of MemAlignment's getters, setters, and defaults.
 */
void TestMemAlignmentCorrectness() {
    // Test default construction
    {
        MemAlignment align;
        EXPECT_EQ(align.baseAddr(), 0);
        EXPECT_EQ(align.rowAddr(), 0);
    }

    // Test setting base address alignment
    {
        MemAlignment align;
        align.baseAddr(256);
        EXPECT_EQ(align.baseAddr(), 256);
        EXPECT_EQ(align.rowAddr(), 0);
    }

    // Test setting row address alignment
    {
        MemAlignment align;
        align.rowAddr(128);
        EXPECT_EQ(align.rowAddr(), 128);
        EXPECT_EQ(align.baseAddr(), 0);
    }

    // Test setting both base and row alignment via chaining
    {
        MemAlignment align;
        align.baseAddr(256).rowAddr(128);
        EXPECT_EQ(align.baseAddr(), 256);
        EXPECT_EQ(align.rowAddr(), 128);
    }

    // Test that setters return a reference to the same object
    {
        MemAlignment align;
        MemAlignment& ref = align.baseAddr(64);
        EXPECT_TRUE(&ref == &align);
    }

    // Test overwriting a previously set alignment value
    {
        MemAlignment align;
        align.baseAddr(64);
        align.baseAddr(32);
        EXPECT_EQ(align.baseAddr(), 32);
    }
}
}  // namespace

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;
    TEST_CASES_BEGIN();

    TEST_CASE(TestMemAlignmentCorrectness());

    TEST_CASES_END();
}
