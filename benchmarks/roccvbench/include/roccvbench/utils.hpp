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

#pragma once

#include <core/hip_assert.h>

#include <chrono>

namespace roccv {
class Tensor;
}

namespace roccvbench {

/**
 * @brief Fills a tensor with random values.
 *
 * @param tensor The tensor to fill with random values.
 */
extern void FillTensor(const roccv::Tensor& tensor);

/**
 * @brief Records execution time of a function, ensuring synchronization using HIP events.
 */
#define ROCCV_BENCH_RECORD_EXECUTION_TIME_HIP(func, executionTime, numRuns)                                     \
    {                                                                                                           \
        for (int i = 0; i < numRuns; i++) {                                                                     \
            hipEvent_t begin, end;                                                                              \
            hipEventCreate(&begin);                                                                             \
            hipEventCreate(&end);                                                                               \
                                                                                                                \
            hipEventRecord(begin);                                                                              \
            auto kernelBegin = std::chrono::high_resolution_clock::now();                                       \
            func;                                                                                               \
            hipEventRecord(end);                                                                                \
            hipEventSynchronize(end);                                                                           \
            auto kernelEnd = std::chrono::high_resolution_clock::now();                                         \
                                                                                                                \
            hipEventDestroy(begin);                                                                             \
            hipEventDestroy(end);                                                                               \
                                                                                                                \
            double kernelDuration = std::chrono::duration<double, std::milli>(kernelEnd - kernelBegin).count(); \
                                                                                                                \
            executionTime += kernelDuration / numRuns;                                                          \
        }                                                                                                       \
    }

/**
 * @brief Records execution time of a host-side function call.
 */
#define ROCCV_BENCH_RECORD_EXECUTION_TIME_HOST(func, executionTime, numRuns)                      \
    {                                                                                             \
        for (int i = 0; i < numRuns; i++) {                                                       \
            auto begin = std::chrono::high_resolution_clock::now();                               \
            func;                                                                                 \
            auto end = std::chrono::high_resolution_clock::now();                                 \
            double funcDuration = std::chrono::duration<double, std::milli>(end - begin).count(); \
            executionTime += funcDuration / numRuns;                                              \
        }                                                                                         \
    }

}  // namespace roccvbench