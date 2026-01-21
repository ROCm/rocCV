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

#include <chrono>
#include <core/tensor.hpp>
#include <random>
#include <vector>

namespace roccvbench {

static constexpr int NUM_WARMUP_RUNS = 5;

/**
 * @brief Generates a one-dimensional vector of the given size and type.
 *
 * @tparam T Datatype to fill the vector with.
 * @param size Size of the vector.
 * @return A vector of type T with random data.
 */
template <typename T>
std::vector<T> RandVector(size_t size) {
    std::random_device dev;
    std::mt19937 gen(dev());
    std::vector<T> result(size);

    if constexpr (std::is_floating_point_v<T>) {
        std::uniform_real_distribution<T> dist(0.0f, 1.0f);
        for (size_t i = 0; i < size; i++) {
            result[i] = dist(gen);
        }
    } else if constexpr (std::is_integral_v<T>) {
        std::uniform_int_distribution<int64_t> dist(std::numeric_limits<T>().min(), std::numeric_limits<T>().max());
        for (size_t i = 0; i < size; i++) {
            result[i] = static_cast<T>(dist(gen));
        }
    } else {
        static_assert(false, "Unsupported data type for random vector fill.\n");
    }

    return result;
}

/**
 * @brief Records the execution time in milliseconds of a block of code <code> by running it <numRuns> times and taking
 * the mean of the results. The resulting mean is written to <executionTime>.
 *
 */
#define ROCCV_BENCH_RECORD_BLOCK(code, executionTime, numRuns)                                                  \
    {                                                                                                           \
        double totalExecutionTime = 0.0;                                                                        \
        int numValidRuns = 0;                                                                                   \
        for (int i = 0; i < numRuns; i++) {                                                                     \
            auto blockStart = std::chrono::high_resolution_clock::now();                                        \
            code;                                                                                               \
            auto blockEnd = std::chrono::high_resolution_clock::now();                                          \
            if (i >= roccvbench::NUM_WARMUP_RUNS) {                                                             \
                totalExecutionTime += std::chrono::duration<double, std::milli>(blockEnd - blockStart).count(); \
                numValidRuns++;                                                                                 \
            }                                                                                                   \
        }                                                                                                       \
        executionTime = totalExecutionTime / numValidRuns;                                                      \
    }

}  // namespace roccvbench