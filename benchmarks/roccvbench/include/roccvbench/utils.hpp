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

#pragma once

#include <chrono>
#include <random>
#include <stdexcept>
#include <vector>

#include "structs.hpp"

namespace roccvbench {

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
 * @brief Gets the value of a parameter from a list of parameters.
 *
 * @tparam T The type of the value to get.
 * @param params The list of parameters to get the value from.
 * @param key The key of the parameter to get the value from.
 * @return The value of the parameter.
 */
template <typename T>
T GetParamValue(const BenchmarkParamsList& params, const std::string& key) {
    for (const auto& param : params) {
        if (param.key == key) {
            return std::any_cast<T>(param.value);
        }
    }
    throw std::runtime_error("Parameter not found: " + key);
}

/**
 * @brief Records the execution time in seconds of a block of code <code> by running it <numRuns> times and taking
 * the mean of the results. The resulting mean is written to <executionTime>.
 *
 */
#define ROCCV_BENCH_RECORD_BLOCK(code, executionTime, numRuns, warmupRuns)                                      \
    {                                                                                                           \
        double totalExecutionTime = 0.0;                                                                        \
        int numValidRuns = 0;                                                                                   \
        for (int i = 0; i < numRuns + warmupRuns; i++) {                                                        \
            auto blockStart = std::chrono::high_resolution_clock::now();                                        \
            code;                                                                                               \
            auto blockEnd = std::chrono::high_resolution_clock::now();                                          \
            if (i >= warmupRuns) {                                                                              \
                totalExecutionTime += std::chrono::duration<double, std::milli>(blockEnd - blockStart).count(); \
                numValidRuns++;                                                                                 \
            }                                                                                                   \
        }                                                                                                       \
        executionTime = (totalExecutionTime / numValidRuns) / 1000.0;                                           \
    }

}  // namespace roccvbench