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

#include <any>
#include <cstddef>
#include <format>
#include <functional>
#include <string>

#include "results.hpp"

namespace roccvbench {

/**
 * @brief Acts as the return value for benchmark units. Contains information pertaining to the benchmark results.
 *
 */
struct BenchmarkResults {
    double executionTime = 0.0;
    size_t readMemoryBytes = 0;
    size_t writtenMemoryBytes = 0;
};

/**
 * @brief Container for configuration options. Passed into all benchmark units.
 *
 */
struct BenchmarkConfig {
    int samples, width, height, runs, warmupRuns;
};

struct BenchmarkParam {
    std::string key;
    std::any value;
    std::string strValue;
};

using BenchmarkParamsList = std::vector<BenchmarkParam>;
using BenchmarkFunc = std::function<BenchmarkResults(const BenchmarkConfig&, BenchmarkParamsList)>;

/**
 * @brief Contains information pertaining to a benchmark. This is created automatically using parameters supplied
 * through a BENCHMARK macro.
 *
 */
struct Benchmark {
    std::string category;
    std::string name;
    BenchmarkFunc func;
    BenchmarkParamsList params;

    /**
     * @brief Returns the display name of the benchmark.
     *
     * @return The display name of the benchmark.
     */
    inline std::string getDisplayName() const {
        std::string paramsStr = "[";
        for (size_t i = 0; i < params.size(); ++i) {
            paramsStr += std::format("{}={}", params[i].key, params[i].strValue);
            if (i + 1 < params.size()) paramsStr += ", ";
        }
        paramsStr += "]";
        return std::format("{}::{}{}", category, name, paramsStr);
    }
};
}  // namespace roccvbench