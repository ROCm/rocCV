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

#include <functional>
#include <string>

#include "structs.hpp"

namespace roccvbench {

#define ROCCVBENCH_CONCAT_(a, b) a##b
#define ROCCVBENCH_CONCAT(a, b) ROCCVBENCH_CONCAT_(a, b)
#define BENCHMARK_FUNC_NAME(cat, name) \
    ROCCVBENCH_CONCAT(ROCCVBENCH_CONCAT(ROCCVBENCH_CONCAT(Benchmark_, cat), name), __LINE__)
#define BENCHMARK_VAR_NAME(cat, name) \
    ROCCVBENCH_CONCAT(ROCCVBENCH_CONCAT(ROCCVBENCH_CONCAT(_benchmark_, cat), name), __LINE__)

/**
 * @brief Benchmark registry singleton. This object is responsible for keeping track of all benchmarks defined through
 * the BENCHMARK macro.
 *
 */
class BenchmarkRegistry {
   public:
    static BenchmarkRegistry& instance() {
        static BenchmarkRegistry reg;
        return reg;
    }

    void registerBenchmark(const std::string& category, const std::string& name, BenchmarkFunc func,
                           BenchmarkParamsList params) {
        if (m_benchmarks.count(category) == 0) {
            m_benchmarks.emplace(category, std::vector<Benchmark>());
        }
        m_benchmarks.at(category).emplace_back(Benchmark{category, name, func, params});
    }

    std::unordered_map<std::string, std::vector<Benchmark>>& getBenchmarks() { return m_benchmarks; }

   private:
    // Store benchmarks in a map with the category as a key, and a list of benchmarks for that category as the value.
    std::unordered_map<std::string, std::vector<Benchmark>> m_benchmarks;
};

#define REGISTER_BENCHMARK(func, name, category, _params_)                                             \
    static bool BENCHMARK_VAR_NAME(category, name) = [] {                                              \
        roccvbench::BenchmarkRegistry::instance().registerBenchmark(#category, #name, func, _params_); \
        return true;                                                                                   \
    }()

#define BENCH_PARAMS(...) roccvbench::BenchmarkParamsList({__VA_ARGS__})
#define BENCH_PARAM(key, value) roccvbench::BenchmarkParam({key, value, #value})

/**
 * @brief Creates a benchmark unit for roccv with a provided name.
 *
 */
#define BENCHMARK(category, name)                                                                               \
    roccvbench::BenchmarkResults BENCHMARK_FUNC_NAME(category, name)(const roccvbench::BenchmarkConfig& config, \
                                                                     roccvbench::BenchmarkParamsList params);   \
    REGISTER_BENCHMARK(BENCHMARK_FUNC_NAME(category, name), name, category, {});                                \
    roccvbench::BenchmarkResults BENCHMARK_FUNC_NAME(category, name)(const roccvbench::BenchmarkConfig& config, \
                                                                     roccvbench::BenchmarkParamsList params)

#define BENCHMARK_P(category, name, _params_)                                                                   \
    roccvbench::BenchmarkResults BENCHMARK_FUNC_NAME(category, name)(const roccvbench::BenchmarkConfig& config, \
                                                                     roccvbench::BenchmarkParamsList params);   \
    REGISTER_BENCHMARK(BENCHMARK_FUNC_NAME(category, name), name, category, _params_);                          \
    roccvbench::BenchmarkResults BENCHMARK_FUNC_NAME(category, name)(const roccvbench::BenchmarkConfig& config, \
                                                                     roccvbench::BenchmarkParamsList params)

}  // namespace roccvbench