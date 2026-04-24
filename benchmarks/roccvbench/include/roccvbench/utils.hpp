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

#include <core/hip_assert.h>
#include <core/util_enums.h>
#include <hip/hip_runtime.h>

#include <chrono>
#include <random>
#include <stdexcept>
#include <utility>
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

namespace detail {

// Per-run timer. CPU specialization uses a monotonic host clock; GPU
// specialization uses HIP events so the recorded interval is the kernel's
// device-side execution time, exclusive of host scheduling and stream-sync
// overhead.
template <eDeviceType DeviceType>
class RunTimer;

template <>
class RunTimer<eDeviceType::CPU> {
   public:
    void start(hipStream_t /*stream*/) { startTime_ = std::chrono::steady_clock::now(); }
    double stopSeconds(hipStream_t /*stream*/) {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - startTime_).count();
    }

   private:
    std::chrono::steady_clock::time_point startTime_;
};

template <>
class RunTimer<eDeviceType::GPU> {
   public:
    RunTimer() {
        HIP_VALIDATE_NO_ERRORS(hipEventCreate(&startEvent_));
        HIP_VALIDATE_NO_ERRORS(hipEventCreate(&stopEvent_));
    }
    ~RunTimer() {
        // Best-effort cleanup; nothing useful to do with errors during destruction.
        (void)hipEventDestroy(startEvent_);
        (void)hipEventDestroy(stopEvent_);
    }
    RunTimer(const RunTimer&) = delete;
    RunTimer& operator=(const RunTimer&) = delete;

    void start(hipStream_t stream) { HIP_VALIDATE_NO_ERRORS(hipEventRecord(startEvent_, stream)); }
    double stopSeconds(hipStream_t stream) {
        HIP_VALIDATE_NO_ERRORS(hipEventRecord(stopEvent_, stream));
        HIP_VALIDATE_NO_ERRORS(hipEventSynchronize(stopEvent_));
        float ms = 0.0f;
        HIP_VALIDATE_NO_ERRORS(hipEventElapsedTime(&ms, startEvent_, stopEvent_));
        return static_cast<double>(ms) / 1000.0;
    }

   private:
    hipEvent_t startEvent_, stopEvent_;
};

}  // namespace detail

/**
 * @brief Runs `fn` (numRuns + warmupRuns) times. The first `warmupRuns` iterations are discarded; each subsequent
 * run's elapsed time (in seconds) is appended to `executionTimes` in execution order. The CPU instantiation uses
 * std::chrono::steady_clock; the GPU instantiation uses HIP events so the recorded interval reflects on-device
 * kernel time only.
 */
template <eDeviceType DeviceType, typename Fn>
inline void RecordRuns(hipStream_t stream, int numRuns, int warmupRuns, std::vector<double>& executionTimes, Fn&& fn) {
    detail::RunTimer<DeviceType> timer;
    for (int i = 0; i < numRuns + warmupRuns; i++) {
        timer.start(stream);
        fn();
        const double t = timer.stopSeconds(stream);
        if (i >= warmupRuns) executionTimes.push_back(t);
    }
}

/// CPU-only convenience overload for benchmarks that don't have a HIP stream (e.g., OpenCV).
template <typename Fn>
inline void RecordRunsCpu(int numRuns, int warmupRuns, std::vector<double>& executionTimes, Fn&& fn) {
    RecordRuns<eDeviceType::CPU>(nullptr, numRuns, warmupRuns, executionTimes, std::forward<Fn>(fn));
}

}  // namespace roccvbench