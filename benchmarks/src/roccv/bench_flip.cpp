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

#include <core/hip_assert.h>

#include <core/image_format.hpp>
#include <core/tensor.hpp>
#include <op_flip.hpp>
#include <roccvbench/registry.hpp>
#include <roccvbench/utils.hpp>

#include "roccv_bench_helpers.hpp"

using namespace roccv;

template <eDeviceType DeviceType>
static roccvbench::BenchmarkResults RunFlipBenchmark(const auto& config, roccvbench::BenchmarkParamsList params) {
    roccvbench::BenchmarkResults results;

    // Get parameters
    ImageFormat format = roccvbench::GetParamValue<ImageFormat>(params, "format");
    int32_t flip_code = roccvbench::GetParamValue<int32_t>(params, "flip_code");

    TensorRequirements reqs = Tensor::CalcRequirements(
        config.samples, (Size2D){static_cast<int>(config.width), static_cast<int>(config.height)}, format);
    Tensor input(reqs);
    Tensor output(reqs);
    RegisterMemoryUsage(input, results.readMemoryBytes);
    RegisterMemoryUsage(output, results.writtenMemoryBytes);
    FillTensor(input);
    Flip op;
    hipStream_t stream;
    HIP_VALIDATE_NO_ERRORS(hipStreamCreate(&stream));
    ROCCV_BENCH_RECORD_BLOCK(
        {
            op(stream, input, output, flip_code, DeviceType);
            if constexpr (DeviceType == eDeviceType::GPU) {
                HIP_VALIDATE_NO_ERRORS(hipStreamSynchronize(stream));
            }
        },
        results.executionTime, config.runs, config.warmupRuns);
    HIP_VALIDATE_NO_ERRORS(hipStreamDestroy(stream));
    return results;
}

#define DEFINE_FLIP_BENCHMARK(name, device, format, flip_code)                                                  \
    BENCHMARK_P(Flip, name, BENCH_PARAMS(BENCH_PARAM("format", format), BENCH_PARAM("flip_code", flip_code))) { \
        return RunFlipBenchmark<device>(config, params);                                                        \
    }

DEFINE_FLIP_BENCHMARK(GPU, eDeviceType::GPU, FMT_RGB8, 0);
DEFINE_FLIP_BENCHMARK(GPU, eDeviceType::GPU, FMT_RGBA8, 0);
DEFINE_FLIP_BENCHMARK(GPU, eDeviceType::GPU, FMT_U8, 0);