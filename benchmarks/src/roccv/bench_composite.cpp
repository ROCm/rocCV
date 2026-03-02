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
#include <operator_types.h>

#include <core/image_format.hpp>
#include <core/tensor.hpp>
#include <op_composite.hpp>
#include <roccvbench/registry.hpp>
#include <roccvbench/utils.hpp>

#include "roccv_bench_helpers.hpp"

using namespace roccv;

template <eDeviceType DeviceType>
static roccvbench::BenchmarkResults RunCompositeBenchmark(roccvbench::BenchmarkParamsList params) {
    roccvbench::BenchmarkResults results;

    int samples = roccvbench::GetParamValue<int>(params, "samples");
    int width = roccvbench::GetParamValue<int>(params, "width");
    int height = roccvbench::GetParamValue<int>(params, "height");
    int runs = roccvbench::GetParamValue<int>(params, "runs");
    int warmupRuns = roccvbench::GetParamValue<int>(params, "warmupRuns");
    ImageFormat in_format = roccvbench::GetParamValue<ImageFormat>(params, "in_format");
    ImageFormat out_format = roccvbench::GetParamValue<ImageFormat>(params, "out_format");

    Tensor::Requirements inReqs = Tensor::CalcRequirements(samples, (Size2D){width, height}, in_format, DeviceType);
    Tensor::Requirements maskReqs = Tensor::CalcRequirements(samples, (Size2D){width, height}, FMT_U8, DeviceType);
    Tensor::Requirements outReqs = Tensor::CalcRequirements(samples, (Size2D){width, height}, out_format, DeviceType);

    Tensor background(inReqs);
    Tensor foreground(inReqs);
    Tensor mask(maskReqs);
    Tensor output(outReqs);

    RegisterMemoryUsage(background, results.readMemoryBytes);
    RegisterMemoryUsage(foreground, results.readMemoryBytes);
    RegisterMemoryUsage(mask, results.readMemoryBytes);
    RegisterMemoryUsage(output, results.writtenMemoryBytes);

    FillTensor(background);
    FillTensor(foreground);
    FillTensor(mask);

    Composite op;

    hipStream_t stream;
    HIP_VALIDATE_NO_ERRORS(hipStreamCreate(&stream));

    ROCCV_BENCH_RECORD_BLOCK(
        {
            op(stream, foreground, background, mask, output, DeviceType);
            if constexpr (DeviceType == eDeviceType::GPU) {
                HIP_VALIDATE_NO_ERRORS(hipStreamSynchronize(stream));
            }
        },
        results.executionTime, runs, warmupRuns);

    HIP_VALIDATE_NO_ERRORS(hipStreamDestroy(stream));

    return results;
}

#define DEFINE_COMPOSITE_BENCHMARK(name, device, in_format, out_format)                                     \
    BENCHMARK_P(Composite, name,                                                                            \
                BENCH_PARAMS(BENCH_PARAM("in_format", in_format), BENCH_PARAM("out_format", out_format))) { \
        return RunCompositeBenchmark<device>(params);                                                       \
    }

// GPU benchmarks
DEFINE_COMPOSITE_BENCHMARK(GPU, eDeviceType::GPU, FMT_RGB8, FMT_RGB8);
DEFINE_COMPOSITE_BENCHMARK(GPU, eDeviceType::GPU, FMT_RGB8, FMT_RGBA8);

// CPU benchmarks
DEFINE_COMPOSITE_BENCHMARK(CPU, eDeviceType::CPU, FMT_RGB8, FMT_RGB8);