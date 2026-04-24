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
#include <op_cvt_color.hpp>
#include <roccvbench/registry.hpp>
#include <roccvbench/utils.hpp>

#include "roccv_bench_helpers.hpp"

using namespace roccv;

template <eDeviceType DeviceType>
static roccvbench::BenchmarkResults RunCvtColorBenchmark(roccvbench::BenchmarkParamsList params) {
    roccvbench::BenchmarkResults results;

    int samples = roccvbench::GetParamValue<int>(params, "samples");
    int width = roccvbench::GetParamValue<int>(params, "width");
    int height = roccvbench::GetParamValue<int>(params, "height");
    int runs = roccvbench::GetParamValue<int>(params, "runs");
    int warmupRuns = roccvbench::GetParamValue<int>(params, "warmupRuns");
    ImageFormat inFormat = roccvbench::GetParamValue<ImageFormat>(params, "inFormat");
    ImageFormat outFormat = roccvbench::GetParamValue<ImageFormat>(params, "outFormat");
    eColorConversionCode conversionCode = roccvbench::GetParamValue<eColorConversionCode>(params, "code");

    Tensor::Requirements inReqs = Tensor::CalcRequirements(samples, {width, height}, inFormat, DeviceType);
    Tensor::Requirements outReqs = Tensor::CalcRequirements(samples, {width, height}, outFormat, DeviceType);
    Tensor input(inReqs);
    Tensor output(outReqs);

    RegisterMemoryUsage(input, results.readMemoryBytes);
    RegisterMemoryUsage(output, results.writtenMemoryBytes);

    FillTensor(input);

    CvtColor op;
    hipStream_t stream;
    HIP_VALIDATE_NO_ERRORS(hipStreamCreate(&stream));

    ROCCV_BENCH_RECORD_BLOCK(
        {
            op(stream, input, output, conversionCode, DeviceType);
            if constexpr (DeviceType == eDeviceType::GPU) {
                HIP_VALIDATE_NO_ERRORS(hipStreamSynchronize(stream));
            }
        },
        results.executionTimes, runs, warmupRuns);

    HIP_VALIDATE_NO_ERRORS(hipStreamDestroy(stream));

    return results;
}

#define DEFINE_CVT_COLOR_BENCHMARK(name, device, inFormat, outFormat, code)                          \
    BENCHMARK_P(CvtColor, name,                                                                      \
                BENCH_PARAMS(BENCH_PARAM("inFormat", inFormat), BENCH_PARAM("outFormat", outFormat), \
                             BENCH_PARAM("code", code))) {                                           \
        return RunCvtColorBenchmark<device>(params);                                                 \
    }

// GPU benchmarks
DEFINE_CVT_COLOR_BENCHMARK(GPU, eDeviceType::GPU, FMT_RGB8, FMT_U8, COLOR_RGB2GRAY);

// CPU benchmarks
DEFINE_CVT_COLOR_BENCHMARK(CPU, eDeviceType::CPU, FMT_RGB8, FMT_U8, COLOR_RGB2GRAY);