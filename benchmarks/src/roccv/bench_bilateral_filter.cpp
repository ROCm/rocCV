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
#include <op_bilateral_filter.hpp>
#include <roccvbench/registry.hpp>
#include <roccvbench/utils.hpp>

#include "roccv_bench_helpers.hpp"

using namespace roccv;

template <eDeviceType DeviceType>
static roccvbench::BenchmarkResults RunBilateralFilterBenchmark(roccvbench::BenchmarkParamsList params) {
    roccvbench::BenchmarkResults results;

    int samples = roccvbench::GetParamValue<int>(params, "samples");
    int width = roccvbench::GetParamValue<int>(params, "width");
    int height = roccvbench::GetParamValue<int>(params, "height");
    int runs = roccvbench::GetParamValue<int>(params, "runs");
    int warmupRuns = roccvbench::GetParamValue<int>(params, "warmupRuns");
    ImageFormat in_format = roccvbench::GetParamValue<ImageFormat>(params, "in_format");
    ImageFormat out_format = roccvbench::GetParamValue<ImageFormat>(params, "out_format");
    int diameter = roccvbench::GetParamValue<int>(params, "diameter");
    float sigmaColor = roccvbench::GetParamValue<float>(params, "sigma_color");
    float sigmaSpace = roccvbench::GetParamValue<float>(params, "sigma_space");
    eBorderType borderType = roccvbench::GetParamValue<eBorderType>(params, "border_type");

    float4 borderValue = make_float4(1.0f, 0.0f, 1.0f, 1.0f);

    TensorRequirements inReqs = Tensor::CalcRequirements(samples, {width, height}, in_format);
    TensorRequirements outReqs = Tensor::CalcRequirements(samples, {width, height}, out_format);
    Tensor input(inReqs);
    Tensor output(outReqs);

    RegisterMemoryUsage(input, results.readMemoryBytes);
    RegisterMemoryUsage(output, results.writtenMemoryBytes);

    FillTensor(input);

    BilateralFilter op;
    hipStream_t stream;
    HIP_VALIDATE_NO_ERRORS(hipStreamCreate(&stream));

    ROCCV_BENCH_RECORD_BLOCK(
        {
            op(stream, input, output, diameter, sigmaColor, sigmaSpace, borderType, borderValue, DeviceType);
            if constexpr (DeviceType == eDeviceType::GPU) {
                HIP_VALIDATE_NO_ERRORS(hipStreamSynchronize(stream))
            }
        },
        results.executionTime, runs, warmupRuns);

    HIP_VALIDATE_NO_ERRORS(hipStreamDestroy(stream));

    return results;
}

#define DEFINE_BILATERAL_FILTER_BENCHMARK(name, device, in_format, out_format, diameter, sigmaColor, sigmaSpace, \
                                          borderType)                                                            \
    BENCHMARK_P(BilateralFilter, name,                                                                           \
                BENCH_PARAMS(BENCH_PARAM("in_format", in_format), BENCH_PARAM("out_format", out_format),         \
                             BENCH_PARAM("diameter", diameter), BENCH_PARAM("sigma_color", sigmaColor),          \
                             BENCH_PARAM("sigma_space", sigmaSpace), BENCH_PARAM("border_type", borderType))) {  \
        return RunBilateralFilterBenchmark<device>(params);                                                      \
    }

DEFINE_BILATERAL_FILTER_BENCHMARK(GPU, eDeviceType::GPU, FMT_RGB8, FMT_RGB8, 15, 75.0f, 75.0f,
                                  eBorderType::BORDER_TYPE_CONSTANT);
DEFINE_BILATERAL_FILTER_BENCHMARK(CPU, eDeviceType::CPU, FMT_RGB8, FMT_RGB8, 15, 75.0f, 75.0f,
                                  eBorderType::BORDER_TYPE_CONSTANT);