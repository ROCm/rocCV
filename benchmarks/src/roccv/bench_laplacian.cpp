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

#include <core/hip_assert.h>

#include <cfenv>
#include <core/image_format.hpp>
#include <core/tensor.hpp>
#include <op_laplacian.hpp>
#include <roccvbench/registry.hpp>
#include <roccvbench/utils.hpp>

#include "roccv_bench_helpers.hpp"

using namespace roccv;

template <eDeviceType DeviceType>
static roccvbench::BenchmarkResults RunLaplacianBenchmark(roccvbench::BenchmarkParamsList params) {
    roccvbench::BenchmarkResults results;

    int samples = roccvbench::GetParamValue<int>(params, "samples");
    int width = roccvbench::GetParamValue<int>(params, "width");
    int height = roccvbench::GetParamValue<int>(params, "height");
    int runs = roccvbench::GetParamValue<int>(params, "runs");
    int warmupRuns = roccvbench::GetParamValue<int>(params, "warmupRuns");
    ImageFormat inFormat = roccvbench::GetParamValue<ImageFormat>(params, "inFormat");
    ImageFormat outFormat = roccvbench::GetParamValue<ImageFormat>(params, "outFormat");
    int ksize = roccvbench::GetParamValue<int>(params, "ksize");
    float scale = roccvbench::GetParamValue<float>(params, "scale");
    eBorderType borderType = roccvbench::GetParamValue<eBorderType>(params, "borderType");

    TensorRequirements inReqs = Tensor::CalcRequirements(samples, {width, height}, inFormat, DeviceType);
    TensorRequirements outReqs = Tensor::CalcRequirements(samples, {width, height}, outFormat, DeviceType);
    Tensor input(inReqs);
    Tensor output(outReqs);

    RegisterMemoryUsage(input, results.readMemoryBytes);
    RegisterMemoryUsage(output, results.writtenMemoryBytes);

    FillTensor(input);

    Laplacian op;

    hipStream_t stream;
    HIP_VALIDATE_NO_ERRORS(hipStreamCreate(&stream));

    roccvbench::RecordRuns<DeviceType>(stream, runs, warmupRuns, results.executionTimes,
                                       [&]() { op(stream, input, output, ksize, scale, borderType, DeviceType); });

    HIP_VALIDATE_NO_ERRORS(hipStreamDestroy(stream));

    return results;
}

#define DEFINE_LAPLACIAN_BENCHMARK(name, device, inFormat, outFormat, ksize, scale, borderType)      \
    BENCHMARK_P(Laplacian, name,                                                                     \
                BENCH_PARAMS(BENCH_PARAM("inFormat", inFormat), BENCH_PARAM("outFormat", outFormat), \
                             BENCH_PARAM("ksize", ksize), BENCH_PARAM("scale", scale),               \
                             BENCH_PARAM("borderType", borderType))) {                               \
        return RunLaplacianBenchmark<device>(params);                                                \
    }

// GPU benchmarks
DEFINE_LAPLACIAN_BENCHMARK(GPU, eDeviceType::GPU, FMT_RGB8, FMT_RGB8, 1, 1.0f, eBorderType::BORDER_TYPE_REFLECT);
DEFINE_LAPLACIAN_BENCHMARK(GPU, eDeviceType::GPU, FMT_RGB8, FMT_RGB8, 3, 1.0f, eBorderType::BORDER_TYPE_REFLECT);
DEFINE_LAPLACIAN_BENCHMARK(GPU, eDeviceType::GPU, FMT_RGB8, FMT_RGB8, 1, -1.0f, eBorderType::BORDER_TYPE_REFLECT);
DEFINE_LAPLACIAN_BENCHMARK(GPU, eDeviceType::GPU, FMT_RGB8, FMT_RGB8, 3, -1.0f, eBorderType::BORDER_TYPE_REFLECT);

DEFINE_LAPLACIAN_BENCHMARK(GPU, eDeviceType::GPU, FMT_U8, FMT_U8, 1, 1.0f, eBorderType::BORDER_TYPE_REFLECT);
DEFINE_LAPLACIAN_BENCHMARK(GPU, eDeviceType::GPU, FMT_U8, FMT_U8, 3, 1.0f, eBorderType::BORDER_TYPE_REFLECT);
DEFINE_LAPLACIAN_BENCHMARK(GPU, eDeviceType::GPU, FMT_U8, FMT_U8, 1, -1.0f, eBorderType::BORDER_TYPE_REFLECT);
DEFINE_LAPLACIAN_BENCHMARK(GPU, eDeviceType::GPU, FMT_U8, FMT_U8, 3, -1.0f, eBorderType::BORDER_TYPE_REFLECT);

// CPU benchmarks
DEFINE_LAPLACIAN_BENCHMARK(CPU, eDeviceType::CPU, FMT_RGB8, FMT_RGB8, 1, 1.0f, eBorderType::BORDER_TYPE_REFLECT);
DEFINE_LAPLACIAN_BENCHMARK(CPU, eDeviceType::CPU, FMT_RGB8, FMT_RGB8, 3, 1.0f, eBorderType::BORDER_TYPE_REFLECT);
DEFINE_LAPLACIAN_BENCHMARK(CPU, eDeviceType::CPU, FMT_RGB8, FMT_RGB8, 1, -1.0f, eBorderType::BORDER_TYPE_REFLECT);
DEFINE_LAPLACIAN_BENCHMARK(CPU, eDeviceType::CPU, FMT_RGB8, FMT_RGB8, 3, -1.0f, eBorderType::BORDER_TYPE_REFLECT);

DEFINE_LAPLACIAN_BENCHMARK(CPU, eDeviceType::CPU, FMT_U8, FMT_U8, 1, 1.0f, eBorderType::BORDER_TYPE_REFLECT);
DEFINE_LAPLACIAN_BENCHMARK(CPU, eDeviceType::CPU, FMT_U8, FMT_U8, 3, 1.0f, eBorderType::BORDER_TYPE_REFLECT);
DEFINE_LAPLACIAN_BENCHMARK(CPU, eDeviceType::CPU, FMT_U8, FMT_U8, 1, -1.0f, eBorderType::BORDER_TYPE_REFLECT);
DEFINE_LAPLACIAN_BENCHMARK(CPU, eDeviceType::CPU, FMT_U8, FMT_U8, 3, -1.0f, eBorderType::BORDER_TYPE_REFLECT);