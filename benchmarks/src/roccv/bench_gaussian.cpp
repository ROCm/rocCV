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
#include <op_gaussian.hpp>
#include <roccvbench/registry.hpp>
#include <roccvbench/utils.hpp>

#include "roccv_bench_helpers.hpp"

using namespace roccv;

template <eDeviceType DeviceType>
static roccvbench::BenchmarkResults RunGaussianBenchmark(roccvbench::BenchmarkParamsList params) {
    roccvbench::BenchmarkResults results;

    int samples = roccvbench::GetParamValue<int>(params, "samples");
    int width = roccvbench::GetParamValue<int>(params, "width");
    int height = roccvbench::GetParamValue<int>(params, "height");
    int runs = roccvbench::GetParamValue<int>(params, "runs");
    int warmupRuns = roccvbench::GetParamValue<int>(params, "warmupRuns");
    ImageFormat inFormat = roccvbench::GetParamValue<ImageFormat>(params, "inFormat");
    ImageFormat outFormat = roccvbench::GetParamValue<ImageFormat>(params, "outFormat");
    int kernelWidth = roccvbench::GetParamValue<int>(params, "kernelWidth");
    int kernelHeight = roccvbench::GetParamValue<int>(params, "kernelHeight");
    double sigmaX = roccvbench::GetParamValue<double>(params, "sigmaX");
    double sigmaY = roccvbench::GetParamValue<double>(params, "sigmaY");
    eBorderType borderType = roccvbench::GetParamValue<eBorderType>(params, "borderType");

    TensorRequirements inReqs = Tensor::CalcRequirements(samples, {width, height}, inFormat, DeviceType);
    TensorRequirements outReqs = Tensor::CalcRequirements(samples, {width, height}, outFormat, DeviceType);
    Tensor input(inReqs);
    Tensor output(outReqs);

    RegisterMemoryUsage(input, results.readMemoryBytes);
    RegisterMemoryUsage(output, results.writtenMemoryBytes);

    FillTensor(input);

    const int prev_round = fegetround();
    fesetround(FE_TONEAREST);
    int maxKernelWidth =
        (kernelWidth > 0)
            ? kernelWidth
            : static_cast<int>(std::rint(sigmaX * (input.dtype().etype() == DATA_TYPE_U8 ? 3 : 4) * 2 + 1)) | 1;
    int maxKernelHeight =
        (kernelHeight > 0)
            ? kernelHeight
            : static_cast<int>(std::rint(sigmaY * (input.dtype().etype() == DATA_TYPE_U8 ? 3 : 4) * 2 + 1)) | 1;
    fesetround(prev_round);

    Gaussian op(maxKernelWidth, maxKernelHeight);

    hipStream_t stream;
    HIP_VALIDATE_NO_ERRORS(hipStreamCreate(&stream));

    roccvbench::RecordRuns<DeviceType>(stream, runs, warmupRuns, results.executionTimes, [&]() {
        op(stream, input, output, kernelWidth, kernelHeight, sigmaX, sigmaY, borderType, DeviceType);
    });

    HIP_VALIDATE_NO_ERRORS(hipStreamDestroy(stream));

    return results;
}

#define DEFINE_GAUSSIAN_BENCHMARK(name, device, inFormat, outFormat, kernelWidth, kernelHeight, sigmaX, sigmaY,  \
                                  borderType)                                                                    \
    BENCHMARK_P(Gaussian, name,                                                                                  \
                BENCH_PARAMS(BENCH_PARAM("inFormat", inFormat), BENCH_PARAM("outFormat", outFormat),             \
                             BENCH_PARAM("kernelWidth", kernelWidth), BENCH_PARAM("kernelHeight", kernelHeight), \
                             BENCH_PARAM("sigmaX", sigmaX), BENCH_PARAM("sigmaY", sigmaY),                       \
                             BENCH_PARAM("borderType", borderType))) {                                               \
        return RunGaussianBenchmark<device>(params);                                                             \
    }

// GPU benchmarks
DEFINE_GAUSSIAN_BENCHMARK(GPU, eDeviceType::GPU, FMT_RGB8, FMT_RGB8, 3, 3, 0.5, 0.5, eBorderType::BORDER_TYPE_REFLECT);
// DEFINE_GAUSSIAN_BENCHMARK(GPU, eDeviceType::GPU, FMT_RGB8, FMT_RGB8, 3, 5, 0.5, 0.75, eBorderType::BORDER_TYPE_REFLECT);
// DEFINE_GAUSSIAN_BENCHMARK(GPU, eDeviceType::GPU, FMT_RGB8, FMT_RGB8, 3, 7, 0.5, 1.0, eBorderType::BORDER_TYPE_REFLECT);
// DEFINE_GAUSSIAN_BENCHMARK(GPU, eDeviceType::GPU, FMT_RGB8, FMT_RGB8, 3, 9, 0.5, 1.5, eBorderType::BORDER_TYPE_REFLECT);
DEFINE_GAUSSIAN_BENCHMARK(GPU, eDeviceType::GPU, FMT_RGB8, FMT_RGB8, 5, 5, 0.75, 0.75, eBorderType::BORDER_TYPE_REFLECT);
DEFINE_GAUSSIAN_BENCHMARK(GPU, eDeviceType::GPU, FMT_RGB8, FMT_RGB8, 7, 7, 1.0, 1.0, eBorderType::BORDER_TYPE_REFLECT);
DEFINE_GAUSSIAN_BENCHMARK(GPU, eDeviceType::GPU, FMT_RGB8, FMT_RGB8, 9, 9, 1.5, 1.5, eBorderType::BORDER_TYPE_REFLECT);
// DEFINE_GAUSSIAN_BENCHMARK(GPU, eDeviceType::GPU, FMT_RGB8, FMT_RGB8, 11, 11, 1.75, 1.75, eBorderType::BORDER_TYPE_REFLECT);
// DEFINE_GAUSSIAN_BENCHMARK(GPU, eDeviceType::GPU, FMT_RGB8, FMT_RGB8, 15, 15, eBorderType::BORDER_TYPE_REFLECT);

DEFINE_GAUSSIAN_BENCHMARK(GPU, eDeviceType::GPU, FMT_U8, FMT_U8, 3, 3, 0.5, 0.5, eBorderType::BORDER_TYPE_REFLECT);
// DEFINE_GAUSSIAN_BENCHMARK(GPU, eDeviceType::GPU, FMT_U8, FMT_U8, 3, 5, 0.5, 0.75, eBorderType::BORDER_TYPE_REFLECT);
// DEFINE_GAUSSIAN_BENCHMARK(GPU, eDeviceType::GPU, FMT_U8, FMT_U8, 3, 7, 0.5, 1.0, eBorderType::BORDER_TYPE_REFLECT);
// DEFINE_GAUSSIAN_BENCHMARK(GPU, eDeviceType::GPU, FMT_U8, FMT_U8, 3, 9, 0.5, 1.5, eBorderType::BORDER_TYPE_REFLECT);
DEFINE_GAUSSIAN_BENCHMARK(GPU, eDeviceType::GPU, FMT_U8, FMT_U8, 5, 5, 0.75, 0.75, eBorderType::BORDER_TYPE_REFLECT);
DEFINE_GAUSSIAN_BENCHMARK(GPU, eDeviceType::GPU, FMT_U8, FMT_U8, 7, 7, 1.0, 1.0, eBorderType::BORDER_TYPE_REFLECT);
DEFINE_GAUSSIAN_BENCHMARK(GPU, eDeviceType::GPU, FMT_U8, FMT_U8, 9, 9, 1.5, 1.5, eBorderType::BORDER_TYPE_REFLECT);
// DEFINE_GAUSSIAN_BENCHMARK(GPU, eDeviceType::GPU, FMT_U8, FMT_U8, 11, 11, 1.75, 1.75, eBorderType::BORDER_TYPE_REFLECT);
// DEFINE_GAUSSIAN_BENCHMARK(GPU, eDeviceType::GPU, FMT_U8, FMT_U8, 15, 15, 2.5, 2.5, eBorderType::BORDER_TYPE_REFLECT);

// CPU benchmarks
DEFINE_GAUSSIAN_BENCHMARK(CPU, eDeviceType::CPU, FMT_RGB8, FMT_RGB8, 3, 3, 0.5, 0.5, eBorderType::BORDER_TYPE_REFLECT);
// DEFINE_GAUSSIAN_BENCHMARK(CPU, eDeviceType::CPU, FMT_RGB8, FMT_RGB8, 3, 5, 0.5, 0.75, eBorderType::BORDER_TYPE_REFLECT);
// DEFINE_GAUSSIAN_BENCHMARK(CPU, eDeviceType::CPU, FMT_RGB8, FMT_RGB8, 3, 7, 0.5, 1.0, eBorderType::BORDER_TYPE_REFLECT);
// DEFINE_GAUSSIAN_BENCHMARK(CPU, eDeviceType::CPU, FMT_RGB8, FMT_RGB8, 3, 9, 0.5, 1.5, eBorderType::BORDER_TYPE_REFLECT);
DEFINE_GAUSSIAN_BENCHMARK(CPU, eDeviceType::CPU, FMT_RGB8, FMT_RGB8, 5, 5, 0.75, 0.75, eBorderType::BORDER_TYPE_REFLECT);
DEFINE_GAUSSIAN_BENCHMARK(CPU, eDeviceType::CPU, FMT_RGB8, FMT_RGB8, 7, 7, 1.0, 1.0, eBorderType::BORDER_TYPE_REFLECT);
DEFINE_GAUSSIAN_BENCHMARK(CPU, eDeviceType::CPU, FMT_RGB8, FMT_RGB8, 9, 9, 1.5, 1.5, eBorderType::BORDER_TYPE_REFLECT);
// DEFINE_GAUSSIAN_BENCHMARK(CPU, eDeviceType::CPU, FMT_RGB8, FMT_RGB8, 11, 11, 1.75, 1.75, eBorderType::BORDER_TYPE_REFLECT);
// DEFINE_GAUSSIAN_BENCHMARK(CPU, eDeviceType::CPU, FMT_RGB8, FMT_RGB8, 15, 15, 2.5, 2.5, eBorderType::BORDER_TYPE_REFLECT);

DEFINE_GAUSSIAN_BENCHMARK(CPU, eDeviceType::CPU, FMT_U8, FMT_U8, 3, 3, 0.5, 0.5, eBorderType::BORDER_TYPE_REFLECT);
// DEFINE_GAUSSIAN_BENCHMARK(CPU, eDeviceType::CPU, FMT_U8, FMT_U8, 3, 5, 0.5, 0.75, eBorderType::BORDER_TYPE_REFLECT);
// DEFINE_GAUSSIAN_BENCHMARK(CPU, eDeviceType::CPU, FMT_U8, FMT_U8, 3, 7, 0.5, 1.0, eBorderType::BORDER_TYPE_REFLECT);
// DEFINE_GAUSSIAN_BENCHMARK(CPU, eDeviceType::CPU, FMT_U8, FMT_U8, 3, 9, 0.5, 1.5, eBorderType::BORDER_TYPE_REFLECT);
DEFINE_GAUSSIAN_BENCHMARK(CPU, eDeviceType::CPU, FMT_U8, FMT_U8, 5, 5, 0.75, 0.75, eBorderType::BORDER_TYPE_REFLECT);
DEFINE_GAUSSIAN_BENCHMARK(CPU, eDeviceType::CPU, FMT_U8, FMT_U8, 7, 7, 1.0, 1.0, eBorderType::BORDER_TYPE_REFLECT);
DEFINE_GAUSSIAN_BENCHMARK(CPU, eDeviceType::CPU, FMT_U8, FMT_U8, 9, 9, 1.5, 1.5, eBorderType::BORDER_TYPE_REFLECT);
// DEFINE_GAUSSIAN_BENCHMARK(CPU, eDeviceType::CPU, FMT_U8, FMT_U8, 11, 11, 1.75, 1.75, eBorderType::BORDER_TYPE_REFLECT);
// DEFINE_GAUSSIAN_BENCHMARK(CPU, eDeviceType::CPU, FMT_U8, FMT_U8, 15, 15, 2.5, 2.5, eBorderType::BORDER_TYPE_REFLECT);