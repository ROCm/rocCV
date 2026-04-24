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
#include <op_custom_crop.hpp>
#include <roccvbench/registry.hpp>
#include <roccvbench/utils.hpp>

#include "roccv_bench_helpers.hpp"

using namespace roccv;

template <eDeviceType DeviceType>
static roccvbench::BenchmarkResults RunCustomCropBenchmark(roccvbench::BenchmarkParamsList params) {
    roccvbench::BenchmarkResults results;

    int samples = roccvbench::GetParamValue<int>(params, "samples");
    int width = roccvbench::GetParamValue<int>(params, "width");
    int height = roccvbench::GetParamValue<int>(params, "height");
    int runs = roccvbench::GetParamValue<int>(params, "runs");
    int warmupRuns = roccvbench::GetParamValue<int>(params, "warmupRuns");
    ImageFormat inFormat = roccvbench::GetParamValue<ImageFormat>(params, "inFormat");
    ImageFormat outFormat = roccvbench::GetParamValue<ImageFormat>(params, "outFormat");
    Box_t cropRect = roccvbench::GetParamValue<Box_t>(params, "cropRect");

    Tensor::Requirements inReqs = Tensor::CalcRequirements(samples, {width, height}, inFormat, DeviceType);
    Tensor::Requirements outReqs = Tensor::CalcRequirements(
        samples, {static_cast<int>(cropRect.width), static_cast<int>(cropRect.height)}, outFormat, DeviceType);
    Tensor input(inReqs);
    Tensor output(outReqs);

    // Kernel only reads the output size from the input tensor, so we need to register the output tensor for reading.
    RegisterMemoryUsage(output, results.readMemoryBytes);
    RegisterMemoryUsage(output, results.writtenMemoryBytes);

    FillTensor(input);

    CustomCrop op;
    hipStream_t stream;
    HIP_VALIDATE_NO_ERRORS(hipStreamCreate(&stream));

    ROCCV_BENCH_RECORD_BLOCK(
        {
            op(stream, input, output, cropRect, DeviceType);
            if constexpr (DeviceType == eDeviceType::GPU) {
                HIP_VALIDATE_NO_ERRORS(hipStreamSynchronize(stream));
            }
        },
        results.executionTimes, runs, warmupRuns);

    HIP_VALIDATE_NO_ERRORS(hipStreamDestroy(stream));

    return results;
}

#define DEFINE_CUSTOM_CROP_BENCHMARK(name, device, inFormat, outFormat, cropRect)                    \
    BENCHMARK_P(CustomCrop, name,                                                                    \
                BENCH_PARAMS(BENCH_PARAM("inFormat", inFormat), BENCH_PARAM("outFormat", outFormat), \
                             BENCH_PARAM("cropRect", cropRect))) {                                   \
        return RunCustomCropBenchmark<device>(params);                                               \
    }

// GPU benchmarks
DEFINE_CUSTOM_CROP_BENCHMARK(GPU, eDeviceType::GPU, FMT_RGB8, FMT_RGB8, ((Box_t){150, 50, 400, 300}));
DEFINE_CUSTOM_CROP_BENCHMARK(GPU, eDeviceType::GPU, FMT_RGBA8, FMT_RGBA8, ((Box_t){150, 50, 400, 300}));
DEFINE_CUSTOM_CROP_BENCHMARK(GPU, eDeviceType::GPU, FMT_U8, FMT_U8, ((Box_t){150, 50, 400, 300}));
DEFINE_CUSTOM_CROP_BENCHMARK(GPU, eDeviceType::GPU, FMT_F32, FMT_F32, ((Box_t){150, 50, 400, 300}));

// CPU benchmarks
DEFINE_CUSTOM_CROP_BENCHMARK(CPU, eDeviceType::CPU, FMT_RGB8, FMT_RGB8, ((Box_t){150, 50, 400, 300}));
DEFINE_CUSTOM_CROP_BENCHMARK(CPU, eDeviceType::CPU, FMT_RGBA8, FMT_RGBA8, ((Box_t){150, 50, 400, 300}));
DEFINE_CUSTOM_CROP_BENCHMARK(CPU, eDeviceType::CPU, FMT_U8, FMT_U8, ((Box_t){150, 50, 400, 300}));