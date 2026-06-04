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
#include <op_brightness_contrast.hpp>
#include <roccvbench/registry.hpp>
#include <roccvbench/utils.hpp>

#include "roccv_bench_helpers.hpp"

using namespace roccv;

// copied over from src/op_brightness_contrast.cpp (in anon. namespace there)
typedef enum eBCType {
    BC_TYPE_DEFAULT = 0,    ///< Use default value.
    BC_TYPE_BROADCAST = 1,  ///< Broadcast single value to all samples.
    BC_TYPE_PER = 2         ///< Per-sample values.
} eBCType;

template <eDeviceType DeviceType>
static roccvbench::BenchmarkResults RunBrightnessContrastBenchmark(roccvbench::BenchmarkParamsList params) {
    roccvbench::BenchmarkResults results;

    int samples = roccvbench::GetParamValue<int>(params, "samples");
    int width = roccvbench::GetParamValue<int>(params, "width");
    int height = roccvbench::GetParamValue<int>(params, "height");
    int runs = roccvbench::GetParamValue<int>(params, "runs");
    int warmupRuns = roccvbench::GetParamValue<int>(params, "warmupRuns");
    ImageFormat inFormat = roccvbench::GetParamValue<ImageFormat>(params, "inFormat");
    ImageFormat outFormat = roccvbench::GetParamValue<ImageFormat>(params, "outFormat");
    eBCType bcType = roccvbench::GetParamValue<eBCType>(params, "bcType");

    Tensor::Requirements inReqs = Tensor::CalcRequirements(samples, {width, height}, inFormat, DeviceType);
    Tensor::Requirements outReqs = Tensor::CalcRequirements(samples, {width, height}, outFormat, DeviceType);
    Tensor input(inReqs);
    Tensor output(outReqs);

    RegisterMemoryUsage(input, results.readMemoryBytes);
    RegisterMemoryUsage(output, results.writtenMemoryBytes);

    FillTensor(input);

    BrightnessContrast op;
    hipStream_t stream;
    HIP_VALIDATE_NO_ERRORS(hipStreamCreate(&stream));

    if (bcType == eBCType::BC_TYPE_DEFAULT) {
        roccvbench::RecordRuns<DeviceType>(stream, runs, warmupRuns, results.executionTimes, [&]() {
            op(stream, input, output, std::nullopt, std::nullopt, std::nullopt, std::nullopt, DeviceType);
        });
    } else {
        int bcSamples = (bcType == eBCType::BC_TYPE_BROADCAST) ? 1 : samples;
        DataType bcDType =
            (inFormat.dtype() == eDataType::DATA_TYPE_S32 || outFormat.dtype() == eDataType::DATA_TYPE_S32)
                ? DataType(eDataType::DATA_TYPE_F64)
                : DataType(eDataType::DATA_TYPE_F32);
        Tensor::Requirements bcReqs = Tensor::CalcRequirements({{bcSamples}, "N"}, bcDType, DeviceType);
        Tensor brightness(bcReqs);
        Tensor contrast(bcReqs);
        Tensor brightnessShift(bcReqs);
        Tensor contrastCenter(bcReqs);

        RegisterMemoryUsage(brightness, results.readMemoryBytes);
        RegisterMemoryUsage(contrast, results.readMemoryBytes);
        RegisterMemoryUsage(brightnessShift, results.readMemoryBytes);
        RegisterMemoryUsage(contrastCenter, results.readMemoryBytes);

        FillTensor(brightness);
        FillTensor(contrast);
        FillTensor(brightnessShift);
        FillTensor(contrastCenter);

        roccvbench::RecordRuns<DeviceType>(stream, runs, warmupRuns, results.executionTimes, [&]() {
            op(stream, input, output, brightness, contrast, brightnessShift, contrastCenter, DeviceType);
        });
    }
    HIP_VALIDATE_NO_ERRORS(hipStreamDestroy(stream));

    return results;
}

#define DEFINE_BRIGHTNESS_CONTRAST_BENCHMARK(name, device, inFormat, outFormat, bcType)              \
    BENCHMARK_P(BrightnessContrast, name,                                                            \
                BENCH_PARAMS(BENCH_PARAM("inFormat", inFormat), BENCH_PARAM("outFormat", outFormat), \
                             BENCH_PARAM("bcType", bcType))) {                                       \
        return RunBrightnessContrastBenchmark<device>(params);                                       \
    }

// clang-format off
// GPU benchmarks
DEFINE_BRIGHTNESS_CONTRAST_BENCHMARK(GPU, eDeviceType::GPU, FMT_U8, FMT_U8, eBCType::BC_TYPE_BROADCAST);
DEFINE_BRIGHTNESS_CONTRAST_BENCHMARK(GPU, eDeviceType::GPU, FMT_U8, FMT_U8, eBCType::BC_TYPE_DEFAULT);
DEFINE_BRIGHTNESS_CONTRAST_BENCHMARK(GPU, eDeviceType::GPU, FMT_U8, FMT_U8, eBCType::BC_TYPE_PER);

DEFINE_BRIGHTNESS_CONTRAST_BENCHMARK(GPU, eDeviceType::GPU, FMT_F32, FMT_F32, eBCType::BC_TYPE_BROADCAST);
DEFINE_BRIGHTNESS_CONTRAST_BENCHMARK(GPU, eDeviceType::GPU, FMT_F32, FMT_F32, eBCType::BC_TYPE_DEFAULT);
DEFINE_BRIGHTNESS_CONTRAST_BENCHMARK(GPU, eDeviceType::GPU, FMT_F32, FMT_F32, eBCType::BC_TYPE_PER);

DEFINE_BRIGHTNESS_CONTRAST_BENCHMARK(GPU, eDeviceType::GPU, FMT_S32, FMT_S32, eBCType::BC_TYPE_BROADCAST);
DEFINE_BRIGHTNESS_CONTRAST_BENCHMARK(GPU, eDeviceType::GPU, FMT_S32, FMT_S32, eBCType::BC_TYPE_DEFAULT); 
DEFINE_BRIGHTNESS_CONTRAST_BENCHMARK(GPU, eDeviceType::GPU, FMT_S32, FMT_S32, eBCType::BC_TYPE_PER); 

DEFINE_BRIGHTNESS_CONTRAST_BENCHMARK(GPU, eDeviceType::GPU, FMT_RGB8, FMT_RGB8, eBCType::BC_TYPE_BROADCAST);
DEFINE_BRIGHTNESS_CONTRAST_BENCHMARK(GPU, eDeviceType::GPU, FMT_RGB8, FMT_RGB8, eBCType::BC_TYPE_DEFAULT);
DEFINE_BRIGHTNESS_CONTRAST_BENCHMARK(GPU, eDeviceType::GPU, FMT_RGB8, FMT_RGB8, eBCType::BC_TYPE_PER);

DEFINE_BRIGHTNESS_CONTRAST_BENCHMARK(GPU, eDeviceType::GPU, FMT_RGBA8, FMT_RGBA8, eBCType::BC_TYPE_BROADCAST);
DEFINE_BRIGHTNESS_CONTRAST_BENCHMARK(GPU, eDeviceType::GPU, FMT_RGBA8, FMT_RGBA8, eBCType::BC_TYPE_DEFAULT); 
DEFINE_BRIGHTNESS_CONTRAST_BENCHMARK(GPU, eDeviceType::GPU, FMT_RGBA8, FMT_RGBA8, eBCType::BC_TYPE_PER); 

// CPU benchmarks
DEFINE_BRIGHTNESS_CONTRAST_BENCHMARK(CPU, eDeviceType::CPU, FMT_S32, FMT_S32, eBCType::BC_TYPE_BROADCAST);
DEFINE_BRIGHTNESS_CONTRAST_BENCHMARK(CPU, eDeviceType::CPU, FMT_S32, FMT_S32, eBCType::BC_TYPE_DEFAULT); 
DEFINE_BRIGHTNESS_CONTRAST_BENCHMARK(CPU, eDeviceType::CPU, FMT_S32, FMT_S32, eBCType::BC_TYPE_PER); 

DEFINE_BRIGHTNESS_CONTRAST_BENCHMARK(CPU, eDeviceType::CPU, FMT_RGB8, FMT_RGB8, eBCType::BC_TYPE_BROADCAST);
DEFINE_BRIGHTNESS_CONTRAST_BENCHMARK(CPU, eDeviceType::CPU, FMT_RGB8, FMT_RGB8, eBCType::BC_TYPE_DEFAULT);
DEFINE_BRIGHTNESS_CONTRAST_BENCHMARK(CPU, eDeviceType::CPU, FMT_RGB8, FMT_RGB8, eBCType::BC_TYPE_PER);
// clang-format on