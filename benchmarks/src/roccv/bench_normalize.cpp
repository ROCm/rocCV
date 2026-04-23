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
#include <op_normalize.hpp>
#include <roccvbench/registry.hpp>
#include <roccvbench/utils.hpp>

#include "roccv_bench_helpers.hpp"

using namespace roccv;

template <eDeviceType DeviceType>
static roccvbench::BenchmarkResults RunNormalizeBenchmark(roccvbench::BenchmarkParamsList params) {
    roccvbench::BenchmarkResults results;

    int samples = roccvbench::GetParamValue<int>(params, "samples");
    int width = roccvbench::GetParamValue<int>(params, "width");
    int height = roccvbench::GetParamValue<int>(params, "height");
    int runs = roccvbench::GetParamValue<int>(params, "runs");
    int warmupRuns = roccvbench::GetParamValue<int>(params, "warmupRuns");
    ImageFormat inFormat = roccvbench::GetParamValue<ImageFormat>(params, "inFormat");
    ImageFormat outFormat = roccvbench::GetParamValue<ImageFormat>(params, "outFormat");

    Tensor::Requirements inReqs = Tensor::CalcRequirements(samples, {width, height}, inFormat, DeviceType);
    Tensor::Requirements outReqs = Tensor::CalcRequirements(samples, {width, height}, outFormat, DeviceType);
    Tensor input(inReqs);
    Tensor output(outReqs);

    Tensor::Requirements paramTensorReqs = Tensor::CalcRequirements(
        TensorShape(TensorLayout(eTensorLayout::TENSOR_LAYOUT_NHWC), {1, 1, 1, inFormat.channels()}),
        DataType(eDataType::DATA_TYPE_F32), DeviceType);
    Tensor scale(paramTensorReqs);
    Tensor base(paramTensorReqs);

    RegisterMemoryUsage(input, results.readMemoryBytes);
    RegisterMemoryUsage(output, results.writtenMemoryBytes);
    RegisterMemoryUsage(scale, results.readMemoryBytes);
    RegisterMemoryUsage(base, results.readMemoryBytes);

    FillTensor(input);
    FillTensor(scale);
    FillTensor(base);

    Normalize op;
    hipStream_t stream;
    HIP_VALIDATE_NO_ERRORS(hipStreamCreate(&stream));

    ROCCV_BENCH_RECORD_BLOCK(
        {
            op(stream, input, base, scale, output, 1.0f, 0.0f, 0.00001f, 0, DeviceType);
            if constexpr (DeviceType == eDeviceType::GPU) {
                HIP_VALIDATE_NO_ERRORS(hipStreamSynchronize(stream));
            }
        },
        results.executionTime, runs, warmupRuns);

    HIP_VALIDATE_NO_ERRORS(hipStreamDestroy(stream));

    return results;
}

#define DEFINE_NORMALIZE_BENCHMARK(name, device, inFormat, outFormat)                                   \
    BENCHMARK_P(Normalize, name,                                                                        \
                BENCH_PARAMS(BENCH_PARAM("inFormat", inFormat), BENCH_PARAM("outFormat", outFormat))) { \
        return RunNormalizeBenchmark<device>(params);                                                   \
    }

// GPU benchmarks
DEFINE_NORMALIZE_BENCHMARK(GPU, eDeviceType::GPU, FMT_RGB8, FMT_RGB8);
DEFINE_NORMALIZE_BENCHMARK(GPU, eDeviceType::GPU, FMT_U8, FMT_U8);
DEFINE_NORMALIZE_BENCHMARK(GPU, eDeviceType::GPU, FMT_F32, FMT_F32);

// CPU benchmarks
DEFINE_NORMALIZE_BENCHMARK(CPU, eDeviceType::CPU, FMT_RGB8, FMT_RGB8);