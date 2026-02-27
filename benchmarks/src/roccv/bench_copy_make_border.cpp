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

#include <core/image_format.hpp>
#include <core/tensor.hpp>
#include <op_copy_make_border.hpp>
#include <roccvbench/registry.hpp>
#include <roccvbench/utils.hpp>

#include "roccv_bench_helpers.hpp"

using namespace roccv;

template <eDeviceType DeviceType>
static roccvbench::BenchmarkResults RunCopyMakeBorderBenchmark(roccvbench::BenchmarkParamsList params) {
    roccvbench::BenchmarkResults results;

    int samples = roccvbench::GetParamValue<int>(params, "samples");
    int width = roccvbench::GetParamValue<int>(params, "width");
    int height = roccvbench::GetParamValue<int>(params, "height");
    int runs = roccvbench::GetParamValue<int>(params, "runs");
    int warmupRuns = roccvbench::GetParamValue<int>(params, "warmupRuns");
    ImageFormat in_format = roccvbench::GetParamValue<ImageFormat>(params, "in_format");
    ImageFormat out_format = roccvbench::GetParamValue<ImageFormat>(params, "out_format");
    eBorderType borderType = roccvbench::GetParamValue<eBorderType>(params, "border_type");
    int top = roccvbench::GetParamValue<int>(params, "border_top");
    int left = roccvbench::GetParamValue<int>(params, "border_left");

    const float4 borderVal = make_float4(0.0f, 0.0f, 0.0f, 1.0f);

    TensorRequirements inReqs = Tensor::CalcRequirements(samples, {width, height}, in_format);
    Tensor::Requirements outReqs = Tensor::CalcRequirements(samples, {width + left * 2, height + top * 2}, out_format);
    Tensor input(inReqs);
    Tensor output(outReqs);

    RegisterMemoryUsage(input, results.readMemoryBytes);
    RegisterMemoryUsage(output, results.writtenMemoryBytes);

    FillTensor(input);

    CopyMakeBorder op;

    hipStream_t stream;
    HIP_VALIDATE_NO_ERRORS(hipStreamCreate(&stream));

    ROCCV_BENCH_RECORD_BLOCK(
        {
            op(stream, input, output, top, left, borderType, borderVal, DeviceType);
            if constexpr (DeviceType == eDeviceType::GPU) {
                HIP_VALIDATE_NO_ERRORS(hipStreamSynchronize(stream));
            }
        },
        results.executionTime, runs, warmupRuns);

    HIP_VALIDATE_NO_ERRORS(hipStreamDestroy(stream));

    return results;
}

#define DEFINE_COPY_MAKE_BORDER_BENCHMARK(name, device, in_format, out_format, border_type, border_top, border_left) \
    BENCHMARK_P(CopyMakeBorder, name,                                                                                \
                BENCH_PARAMS(BENCH_PARAM("in_format", in_format), BENCH_PARAM("out_format", out_format),             \
                             BENCH_PARAM("border_type", border_type), BENCH_PARAM("border_top", border_top),         \
                             BENCH_PARAM("border_left", border_left))) {                                             \
        return RunCopyMakeBorderBenchmark<device>(params);                                                           \
    }

DEFINE_COPY_MAKE_BORDER_BENCHMARK(GPU, eDeviceType::GPU, FMT_RGB8, FMT_RGB8, eBorderType::BORDER_TYPE_CONSTANT, 9, 9);
DEFINE_COPY_MAKE_BORDER_BENCHMARK(GPU, eDeviceType::GPU, FMT_RGB8, FMT_RGB8, eBorderType::BORDER_TYPE_REPLICATE, 9, 9);
DEFINE_COPY_MAKE_BORDER_BENCHMARK(GPU, eDeviceType::GPU, FMT_RGB8, FMT_RGB8, eBorderType::BORDER_TYPE_REFLECT, 9, 9);
DEFINE_COPY_MAKE_BORDER_BENCHMARK(GPU, eDeviceType::GPU, FMT_RGB8, FMT_RGB8, eBorderType::BORDER_TYPE_REFLECT101, 9, 9);
DEFINE_COPY_MAKE_BORDER_BENCHMARK(GPU, eDeviceType::GPU, FMT_RGB8, FMT_RGB8, eBorderType::BORDER_TYPE_WRAP, 9, 9);
