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
#include <op_copy_make_border.hpp>
#include <roccvbench/registry.hpp>
#include <roccvbench/utils.hpp>

#include "roccv_bench_helpers.hpp"

using namespace roccv;

BENCHMARK(CopyMakeBorderConstant, GPU) {
    roccvbench::BenchmarkResults results;

    const int top = 9;
    const int left = 9;
    const float4 borderVal = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
    const eBorderType borderType = eBorderType::BORDER_TYPE_CONSTANT;

    TensorRequirements inReqs = Tensor::CalcRequirements(config.samples, {config.width, config.height}, FMT_RGB8);
    Tensor::Requirements outReqs =
        Tensor::CalcRequirements(config.samples, {config.width + left * 2, config.height + top * 2}, FMT_RGB8);
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
            op(stream, input, output, top, left, borderType, borderVal);
            HIP_VALIDATE_NO_ERRORS(hipStreamSynchronize(stream))
        },
        results.executionTime, config.runs, config.warmupRuns);

    HIP_VALIDATE_NO_ERRORS(hipStreamDestroy(stream));

    return results;
}

BENCHMARK(CopyMakeBorderConstant, CPU) {
    roccvbench::BenchmarkResults results;

    const int top = 9;
    const int left = 9;
    const float4 borderVal = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
    const eBorderType borderType = eBorderType::BORDER_TYPE_CONSTANT;

    TensorRequirements inReqs =
        Tensor::CalcRequirements(config.samples, {config.width, config.height}, FMT_RGB8, eDeviceType::CPU);
    Tensor::Requirements outReqs = Tensor::CalcRequirements(
        config.samples, {config.width + left * 2, config.height + top * 2}, FMT_RGB8, eDeviceType::CPU);
    Tensor input(inReqs);
    Tensor output(outReqs);

    RegisterMemoryUsage(input, results.readMemoryBytes);
    RegisterMemoryUsage(output, results.writtenMemoryBytes);

    FillTensor(input);

    CopyMakeBorder op;
    ROCCV_BENCH_RECORD_BLOCK(
        { op(nullptr, input, output, top, left, borderType, borderVal, eDeviceType::CPU); }, results.executionTime,
        config.runs, config.warmupRuns);

    return results;
}

BENCHMARK(CopyMakeBorderReflect, GPU) {
    roccvbench::BenchmarkResults results;
    results.executionTime = 0.0f;

    const int top = 9;
    const int left = 9;
    const float4 borderVal = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
    const eBorderType borderType = eBorderType::BORDER_TYPE_REFLECT;

    TensorRequirements inReqs = Tensor::CalcRequirements(config.samples, {config.width, config.height}, FMT_RGB8);
    Tensor::Requirements outReqs =
        Tensor::CalcRequirements(config.samples, {config.width + left * 2, config.height + top * 2}, FMT_RGB8);
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
            op(stream, input, output, top, left, borderType, borderVal);
            HIP_VALIDATE_NO_ERRORS(hipStreamSynchronize(stream))
        },
        results.executionTime, config.runs, config.warmupRuns);

    HIP_VALIDATE_NO_ERRORS(hipStreamDestroy(stream));

    return results;
}

BENCHMARK(CopyMakeBorderReflect, CPU) {
    roccvbench::BenchmarkResults results;
    results.executionTime = 0.0f;

    const int top = 9;
    const int left = 9;
    const float4 borderVal = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
    const eBorderType borderType = eBorderType::BORDER_TYPE_REFLECT;

    TensorRequirements inReqs =
        Tensor::CalcRequirements(config.samples, {config.width, config.height}, FMT_RGB8, eDeviceType::CPU);
    Tensor::Requirements outReqs = Tensor::CalcRequirements(
        config.samples, {config.width + left * 2, config.height + top * 2}, FMT_RGB8, eDeviceType::CPU);
    Tensor input(inReqs);
    Tensor output(outReqs);

    RegisterMemoryUsage(input, results.readMemoryBytes);
    RegisterMemoryUsage(output, results.writtenMemoryBytes);

    FillTensor(input);

    CopyMakeBorder op;
    ROCCV_BENCH_RECORD_BLOCK(
        { op(nullptr, input, output, top, left, borderType, borderVal, eDeviceType::CPU); }, results.executionTime,
        config.runs, config.warmupRuns);

    return results;
}