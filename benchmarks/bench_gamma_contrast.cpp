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

#include <core/tensor.hpp>
#include <op_gamma_contrast.hpp>
#include <roccvbench/registry.hpp>
#include <roccvbench/utils.hpp>

using namespace roccv;

BENCHMARK(GammaContrast, GPU) {
    roccvbench::BenchmarkResults results;
    results.execution_time = 0.0f;

    TensorRequirements reqs = Tensor::CalcRequirements(
        TensorShape(TensorLayout(TENSOR_LAYOUT_NHWC), {config.batches, config.height, config.width, 3}),
        DataType(DATA_TYPE_U8));
    Tensor input(reqs);
    Tensor output(reqs);
    Tensor gamma(TensorShape(TensorLayout(eTensorLayout::TENSOR_LAYOUT_N), {config.batches}),
                 DataType(eDataType::DATA_TYPE_F32));

    roccvbench::FillTensor(input);
    roccvbench::FillTensor(gamma);

    for (int i = 0; i < config.runs; i++) {
        hipStream_t stream;
        hipEvent_t begin, end;
        HIP_VALIDATE_NO_ERRORS(hipEventCreate(&begin));
        HIP_VALIDATE_NO_ERRORS(hipEventCreate(&end));
        HIP_VALIDATE_NO_ERRORS(hipStreamCreate(&stream));

        GammaContrast op;
        HIP_VALIDATE_NO_ERRORS(hipEventRecord(begin, stream));
        op(nullptr, input, output, gamma);
        HIP_VALIDATE_NO_ERRORS(hipEventRecord(end, stream));
        HIP_VALIDATE_NO_ERRORS(hipEventSynchronize(end));

        float execution_time;
        HIP_VALIDATE_NO_ERRORS(hipEventElapsedTime(&execution_time, begin, end));
        HIP_VALIDATE_NO_ERRORS(hipEventDestroy(begin));
        HIP_VALIDATE_NO_ERRORS(hipEventDestroy(end));
        HIP_VALIDATE_NO_ERRORS(hipStreamDestroy(stream));

        results.execution_time += execution_time / config.runs;
    }

    return results;
}

BENCHMARK(GammaContrast, CPU) {
    roccvbench::BenchmarkResults results;
    results.execution_time = 0.0f;

    TensorRequirements reqs = Tensor::CalcRequirements(
        TensorShape(TensorLayout(TENSOR_LAYOUT_NHWC), {config.batches, config.height, config.width, 3}),
        DataType(DATA_TYPE_U8), eDeviceType::CPU);
    Tensor input(reqs);
    Tensor output(reqs);
    Tensor gamma(TensorShape(TensorLayout(eTensorLayout::TENSOR_LAYOUT_N), {config.batches}),
                 DataType(eDataType::DATA_TYPE_F32), eDeviceType::CPU);

    roccvbench::FillTensor(input);
    roccvbench::FillTensor(gamma);

    for (int i = 0; i < config.runs; i++) {
        hipStream_t stream;
        hipEvent_t begin, end;
        HIP_VALIDATE_NO_ERRORS(hipEventCreate(&begin));
        HIP_VALIDATE_NO_ERRORS(hipEventCreate(&end));
        HIP_VALIDATE_NO_ERRORS(hipStreamCreate(&stream));

        GammaContrast op;
        HIP_VALIDATE_NO_ERRORS(hipEventRecord(begin, stream));
        op(nullptr, input, output, gamma, eDeviceType::CPU);
        HIP_VALIDATE_NO_ERRORS(hipEventRecord(end, stream));
        HIP_VALIDATE_NO_ERRORS(hipEventSynchronize(end));

        float execution_time;
        HIP_VALIDATE_NO_ERRORS(hipEventElapsedTime(&execution_time, begin, end));
        HIP_VALIDATE_NO_ERRORS(hipEventDestroy(begin));
        HIP_VALIDATE_NO_ERRORS(hipEventDestroy(end));
        HIP_VALIDATE_NO_ERRORS(hipStreamDestroy(stream));

        results.execution_time += execution_time / config.runs;
    }

    return results;
}