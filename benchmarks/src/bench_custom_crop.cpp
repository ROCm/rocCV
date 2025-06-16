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
#include <op_custom_crop.hpp>
#include <roccvbench/registry.hpp>
#include <roccvbench/utils.hpp>

using namespace roccv;

BENCHMARK(CustomCrop, GPU) {
    roccvbench::BenchmarkResults results;
    results.execution_time = 0.0f;

    TensorRequirements reqs = Tensor::CalcRequirements(
        TensorShape(TensorLayout(TENSOR_LAYOUT_NHWC), {config.batches, config.height, config.width, 3}),
        DataType(DATA_TYPE_U8));
    Tensor input(reqs);

    Box_t cropRect = {150, 50, 400, 300};
    Tensor output(TensorShape(TensorLayout(eTensorLayout::TENSOR_LAYOUT_NHWC),
                              {config.batches, cropRect.height, cropRect.width, 3}),
                  input.dtype());

    roccvbench::FillTensor(input);

    CustomCrop op;
    ROCCV_BENCH_RECORD_EXECUTION_TIME(op(nullptr, input, output, cropRect), results.execution_time, config.runs);

    return results;
}

BENCHMARK(CustomCrop, CPU) {
    roccvbench::BenchmarkResults results;
    results.execution_time = 0.0f;

    TensorRequirements reqs = Tensor::CalcRequirements(
        TensorShape(TensorLayout(TENSOR_LAYOUT_NHWC), {config.batches, config.height, config.width, 3}),
        DataType(DATA_TYPE_U8), eDeviceType::CPU);
    Tensor input(reqs);

    Box_t cropRect = {150, 50, 400, 300};
    Tensor output(TensorShape(TensorLayout(eTensorLayout::TENSOR_LAYOUT_NHWC),
                              {config.batches, cropRect.height, cropRect.width, 3}),
                  input.dtype(), eDeviceType::CPU);

    roccvbench::FillTensor(input);

    CustomCrop op;
    ROCCV_BENCH_RECORD_EXECUTION_TIME(op(nullptr, input, output, cropRect, eDeviceType::CPU), results.execution_time,
                                      config.runs);

    return results;
}