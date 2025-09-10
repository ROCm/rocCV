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
#include <op_resize.hpp>
#include <roccvbench/registry.hpp>
#include <roccvbench/utils.hpp>

using namespace roccv;

BENCHMARK(Resize, GPU) {
    roccvbench::BenchmarkResults results;
    results.executionTime = 0.0f;

    Tensor::Requirements inputReqs =
        Tensor::CalcRequirements(config.samples, (Size2D){config.width, config.height}, FMT_RGB8);
    Tensor::Requirements outputReqs =
        Tensor::CalcRequirements(config.samples, (Size2D){config.width * 2, config.height * 2}, FMT_RGB8);
    Tensor input(inputReqs);
    Tensor output(outputReqs);

    roccvbench::FillTensor(input);

    Resize op;
    ROCCV_BENCH_RECORD_EXECUTION_TIME_HIP(op(nullptr, input, output, eInterpolationType::INTERP_TYPE_LINEAR),
                                          results.executionTime, config.runs);

    return results;
}

BENCHMARK(Resize, CPU) {
    roccvbench::BenchmarkResults results;
    results.executionTime = 0.0f;

    Tensor::Requirements inputReqs =
        Tensor::CalcRequirements(config.samples, (Size2D){config.width, config.height}, FMT_RGB8, eDeviceType::CPU);
    Tensor::Requirements outputReqs = Tensor::CalcRequirements(
        config.samples, (Size2D){config.width * 2, config.height * 2}, FMT_RGB8, eDeviceType::CPU);
    Tensor input(inputReqs);
    Tensor output(outputReqs);

    roccvbench::FillTensor(input);

    Resize op;
    ROCCV_BENCH_RECORD_EXECUTION_TIME_HOST(
        op(nullptr, input, output, eInterpolationType::INTERP_TYPE_LINEAR, eDeviceType::CPU), results.executionTime,
        config.runs);

    return results;
}