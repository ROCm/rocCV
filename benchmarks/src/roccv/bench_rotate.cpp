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
#include <op_rotate.hpp>
#include <roccvbench/registry.hpp>
#include <roccvbench/utils.hpp>

#include "roccv_bench_helpers.hpp"

using namespace roccv;

namespace {
/**
 * @brief Computes the shift required to move the resulting rotated image back to the center of the image.
 *
 * @param centerX The x coordinate for the center of the image.
 * @param centerY The y coordinate for the center of the image.
 * @param angle The angle in degrees the resulting image will be rotated.
 * @return A double2 with the shift required to translate the image back to its center after a rotation.
 */
double2 ComputeCenterShift(const double centerX, const double centerY, const double angle) {
    double xShift = (1 - cos(angle * M_PI / 180)) * centerX - sin(angle * M_PI / 180) * centerY;
    double yShift = sin(angle * M_PI / 180) * centerX + (1 - cos(angle * M_PI / 180)) * centerY;
    return {xShift, yShift};
}
}  // namespace

BENCHMARK(Rotate, GPU) {
    roccvbench::BenchmarkResults results;

    TensorRequirements reqs = Tensor::CalcRequirements(
        TensorShape(TensorLayout(TENSOR_LAYOUT_NHWC), {config.samples, config.height, config.width, 3}),
        DataType(DATA_TYPE_U8));
    Tensor input(reqs);
    Tensor output(reqs);

    RegisterMemoryUsage(input, results.readMemoryBytes);
    RegisterMemoryUsage(output, results.writtenMemoryBytes);

    FillTensor(input);

    const double angle = 180;
    const double centerX = (config.width - 1) / 2.0;
    const double centerY = (config.height - 1) / 2.0;
    const double2 shift = ComputeCenterShift(centerX, centerY, angle);

    Rotate op;
    hipStream_t stream;
    HIP_VALIDATE_NO_ERRORS(hipStreamCreate(&stream));

    ROCCV_BENCH_RECORD_BLOCK(
        {
            op(stream, input, output, angle, shift, eInterpolationType::INTERP_TYPE_LINEAR);
            HIP_VALIDATE_NO_ERRORS(hipStreamSynchronize(stream))
        },
        results.executionTime, config.runs, config.warmupRuns);

    HIP_VALIDATE_NO_ERRORS(hipStreamDestroy(stream));

    return results;
}

BENCHMARK(Rotate, CPU) {
    roccvbench::BenchmarkResults results;

    TensorRequirements reqs = Tensor::CalcRequirements(
        TensorShape(TensorLayout(TENSOR_LAYOUT_NHWC), {config.samples, config.height, config.width, 3}),
        DataType(DATA_TYPE_U8), eDeviceType::CPU);
    Tensor input(reqs);
    Tensor output(reqs);

    RegisterMemoryUsage(input, results.readMemoryBytes);
    RegisterMemoryUsage(output, results.writtenMemoryBytes);

    FillTensor(input);

    const double angle = 180;
    const double centerX = (config.width - 1) / 2.0;
    const double centerY = (config.height - 1) / 2.0;
    const double2 shift = ComputeCenterShift(centerX, centerY, angle);

    Rotate op;
    ROCCV_BENCH_RECORD_BLOCK(
        { op(nullptr, input, output, angle, shift, eInterpolationType::INTERP_TYPE_LINEAR, eDeviceType::CPU); },
        results.executionTime, config.runs, config.warmupRuns);

    return results;
}