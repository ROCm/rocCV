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

template <eDeviceType DeviceType>
static roccvbench::BenchmarkResults RunRotateBenchmark(roccvbench::BenchmarkParamsList params) {
    roccvbench::BenchmarkResults results;

    int samples = roccvbench::GetParamValue<int>(params, "samples");
    int width = roccvbench::GetParamValue<int>(params, "width");
    int height = roccvbench::GetParamValue<int>(params, "height");
    int runs = roccvbench::GetParamValue<int>(params, "runs");
    int warmupRuns = roccvbench::GetParamValue<int>(params, "warmupRuns");
    ImageFormat in_format = roccvbench::GetParamValue<ImageFormat>(params, "in_format");
    ImageFormat out_format = roccvbench::GetParamValue<ImageFormat>(params, "out_format");
    eInterpolationType interpolation = roccvbench::GetParamValue<eInterpolationType>(params, "interpolation");
    double angle = roccvbench::GetParamValue<double>(params, "angle");

    Tensor::Requirements inReqs = Tensor::CalcRequirements(samples, {width, height}, in_format, DeviceType);
    Tensor::Requirements outReqs = Tensor::CalcRequirements(samples, {width, height}, out_format, DeviceType);
    Tensor input(inReqs);
    Tensor output(outReqs);

    RegisterMemoryUsage(input, results.readMemoryBytes);
    RegisterMemoryUsage(output, results.writtenMemoryBytes);

    FillTensor(input);

    const double centerX = (width - 1) / 2.0;
    const double centerY = (height - 1) / 2.0;
    const double2 shift = ComputeCenterShift(centerX, centerY, angle);

    Rotate op;
    hipStream_t stream;
    HIP_VALIDATE_NO_ERRORS(hipStreamCreate(&stream));

    ROCCV_BENCH_RECORD_BLOCK(
        {
            op(stream, input, output, angle, shift, interpolation, DeviceType);
            if constexpr (DeviceType == eDeviceType::GPU) {
                HIP_VALIDATE_NO_ERRORS(hipStreamSynchronize(stream));
            }
        },
        results.executionTime, runs, warmupRuns);

    HIP_VALIDATE_NO_ERRORS(hipStreamDestroy(stream));

    return results;
}

#define DEFINE_ROTATE_BENCHMARK(name, device, in_format, out_format, interpolation, angle)                \
    BENCHMARK_P(Rotate, name,                                                                             \
                BENCH_PARAMS(BENCH_PARAM("in_format", in_format), BENCH_PARAM("out_format", out_format),  \
                             BENCH_PARAM("interpolation", interpolation), BENCH_PARAM("angle", angle))) { \
        return RunRotateBenchmark<device>(params);                                                        \
    }

DEFINE_ROTATE_BENCHMARK(GPU, eDeviceType::GPU, FMT_RGB8, FMT_RGB8, eInterpolationType::INTERP_TYPE_LINEAR, 180.0);
DEFINE_ROTATE_BENCHMARK(GPU, eDeviceType::GPU, FMT_RGBA8, FMT_RGBA8, eInterpolationType::INTERP_TYPE_LINEAR, 180.0);
DEFINE_ROTATE_BENCHMARK(GPU, eDeviceType::GPU, FMT_U8, FMT_U8, eInterpolationType::INTERP_TYPE_LINEAR, 180.0);
DEFINE_ROTATE_BENCHMARK(CPU, eDeviceType::CPU, FMT_RGB8, FMT_RGB8, eInterpolationType::INTERP_TYPE_LINEAR, 180.0);