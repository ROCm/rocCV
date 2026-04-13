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
#include <op_warp_affine.hpp>
#include <roccvbench/registry.hpp>
#include <roccvbench/utils.hpp>

#include "roccv_bench_helpers.hpp"

using namespace roccv;

template <eDeviceType DeviceType>
static roccvbench::BenchmarkResults RunWarpAffineBenchmark(roccvbench::BenchmarkParamsList params) {
    roccvbench::BenchmarkResults results;

    int samples = roccvbench::GetParamValue<int>(params, "samples");
    int width = roccvbench::GetParamValue<int>(params, "width");
    int height = roccvbench::GetParamValue<int>(params, "height");
    int runs = roccvbench::GetParamValue<int>(params, "runs");
    int warmupRuns = roccvbench::GetParamValue<int>(params, "warmupRuns");
    ImageFormat inFormat = roccvbench::GetParamValue<ImageFormat>(params, "inFormat");
    ImageFormat outFormat = roccvbench::GetParamValue<ImageFormat>(params, "outFormat");
    eInterpolationType interpolation = roccvbench::GetParamValue<eInterpolationType>(params, "interpolation");
    eBorderType border = roccvbench::GetParamValue<eBorderType>(params, "border");

    Tensor::Requirements inReqs = Tensor::CalcRequirements(samples, (Size2D){width, height}, inFormat, DeviceType);
    Tensor::Requirements outReqs = Tensor::CalcRequirements(samples, (Size2D){width, height}, outFormat, DeviceType);
    Tensor input(inReqs);
    Tensor output(outReqs);

    AffineTransform affineMatrix = {2.f, 2.f, 0.f, 3.f, 1.f, 0.f};

    FillTensor(input);

    RegisterMemoryUsage(input, results.readMemoryBytes);
    RegisterMemoryUsage(output, results.writtenMemoryBytes);

    WarpAffine op;
    hipStream_t stream;
    HIP_VALIDATE_NO_ERRORS(hipStreamCreate(&stream));

    ROCCV_BENCH_RECORD_BLOCK(
        {
            op(stream, input, output, affineMatrix, true, interpolation, border, make_float4(0.0f, 0.0f, 0.0f, 1.0f),
               DeviceType);
            if constexpr (DeviceType == eDeviceType::GPU) {
                HIP_VALIDATE_NO_ERRORS(hipStreamSynchronize(stream));
            }
        },
        results.executionTime, runs, warmupRuns);

    HIP_VALIDATE_NO_ERRORS(hipStreamDestroy(stream));

    return results;
}

#define DEFINE_WARP_AFFINE_BENCHMARK(name, device, inFormat, outFormat, interpolation, border)              \
    BENCHMARK_P(WarpAffine, name,                                                                           \
                BENCH_PARAMS(BENCH_PARAM("inFormat", inFormat), BENCH_PARAM("outFormat", outFormat),        \
                             BENCH_PARAM("interpolation", interpolation), BENCH_PARAM("border", border))) { \
        return RunWarpAffineBenchmark<device>(params);                                                      \
    }

// GPU benchmarks
DEFINE_WARP_AFFINE_BENCHMARK(GPU, eDeviceType::GPU, FMT_RGB8, FMT_RGB8, eInterpolationType::INTERP_TYPE_NEAREST,
                             eBorderType::BORDER_TYPE_CONSTANT);
DEFINE_WARP_AFFINE_BENCHMARK(GPU, eDeviceType::GPU, FMT_RGB8, FMT_RGB8, eInterpolationType::INTERP_TYPE_LINEAR,
                             eBorderType::BORDER_TYPE_CONSTANT);
DEFINE_WARP_AFFINE_BENCHMARK(GPU, eDeviceType::GPU, FMT_RGB8, FMT_RGB8, eInterpolationType::INTERP_TYPE_CUBIC,
                             eBorderType::BORDER_TYPE_CONSTANT);
DEFINE_WARP_AFFINE_BENCHMARK(GPU, eDeviceType::GPU, FMT_RGBA8, FMT_RGBA8, eInterpolationType::INTERP_TYPE_NEAREST,
                             eBorderType::BORDER_TYPE_CONSTANT);
DEFINE_WARP_AFFINE_BENCHMARK(GPU, eDeviceType::GPU, FMT_RGBA8, FMT_RGBA8, eInterpolationType::INTERP_TYPE_LINEAR,
                             eBorderType::BORDER_TYPE_CONSTANT);
DEFINE_WARP_AFFINE_BENCHMARK(GPU, eDeviceType::GPU, FMT_RGBA8, FMT_RGBA8, eInterpolationType::INTERP_TYPE_CUBIC,
                             eBorderType::BORDER_TYPE_CONSTANT);
DEFINE_WARP_AFFINE_BENCHMARK(GPU, eDeviceType::GPU, FMT_U8, FMT_U8, eInterpolationType::INTERP_TYPE_NEAREST,
                             eBorderType::BORDER_TYPE_CONSTANT);
DEFINE_WARP_AFFINE_BENCHMARK(GPU, eDeviceType::GPU, FMT_U8, FMT_U8, eInterpolationType::INTERP_TYPE_LINEAR,
                             eBorderType::BORDER_TYPE_CONSTANT);
DEFINE_WARP_AFFINE_BENCHMARK(GPU, eDeviceType::GPU, FMT_U8, FMT_U8, eInterpolationType::INTERP_TYPE_CUBIC,
                             eBorderType::BORDER_TYPE_CONSTANT);
DEFINE_WARP_AFFINE_BENCHMARK(GPU, eDeviceType::GPU, FMT_F32, FMT_F32, eInterpolationType::INTERP_TYPE_NEAREST,
                             eBorderType::BORDER_TYPE_CONSTANT);
DEFINE_WARP_AFFINE_BENCHMARK(GPU, eDeviceType::GPU, FMT_F32, FMT_F32, eInterpolationType::INTERP_TYPE_LINEAR,
                             eBorderType::BORDER_TYPE_CONSTANT);
DEFINE_WARP_AFFINE_BENCHMARK(GPU, eDeviceType::GPU, FMT_F32, FMT_F32, eInterpolationType::INTERP_TYPE_CUBIC,
                             eBorderType::BORDER_TYPE_CONSTANT);
DEFINE_WARP_AFFINE_BENCHMARK(GPU, eDeviceType::GPU, FMT_F32, FMT_F32, eInterpolationType::INTERP_TYPE_CUBIC,
                             eBorderType::BORDER_TYPE_REFLECT);
DEFINE_WARP_AFFINE_BENCHMARK(GPU, eDeviceType::GPU, FMT_U8, FMT_U8, eInterpolationType::INTERP_TYPE_CUBIC,
                             eBorderType::BORDER_TYPE_REFLECT);
DEFINE_WARP_AFFINE_BENCHMARK(GPU, eDeviceType::GPU, FMT_RGB8, FMT_RGB8, eInterpolationType::INTERP_TYPE_CUBIC,
                             eBorderType::BORDER_TYPE_REFLECT);

// CPU benchmarks
DEFINE_WARP_AFFINE_BENCHMARK(CPU, eDeviceType::CPU, FMT_RGB8, FMT_RGB8, eInterpolationType::INTERP_TYPE_NEAREST,
                             eBorderType::BORDER_TYPE_CONSTANT);
DEFINE_WARP_AFFINE_BENCHMARK(CPU, eDeviceType::CPU, FMT_RGB8, FMT_RGB8, eInterpolationType::INTERP_TYPE_LINEAR,
                             eBorderType::BORDER_TYPE_CONSTANT);
DEFINE_WARP_AFFINE_BENCHMARK(CPU, eDeviceType::CPU, FMT_RGB8, FMT_RGB8, eInterpolationType::INTERP_TYPE_CUBIC,
                             eBorderType::BORDER_TYPE_CONSTANT);