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

#include <roccvbench/registry.hpp>
#include <roccvbench/utils.hpp>

#include "opencv_bench_helpers.hpp"

template <typename T>
static roccvbench::BenchmarkResults RunCopyMakeBorderBenchmark(roccvbench::BenchmarkParamsList params) {
    roccvbench::BenchmarkResults results;

    int samples = roccvbench::GetParamValue<int>(params, "samples");
    int width = roccvbench::GetParamValue<int>(params, "width");
    int height = roccvbench::GetParamValue<int>(params, "height");
    int runs = roccvbench::GetParamValue<int>(params, "runs");
    int warmupRuns = roccvbench::GetParamValue<int>(params, "warmupRuns");
    int inFormat = roccvbench::GetParamValue<int>(params, "inFormat");
    int outFormat = roccvbench::GetParamValue<int>(params, "outFormat");
    int border = roccvbench::GetParamValue<int>(params, "border");
    int top = roccvbench::GetParamValue<int>(params, "borderTop");
    int left = roccvbench::GetParamValue<int>(params, "borderLeft");

    std::vector<cv::Mat> mats = GenerateMats<T>(samples, width, height, inFormat);
    std::vector<cv::Mat> outputs = CreateOutputMats(samples, width + left * 2, height + top * 2, outFormat);

    RegisterMemoryUsage(mats, results.readMemoryBytes);
    RegisterMemoryUsage(outputs, results.writtenMemoryBytes);

    roccvbench::RecordRunsCpu(runs, warmupRuns, results.executionTimes, [&]() {
        for (size_t i = 0; i < mats.size(); i++) {
            cv::copyMakeBorder(mats[i], outputs[i], top, top, left, left, border, 0);
        }
    });

    return results;
}
#define DEFINE_COPY_MAKE_BORDER_BENCHMARK(name, T, inFormat, outFormat, border, borderTop, borderLeft)                \
    BENCHMARK_P(                                                                                                      \
        CopyMakeBorder, name,                                                                                         \
        BENCH_PARAMS(BENCH_PARAM_STR("inFormat", inFormat, #inFormat),                                                \
                     BENCH_PARAM_STR("outFormat", outFormat, #outFormat), BENCH_PARAM_STR("border", border, #border), \
                     BENCH_PARAM("borderTop", borderTop), BENCH_PARAM("borderLeft", borderLeft))) {                   \
        return RunCopyMakeBorderBenchmark<T>(params);                                                                 \
    }

DEFINE_COPY_MAKE_BORDER_BENCHMARK(OpenCV, uint8_t, CV_8UC3, CV_8UC3, CV_HAL_BORDER_CONSTANT, 9, 9);
DEFINE_COPY_MAKE_BORDER_BENCHMARK(OpenCV, uint8_t, CV_8UC3, CV_8UC3, CV_HAL_BORDER_REPLICATE, 9, 9);
DEFINE_COPY_MAKE_BORDER_BENCHMARK(OpenCV, uint8_t, CV_8UC3, CV_8UC3, CV_HAL_BORDER_REFLECT, 9, 9);
DEFINE_COPY_MAKE_BORDER_BENCHMARK(OpenCV, uint8_t, CV_8UC3, CV_8UC3, CV_HAL_BORDER_REFLECT_101, 9, 9);
DEFINE_COPY_MAKE_BORDER_BENCHMARK(OpenCV, uint8_t, CV_8UC3, CV_8UC3, CV_HAL_BORDER_WRAP, 9, 9);
