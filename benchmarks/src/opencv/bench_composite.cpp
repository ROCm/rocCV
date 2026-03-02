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

#include <opencv2/opencv.hpp>
#include <roccvbench/registry.hpp>
#include <roccvbench/utils.hpp>

#include "opencv_bench_helpers.hpp"

template <typename T, typename WeightType>
static roccvbench::BenchmarkResults RunCompositeBenchmark(roccvbench::BenchmarkParamsList params) {
    roccvbench::BenchmarkResults results;

    int samples = roccvbench::GetParamValue<int>(params, "samples");
    int width = roccvbench::GetParamValue<int>(params, "width");
    int height = roccvbench::GetParamValue<int>(params, "height");
    int runs = roccvbench::GetParamValue<int>(params, "runs");
    int warmupRuns = roccvbench::GetParamValue<int>(params, "warmupRuns");
    int in_format = roccvbench::GetParamValue<int>(params, "in_format");
    int alpha_format = roccvbench::GetParamValue<int>(params, "alpha_format");
    int out_format = roccvbench::GetParamValue<int>(params, "out_format");

    std::vector<cv::Mat> backgrounds = GenerateMats<T>(samples, width, height, in_format);
    std::vector<cv::Mat> foregrounds = GenerateMats<T>(samples, width, height, in_format);
    std::vector<cv::Mat> weights1 = GenerateMats<WeightType>(samples, width, height, alpha_format);
    std::vector<cv::Mat> weights2 = GenerateMats<WeightType>(samples, width, height, alpha_format);
    std::vector<cv::Mat> outputs = CreateOutputMats(samples, width, height, out_format);

    RegisterMemoryUsage(backgrounds, results.readMemoryBytes);
    RegisterMemoryUsage(foregrounds, results.readMemoryBytes);
    RegisterMemoryUsage(weights1, results.readMemoryBytes);
    RegisterMemoryUsage(weights2, results.readMemoryBytes);
    RegisterMemoryUsage(outputs, results.writtenMemoryBytes);

    ROCCV_BENCH_RECORD_BLOCK(
        {
            for (size_t i = 0; i < backgrounds.size(); i++) {
                cv::blendLinear(backgrounds[i], foregrounds[i], weights1[i], weights2[i], outputs[i]);
            }
        },
        results.executionTime, runs, warmupRuns);
    return results;
}

#define DEFINE_COMPOSITE_BENCHMARK(name, T, WeightType, in_format, alpha_format, out_format) \
    BENCHMARK_P(Composite, name,                                                             \
                BENCH_PARAMS(BENCH_PARAM_STR("in_format", in_format, #in_format),            \
                             BENCH_PARAM_STR("alpha_format", alpha_format, #alpha_format),   \
                             BENCH_PARAM_STR("out_format", out_format, #out_format))) {      \
        return RunCompositeBenchmark<T, WeightType>(params);                                 \
    }

DEFINE_COMPOSITE_BENCHMARK(OpenCV, uint8_t, float, CV_8UC3, CV_32F, CV_8UC3);