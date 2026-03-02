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

template <typename T>
static roccvbench::BenchmarkResults RunCvtColorBenchmark(roccvbench::BenchmarkParamsList params) {
    roccvbench::BenchmarkResults results;

    int samples = roccvbench::GetParamValue<int>(params, "samples");
    int width = roccvbench::GetParamValue<int>(params, "width");
    int height = roccvbench::GetParamValue<int>(params, "height");
    int runs = roccvbench::GetParamValue<int>(params, "runs");
    int warmupRuns = roccvbench::GetParamValue<int>(params, "warmupRuns");
    int in_format = roccvbench::GetParamValue<int>(params, "in_format");
    int out_format = roccvbench::GetParamValue<int>(params, "out_format");
    cv::ColorConversionCodes color_conversion_code =
        roccvbench::GetParamValue<cv::ColorConversionCodes>(params, "color_conversion_code");

    std::vector<cv::Mat> mats = GenerateMats<T>(samples, width, height, in_format);
    std::vector<cv::Mat> outputs = CreateOutputMats(samples, width, height, out_format);

    RegisterMemoryUsage(mats, results.readMemoryBytes);
    RegisterMemoryUsage(outputs, results.writtenMemoryBytes);

    ROCCV_BENCH_RECORD_BLOCK(
        {
            for (size_t i = 0; i < mats.size(); i++) {
                cv::cvtColor(mats[i], outputs[i], color_conversion_code);
            }
        },
        results.executionTime, runs, warmupRuns);
    return results;
}

#define DEFINE_CVT_COLOR_BENCHMARK(name, T, in_format, out_format, color_conversion_code)    \
    BENCHMARK_P(CvtColor, name,                                                              \
                BENCH_PARAMS(BENCH_PARAM_STR("in_format", in_format, #in_format),            \
                             BENCH_PARAM_STR("out_format", out_format, #out_format),         \
                             BENCH_PARAM("color_conversion_code", color_conversion_code))) { \
        return RunCvtColorBenchmark<T>(params);                                              \
    }

DEFINE_CVT_COLOR_BENCHMARK(OpenCV, uint8_t, CV_8UC3, CV_8UC1, cv::COLOR_RGB2GRAY);
DEFINE_CVT_COLOR_BENCHMARK(OpenCV, uint8_t, CV_8UC3, CV_8UC1, cv::COLOR_BGR2GRAY);
DEFINE_CVT_COLOR_BENCHMARK(OpenCV, uint8_t, CV_8UC3, CV_8UC3, cv::COLOR_RGB2BGR);
DEFINE_CVT_COLOR_BENCHMARK(OpenCV, uint8_t, CV_8UC3, CV_8UC3, cv::COLOR_RGB2YUV);
DEFINE_CVT_COLOR_BENCHMARK(OpenCV, uint8_t, CV_8UC3, CV_8UC3, cv::COLOR_BGR2YUV);
DEFINE_CVT_COLOR_BENCHMARK(OpenCV, uint8_t, CV_8UC3, CV_8UC3, cv::COLOR_YUV2RGB);
DEFINE_CVT_COLOR_BENCHMARK(OpenCV, uint8_t, CV_8UC3, CV_8UC3, cv::COLOR_YUV2BGR);