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
    int inFormat = roccvbench::GetParamValue<int>(params, "inFormat");
    int outFormat = roccvbench::GetParamValue<int>(params, "outFormat");
    cv::ColorConversionCodes code = roccvbench::GetParamValue<cv::ColorConversionCodes>(params, "code");

    std::vector<cv::Mat> mats = GenerateMats<T>(samples, width, height, inFormat);
    std::vector<cv::Mat> outputs = CreateOutputMats(samples, width, height, outFormat);

    RegisterMemoryUsage(mats, results.readMemoryBytes);
    RegisterMemoryUsage(outputs, results.writtenMemoryBytes);

    roccvbench::RecordRunsCpu(runs, warmupRuns, results.executionTimes, [&]() {
        for (size_t i = 0; i < mats.size(); i++) {
            cv::cvtColor(mats[i], outputs[i], code);
        }
    });
    return results;
}

#define DEFINE_CVT_COLOR_BENCHMARK(name, T, inFormat, outFormat, code)                                          \
    BENCHMARK_P(CvtColor, name,                                                                                 \
                BENCH_PARAMS(BENCH_PARAM_STR("inFormat", inFormat, #inFormat),                                  \
                             BENCH_PARAM_STR("outFormat", outFormat, #outFormat), BENCH_PARAM("code", code))) { \
        return RunCvtColorBenchmark<T>(params);                                                                 \
    }

DEFINE_CVT_COLOR_BENCHMARK(OpenCV, uint8_t, CV_8UC3, CV_8UC1, cv::COLOR_RGB2GRAY);
DEFINE_CVT_COLOR_BENCHMARK(OpenCV, uint8_t, CV_8UC3, CV_8UC1, cv::COLOR_BGR2GRAY);
DEFINE_CVT_COLOR_BENCHMARK(OpenCV, uint8_t, CV_8UC3, CV_8UC3, cv::COLOR_RGB2BGR);
DEFINE_CVT_COLOR_BENCHMARK(OpenCV, uint8_t, CV_8UC3, CV_8UC3, cv::COLOR_RGB2YUV);
DEFINE_CVT_COLOR_BENCHMARK(OpenCV, uint8_t, CV_8UC3, CV_8UC3, cv::COLOR_BGR2YUV);
DEFINE_CVT_COLOR_BENCHMARK(OpenCV, uint8_t, CV_8UC3, CV_8UC3, cv::COLOR_YUV2RGB);
DEFINE_CVT_COLOR_BENCHMARK(OpenCV, uint8_t, CV_8UC3, CV_8UC3, cv::COLOR_YUV2BGR);