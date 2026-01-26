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

BENCHMARK(Composite, OpenCV) {
    roccvbench::BenchmarkResults results;

    std::vector<cv::Mat> backgrounds = GenerateMats<uint8_t>(config.samples, config.width, config.height, CV_8UC3);
    std::vector<cv::Mat> foregrounds = GenerateMats<uint8_t>(config.samples, config.width, config.height, CV_8UC3);
    std::vector<cv::Mat> weights1 = GenerateMats<float>(config.samples, config.width, config.height, CV_32F);
    std::vector<cv::Mat> weights2 = GenerateMats<float>(config.samples, config.width, config.height, CV_32F);
    std::vector<cv::Mat> outputs = CreateOutputMats(config.samples, config.width, config.height, CV_8UC3);

    RegisterMemoryUsage(backgrounds, results.inputMemoryBytes);
    RegisterMemoryUsage(foregrounds, results.inputMemoryBytes);
    RegisterMemoryUsage(weights1, results.inputMemoryBytes);
    RegisterMemoryUsage(weights2, results.inputMemoryBytes);
    RegisterMemoryUsage(outputs, results.outputMemoryBytes);

    ROCCV_BENCH_RECORD_BLOCK(
        {
            for (size_t i = 0; i < backgrounds.size(); i++) {
                cv::blendLinear(backgrounds[i], foregrounds[i], weights1[i], weights2[i], outputs[i]);
            }
        },
        results.executionTime, config.runs, config.warmupRuns);
    return results;
}