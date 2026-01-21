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

BENCHMARK(WarpPerspective, OpenCV) {
    roccvbench::BenchmarkResults results;

    std::vector<cv::Mat> mats = GenerateMats<uint8_t>(config.samples, config.width, config.height, CV_8UC3);
    std::vector<cv::Mat> outputs = CreateOutputMats(config.samples, config.width, config.height, CV_8UC3);

    std::vector<float> transformData = {1, 0, 0, 0, 1, 0, -0.001, 0, 1};
    cv::Mat transform(3, 3, CV_32F, transformData.data());

    RegisterMemoryUsage(mats, results.inputMemoryBytes);
    RegisterMemoryUsage(transform, results.inputMemoryBytes);
    RegisterMemoryUsage(outputs, results.outputMemoryBytes);

    ROCCV_BENCH_RECORD_BLOCK(
        {
            for (size_t i = 0; i < mats.size(); i++) {
                cv::warpPerspective(mats[i], outputs[i], transform, outputs[i].size(), cv::INTER_LINEAR,
                                    cv::BORDER_CONSTANT, 0);
            }
        },
        results.executionTime, config.runs);
    return results;
}