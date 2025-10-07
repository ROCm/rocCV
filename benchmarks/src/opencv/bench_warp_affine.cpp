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

BENCHMARK(WarpAffine, OpenCV) {
    roccvbench::BenchmarkResults results;

    std::vector<cv::Mat> mats = GenerateMats<uint8_t>(config.samples, config.width, config.height, CV_8UC3);
    cv::Mat outputMat(config.height, config.width, CV_8UC3);

    std::vector<float> affineMatData = {1, 0, 0, 1, -1, 120};
    cv::Mat affineMat(2, 3, CV_32F, affineMatData.data());

    ROCCV_BENCH_RECORD_BLOCK(
        {
            for (const auto& mat : mats) {
                cv::warpAffine(mat, outputMat, affineMat, outputMat.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, 0);
            }
        },
        results.executionTime, config.runs);
    return results;
}