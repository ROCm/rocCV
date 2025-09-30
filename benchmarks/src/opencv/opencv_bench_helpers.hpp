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

#pragma once

#include <opencv2/opencv.hpp>
#include <roccvbench/utils.hpp>

template <typename T>
cv::Mat GenerateMat(int width, int height, int datatype) {
    const size_t vecSize = width * height * CV_MAT_CN(datatype);
    std::vector<T> data = roccvbench::RandVector<T>(vecSize);

    cv::Mat mat(height, width, datatype, data.data());
    return mat.clone();
}

template <typename T>
std::vector<cv::Mat> GenerateMats(int samples, int width, int height, int datatype) {
    std::vector<cv::Mat> batch;
    batch.reserve(samples);
    for (int i = 0; i < samples; i++) {
        batch.push_back(GenerateMat<T>(width, height, datatype));
    }
    return batch;
}