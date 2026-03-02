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

/**
 * @brief Generates a random OpenCV image as a matrix.
 *
 * @tparam T The base datatype of the generated data.
 * @param width Image width.
 * @param height Image height.
 * @param datatype OpenCV datatype for the image. (e.g. CV_U8C3)
 * @return A cv::Mat with randomly generated data.
 */
template <typename T>
inline cv::Mat GenerateMat(int width, int height, int datatype) {
    const size_t vecSize = width * height * CV_MAT_CN(datatype);
    std::vector<T> data = roccvbench::RandVector<T>(vecSize);

    cv::Mat mat(height, width, datatype, data.data());
    return mat.clone();
}

/**
 * @brief Creates a list of empty cv::Mat's to be used as output matrices.
 *
 * @param samples Number of images in the batch.
 * @param width Image width.
 * @param height Image height.
 * @param datatype OpenCV datatype for the image. (e.g. CV_U8C3)
 * @return A list of cv::Mat.
 */
inline std::vector<cv::Mat> CreateOutputMats(int samples, int width, int height, int datatype) {
    std::vector<cv::Mat> mats;
    mats.reserve(samples);
    for (int i = 0; i < samples; i++) {
        mats.push_back(cv::Mat(height, width, datatype));
    }
    return mats;
}

/**
 * @brief Generates list of random OpenCV images.
 *
 * @tparam T Base datatype of the generated data.
 * @param samples Number of images in the batch.
 * @param width Image width.
 * @param height Image height.
 * @param datatype OpenCV datatype for the image. (e.g. CV_U8C3)
 * @return A list of cv::Mat with randomly generated data.
 */
template <typename T>
inline std::vector<cv::Mat> GenerateMats(int samples, int width, int height, int datatype) {
    std::vector<cv::Mat> batch;
    batch.reserve(samples);
    for (int i = 0; i < samples; i++) {
        batch.push_back(GenerateMat<T>(width, height, datatype));
    }
    return batch;
}

/**
 * @brief Registers the memory usage of a cv::Mat.
 *
 * @param mat The cv::Mat to register the memory usage of.
 * @param memoryUsage The memory usage to register.
 */
inline void RegisterMemoryUsage(const cv::Mat& mat, size_t& memoryUsage) {
    memoryUsage += mat.total() * mat.elemSize();
}

/**
 * @brief Registers the memory usage of a list of cv::Mat.
 *
 * @param mats The list of cv::Mat to register the memory usage of.
 * @param memoryUsage The memory usage to register.
 */
inline void RegisterMemoryUsage(const std::vector<cv::Mat>& mats, size_t& memoryUsage) {
    for (const auto& mat : mats) {
        RegisterMemoryUsage(mat, memoryUsage);
    }
}