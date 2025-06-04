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

#include <core/tensor.hpp>
#include <filesystem>
#include <opencv2/opencv.hpp>

inline void CheckHIPError(hipError_t code, const char *file, const int line) {
    if (code != hipSuccess) {
        const char *errorMessage = hipGetErrorString(code);
        const std::string message = "HIP error returned at " + std::string(file) + ":" + std::to_string(line) +
                                    ", Error code: " + std::to_string(code) + " (" + std::string(errorMessage) + ")";
        throw std::runtime_error(message);
    }
}

#define CHECK_HIP_ERROR(val)                      \
    {                                             \
        CheckHIPError((val), __FILE__, __LINE__); \
    }

bool ContainsExtension(const std::filesystem::path &path, const std::vector<std::string> &extension_list) {
    for (auto extension : extension_list) {
        if (path.extension() == extension) return true;
    }

    return false;
}

/**
 * @brief Loads images into the GPU memory specified.
 *
 * @param images_dir Either a directory or a single image to load into GPU memory.
 * @param num_images The number of images to load into GPU memory.
 * @param gpu_input A pointer to valid GPU memory.
 */
void DecodeRGBIImage(const std::string &images_dir, int num_images, void *gpu_input) {
    const std::vector<std::string> supportedExtensions = {".bmp", ".jpg", ".jpeg", ".png"};

    std::vector<std::string> imageFiles;
    if (std::filesystem::is_directory(images_dir)) {
        // A directory is provided. Collect all supported files in the directory (non-recursively).
        for (auto file : std::filesystem::directory_iterator(images_dir)) {
            if (!std::filesystem::is_directory(file.path()) && ContainsExtension(file.path(), supportedExtensions)) {
                imageFiles.push_back(file.path());
            }
        }

        // Throw an error if there were no valid images found in the given directory
        if (imageFiles.empty()) {
            throw std::runtime_error("No valid images found in directory " + images_dir);
        }
    } else {
        // A single image file is provided
        if (!ContainsExtension(images_dir, supportedExtensions))
            throw std::runtime_error("Cannot decode " + images_dir + ". File type not supported.\n");
        imageFiles.push_back(images_dir);
    }

    // Load images into provided GPU memory
    size_t mem_offset = 0;
    for (int b = 0; b < num_images; b++) {
        cv::Mat inputMat = cv::imread(imageFiles[b]);
        if (inputMat.empty()) {
            throw std::runtime_error("Unable to load image " + imageFiles[b]);
        }

        size_t imageSize = inputMat.rows * inputMat.cols * inputMat.channels() * sizeof(uint8_t);
        CHECK_HIP_ERROR(
            hipMemcpy(static_cast<uint8_t *>(gpu_input) + mem_offset, inputMat.data, imageSize, hipMemcpyHostToDevice));
        mem_offset += imageSize;
    }
}

/**
 * @brief Writes a batch of 3-channel RGBI images in a tensor to .bmp files. This will also block on the provided
 * stream.
 *
 * @param tensor A tensor containing a batch of RGBI images.
 * @param stream The HIP stream to synchronize with.
 */
void WriteRGBITensor(const roccv::Tensor &tensor, hipStream_t stream) {
    CHECK_HIP_ERROR(hipStreamSynchronize(stream));

    auto srcData = tensor.exportData<roccv::TensorDataStrided>();
    int batchSize = tensor.shape(tensor.layout().batch_index());
    int height = tensor.shape(tensor.layout().height_index());
    int width = tensor.shape(tensor.layout().width_index());

    // Write each image in the batch to separate .bmp files
    for (int b = 0; b < batchSize; b++) {
        std::ostringstream outFilename;
        outFilename << "./roccvtest_" << b << ".bmp";

        cv::Mat outputMat(height, width, CV_8UC3);
        CHECK_HIP_ERROR(hipMemcpy(outputMat.data, srcData.basePtr(),
                                  (tensor.shape().size() / batchSize) * tensor.dtype().size(), hipMemcpyDeviceToHost));
        cv::imwrite(outFilename.str().c_str(), outputMat);
    }
}