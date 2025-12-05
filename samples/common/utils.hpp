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

inline bool ContainsExtension(const std::filesystem::path &path, const std::vector<std::string> &extension_list) {
    for (auto extension : extension_list) {
        if (path.extension() == extension) return true;
    }

    return false;
}

struct MemcpyParams {
    void *basePtr = nullptr;  // Base pointer to the tensor data
    size_t rowPitch = 0;      // Number of bytes per row, including padding
    size_t rowBytes = 0;      // Number of bytes per row, not including padding
    size_t imageBytes = 0;    // Number of bytes per image, including padding (rowBytes * height)
};

/**
 * @brief Gets the memcpy parameters for a tensor to perform a memcpy2D operation.
 *
 * @param tensor The tensor to get the memcpy parameters for.
 * @return The memcpy parameters to perform a memcpy2D operation.
 */
inline MemcpyParams GetMemcpyParams(const roccv::Tensor &tensor) {
    MemcpyParams params;

    roccv::TensorDataStrided tensorData = tensor.exportData<roccv::TensorDataStrided>();
    params.rowPitch = tensorData.stride(tensor.layout().height_index());
    params.rowBytes = tensor.shape(tensor.layout().width_index()) * tensor.shape(tensor.layout().channels_index()) *
                      tensor.dtype().size();
    params.imageBytes = params.rowPitch * tensor.shape(tensor.layout().height_index());
    params.basePtr = tensorData.basePtr();

    return params;
}

/**
 * @brief Loads an image, or multiple images if given a directory, into a tensor. Will be in NHWC layout and U8 format.
 * All images must be of the same size and format. This is a blocking operation.
 *
 * @param image_path The path to the image to load. If a directory is provided, all supported images in the directory
 * will be loaded.
 * @param device The device to load the images onto. Defaults to GPU.
 * @return A NHWC tensor containing the loaded images.
 */
inline roccv::Tensor LoadImages(const std::string &image_path, eDeviceType device = eDeviceType::GPU) {
    const std::vector<std::string> supportedExtensions = {".bmp", ".jpg", ".jpeg", ".png"};

    std::vector<cv::Mat> images;

    int width = -1;
    int height = -1;
    int channels = -1;

    // Load images from directory or file if a single image is provided
    if (std::filesystem::is_directory(image_path)) {
        for (auto file : std::filesystem::directory_iterator(image_path)) {
            if (!std::filesystem::is_directory(file.path()) && ContainsExtension(file.path(), supportedExtensions)) {
                images.push_back(cv::imread(file.path()));

                // Check if all images are of the same size
                if (width == -1 && height == -1 && channels == -1) {
                    width = images.back().cols;
                    height = images.back().rows;
                    channels = images.back().channels();
                } else if (images.back().cols != width || images.back().rows != height ||
                           images.back().channels() != channels) {
                    throw std::runtime_error("All images must be of the same size and format");
                }
            }
        }
    } else if (std::filesystem::is_regular_file(image_path) && ContainsExtension(image_path, supportedExtensions)) {
        images.push_back(cv::imread(image_path));
        width = images.back().cols;
        height = images.back().rows;
        channels = images.back().channels();
    } else {
        throw std::runtime_error("Cannot decode " + image_path + ". File type not supported.\n");
    }

    if (images.empty()) {
        throw std::runtime_error("No valid images found in directory " + image_path);
    }

    // Create tensor and prepare arguments for hipMemcpy2D
    roccv::Tensor tensor(images.size(), roccv::Size2D(width, height),
                         roccv::ImageFormat(eDataType::DATA_TYPE_U8, channels), device);

    MemcpyParams params = GetMemcpyParams(tensor);

    // Copy images into tensor
    hipMemcpyKind kind = (device == eDeviceType::GPU) ? hipMemcpyHostToDevice : hipMemcpyHostToHost;
    for (int i = 0; i < images.size(); i++) {
        CHECK_HIP_ERROR(hipMemcpy2D(static_cast<uint8_t *>(params.basePtr) + i * params.imageBytes, params.rowPitch,
                                    images[i].data, params.rowBytes, params.rowBytes, height, kind));
    }
    CHECK_HIP_ERROR(hipDeviceSynchronize());

    return tensor;
}

/**
 * @brief Writes a batch of images from a tensor to the specified output path. This is a blocking operation.
 *
 * @param tensor The tensor to write the images from.
 * @param output_path The path to write the images to. If a directory is provided, the images will be written to the
 * directory.
 */
inline void WriteImages(const roccv::Tensor &tensor, const std::string &output_path) {
    if (tensor.layout() != eTensorLayout::TENSOR_LAYOUT_NHWC && tensor.layout() != eTensorLayout::TENSOR_LAYOUT_HWC) {
        throw std::runtime_error(
            "Unsupported tensor layout in WriteImages(). Only NHWC and HWC layouts are supported.");
    }

    int64_t height = tensor.shape(tensor.layout().height_index());
    int64_t width = tensor.shape(tensor.layout().width_index());
    int64_t batchSize =
        tensor.layout().batch_index() < 0 ? 1 : tensor.shape(tensor.layout().batch_index());  // Support for HWC layout
    int64_t channels = tensor.shape(tensor.layout().channels_index());

    // Get OpenCV image format
    int64_t cvFormat = CV_MAKETYPE(CV_8U, channels);

    // Get memcpy parameters
    MemcpyParams params = GetMemcpyParams(tensor);
    hipMemcpyKind kind = (tensor.device() == eDeviceType::GPU) ? hipMemcpyDeviceToHost : hipMemcpyHostToHost;

    // Copy images from tensor to OpenCV image vector
    std::vector<cv::Mat> images(batchSize);
    for (int i = 0; i < batchSize; i++) {
        images[i] = cv::Mat(height, width, cvFormat);
        CHECK_HIP_ERROR(hipMemcpy2D(images[i].data, params.rowBytes,
                                    static_cast<uint8_t *>(params.basePtr) + i * params.imageBytes, params.rowPitch,
                                    params.rowBytes, height, kind));
    }
    CHECK_HIP_ERROR(hipDeviceSynchronize());

    std::filesystem::path outputPath(output_path);
    if (outputPath.extension().empty()) {
        for (int i = 0; i < batchSize; i++) {
            std::filesystem::create_directories(outputPath);
            std::filesystem::path outFilename = outputPath / std::format("image_{}.bmp", i);
            cv::imwrite(outFilename.string(), images[i]);
        }
    } else {
        cv::imwrite(output_path, images[0]);
    }
}

/**
 * @brief Loads images into the GPU memory specified.
 *
 * @param images_dir Either a directory or a single image to load into GPU memory.
 * @param num_images The number of images to load into GPU memory.
 * @param gpu_input A pointer to valid GPU memory.
 */
inline void DecodeRGBIImage(const std::string &images_dir, int num_images, void *gpu_input) {
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
inline void WriteRGBITensor(const roccv::Tensor &tensor, hipStream_t stream) {
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