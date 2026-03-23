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
#include <format>
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
 * All images must be of the same size and format. This operation will block on the provided stream.
 *
 * @param image_path The path to the image to load. If a directory is provided, all supported images in the directory
 * will be loaded.
 * @param device The device to load the images onto. Defaults to GPU.
 * @param openCVFlags The OpenCV flags to use when loading the images. Defaults to IMREAD_UNCHANGED.
 * @return A NHWC tensor containing the loaded images.
 */
inline roccv::Tensor LoadImages(hipStream_t stream, const std::string &image_path,
                                eDeviceType device = eDeviceType::GPU, int openCVFlags = cv::IMREAD_UNCHANGED) {
    const std::vector<std::string> supportedExtensions = {".bmp", ".jpg", ".jpeg", ".png"};

    std::vector<cv::Mat> images;

    int width = -1;
    int height = -1;
    int channels = -1;

    // Load images from directory or file if a single image is provided
    if (std::filesystem::is_directory(image_path)) {
        for (auto file : std::filesystem::directory_iterator(image_path)) {
            if (!std::filesystem::is_directory(file.path()) && ContainsExtension(file.path(), supportedExtensions)) {
                cv::Mat image = cv::imread(file.path(), openCVFlags);
                if (image.empty()) {
                    throw std::runtime_error("Cannot decode " + file.path().string() + ". File type not supported.\n");
                }
                images.push_back(image);

                // Check if all images are of the same size
                if (width == -1 && height == -1 && channels == -1) {
                    width = image.cols;
                    height = image.rows;
                    channels = image.channels();
                } else if (image.cols != width || image.rows != height || image.channels() != channels) {
                    throw std::runtime_error("All images must be of the same size and format");
                }
            }
        }
    } else if (std::filesystem::is_regular_file(image_path) && ContainsExtension(image_path, supportedExtensions)) {
        cv::Mat image = cv::imread(image_path, openCVFlags);
        if (image.empty()) {
            throw std::runtime_error("Cannot decode " + image_path + ". File type not supported.\n");
        }
        images.push_back(image);
        width = image.cols;
        height = image.rows;
        channels = image.channels();
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
        CHECK_HIP_ERROR(hipMemcpy2DAsync(static_cast<uint8_t *>(params.basePtr) + i * params.imageBytes,
                                         params.rowPitch, images[i].data, params.rowBytes, params.rowBytes, height,
                                         kind, stream));
    }

    // Ensure all memory operations are completed before returning the tensor
    CHECK_HIP_ERROR(hipStreamSynchronize(stream));

    return tensor;
}

/**
 * @brief Writes a batch of images from a tensor to the specified output path. This is a blocking operation.
 *
 * @param tensor The tensor to write the images from.
 * @param output_path The path to write the images to. If a directory is provided, the images will be written to the
 * directory.
 */
inline void WriteImages(hipStream_t stream, const roccv::Tensor &tensor, const std::string &output_path) {
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
        CHECK_HIP_ERROR(hipMemcpy2DAsync(images[i].data, params.rowBytes,
                                         static_cast<uint8_t *>(params.basePtr) + i * params.imageBytes,
                                         params.rowPitch, params.rowBytes, height, kind, stream));
    }

    // Ensure all memory operations are completed before writing images
    CHECK_HIP_ERROR(hipStreamSynchronize(stream));

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