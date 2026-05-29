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

#include <getopt.h>
#include <stdint.h>

#include <algorithm>
#include <core/image.hpp>
#include <core/image_batch_var_shape.hpp>
#include <core/image_format.hpp>
#include <core/tensor.hpp>
#include <filesystem>
#include <iostream>
#include <op_resize.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "common/utils.hpp"

/**
 * @brief Variable-shape Resize sample app.
 *
 * Loads every image in a directory (each potentially a different size) into a single
 * roccv::ImageBatchVarShape, then resizes the whole batch into one uniform, constant-sized
 * output tensor using the variable-shape Resize overload. This is a GPU-only path.
 *
 * Image Directory -> ImageBatchVarShape -> Resize (varshape) -> WriteImage
 */

void ShowUsage() {
    std::cout << "usage: ./resize_var_shape -i <image directory> [-W <width>] [-H <height>]\n"
              << "  -i  Path to a directory of images (or a single image file). Required.\n"
              << "  -W  Output width in pixels.  Optional; default: 224.\n"
              << "  -H  Output height in pixels. Optional; default: 224.\n"
              << "  -h  Show this help message.\n"
              << "Resized images are written to ./output/image_<index>.bmp.\n";
}

int ParseArgs(int argc, char *argv[], std::string &imageDir, int &outWidth, int &outHeight) {
    static struct option long_options[] = {{"help", no_argument, 0, 'h'},
                                           {"imageDir", required_argument, 0, 'i'},
                                           {"width", required_argument, 0, 'W'},
                                           {"height", required_argument, 0, 'H'},
                                           {0, 0, 0, 0}};

    int long_index = 0;
    int opt = 0;
    while ((opt = getopt_long(argc, argv, "hi:W:H:", long_options, &long_index)) != -1) {
        switch (opt) {
            case 'h':
                ShowUsage();
                return -1;
            case 'i':
                imageDir = optarg;
                break;
            case 'W':
                outWidth = std::stoi(optarg);
                break;
            case 'H':
                outHeight = std::stoi(optarg);
                break;
            default:
                ShowUsage();
                return -1;
        }
    }

    if (imageDir.empty()) {
        ShowUsage();
        std::cerr << "Error: an input image directory (-i) is required.\n";
        return -1;
    }
    if (outWidth <= 0 || outHeight <= 0) {
        ShowUsage();
        std::cerr << "Error: output width/height must be positive.\n";
        return -1;
    }
    return 0;
}

/**
 * @brief Collects supported image files from a directory (non-recursive) or accepts a single image file.
 */
std::vector<std::string> CollectImageFiles(const std::string &path) {
    const std::vector<std::string> supportedExtensions = {".bmp", ".jpg", ".jpeg", ".png"};

    std::vector<std::string> imageFiles;
    if (std::filesystem::is_directory(path)) {
        for (const auto &entry : std::filesystem::directory_iterator(path)) {
            if (!std::filesystem::is_directory(entry.path()) && ContainsExtension(entry.path(), supportedExtensions)) {
                imageFiles.push_back(entry.path().string());
            }
        }
        // Sort for a deterministic batch order across runs.
        std::sort(imageFiles.begin(), imageFiles.end());
    } else {
        if (!ContainsExtension(path, supportedExtensions)) {
            throw std::runtime_error("Cannot decode " + path + ". File type not supported.");
        }
        imageFiles.push_back(path);
    }

    if (imageFiles.empty()) {
        throw std::runtime_error("No supported images found in " + path);
    }
    return imageFiles;
}

int main(int argc, char *argv[]) {
    std::string imageDir;
    int outWidth = 224;
    int outHeight = 224;

    int retval = ParseArgs(argc, argv, imageDir, outWidth, outHeight);
    if (retval != 0) {
        return retval;
    }

    try {
        // tag: Create the HIP stream
        hipStream_t stream;
        CHECK_HIP_ERROR(hipStreamCreate(&stream));

        // tag: Gather input images
        std::vector<std::string> imageFiles = CollectImageFiles(imageDir);
        const int numImages = static_cast<int>(imageFiles.size());
        std::cout << "Found " << numImages << " image(s) in " << imageDir << ".\n";

        // tag: Build a variable-shape image batch
        // Each image keeps its own (potentially unique) dimensions. The batch holds GPU-resident images; the
        // input is loaded host-side via OpenCV and copied straight into each image's device plane.
        roccv::ImageBatchVarShape batch(numImages);

        for (const std::string &file : imageFiles) {
            // OpenCV loads as 8-bit BGR; we treat the interleaved 3-channel buffer as FMT_RGB8 without a color
            // conversion (output is written back through OpenCV, so the channel order round-trips consistently).
            cv::Mat inputMat = cv::imread(file, cv::IMREAD_COLOR);
            if (inputMat.empty()) {
                throw std::runtime_error("Unable to load image " + file);
            }

            roccv::Size2D size{inputMat.cols, inputMat.rows};
            roccv::Image image(size, roccv::FMT_RGB8, eDeviceType::GPU);

            // Copy host pixels into the image's GPU plane, honoring both the source (cv::Mat) and destination
            // (rocCV plane) row strides.
            auto imageData = image.exportData<roccv::ImageDataStridedHip>();
            const roccv::ImagePlaneStrided &plane = imageData.plane(0);
            const size_t rowBytes = static_cast<size_t>(inputMat.cols) * inputMat.channels() * sizeof(uint8_t);
            CHECK_HIP_ERROR(hipMemcpy2D(plane.basePtr, plane.rowStride, inputMat.data, inputMat.step, rowBytes,
                                        inputMat.rows, hipMemcpyHostToDevice));

            batch.pushBack(image);
            std::cout << "  loaded " << file << " (" << size.w << "x" << size.h << ")\n";
        }

        // tag: Allocate the uniform, constant-sized output tensor (NHWC, GPU)
        roccv::Tensor outputTensor(numImages, {outWidth, outHeight}, roccv::FMT_RGB8);

        // tag: Run the variable-shape Resize. Every image in the batch is resized into the same output size.
        std::cout << "Resizing batch to " << outWidth << "x" << outHeight << " (LINEAR interpolation)...\n";
        roccv::Resize resizeOp;
        resizeOp(stream, batch, outputTensor, INTERP_TYPE_LINEAR, eDeviceType::GPU);

        // tag: Copy results back to the host and write each resized image to ./output/image_<index>.bmp
        WriteImages(stream, outputTensor, "./output");
        std::cout << "Wrote " << numImages << " resized image(s) to ./output/image_<index>.bmp\n";

        // tag: Clean up
        CHECK_HIP_ERROR(hipStreamDestroy(stream));
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
