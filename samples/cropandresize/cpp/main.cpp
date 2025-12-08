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
#include <math.h>
#include <stdint.h>

#include <chrono>
#include <core/image_format.hpp>
#include <core/tensor.hpp>
#include <iostream>
#include <op_custom_crop.hpp>
#include <op_resize.hpp>
#include <string>

#include "common/utils.hpp"
#include "core/tensor_shape.hpp"

/**
 * @file main.cpp
 * @brief Crop and Resize sample app.
 *
 * The Crop and Resize is a simple pipeline which demonstrates usage of the
 * rocCV Tensor along with the Custom Crop and Resize operators.
 *
 * Input Batch Tensor -> Crop -> Resize -> WriteImage
 */

using namespace roccv;

struct Config {
    std::string inputPath;
    std::string outputPath = "output";
    eDeviceType device = eDeviceType::GPU;
    Size2D resizeShape = {320, 480};
    Box_t cropRect = {50, 150, 400, 300};
    eInterpolationType interpolation = eInterpolationType::INTERP_TYPE_LINEAR;
    int deviceId = 0;
};

void PrintUsage(const char* programName) {
    // clang-format off
    std::cout << "rocCV Crop and Resize Sample Application\n";
    std::cout << "----------------------------------------\n";
    std::cout << "This sample demonstrates how to set up a simple image processing pipeline using rocCV.\n";
    std::cout << "It shows reading a batch of images, cropping them to a specified rectangle,\n";
    std::cout << "resizing the cropped images to a target shape, and writing the outputs to image files.\n";
    std::cout << "You may select the device (CPU or GPU), interpolation type, input size, and more.\n";
    std::cout << '\n';
    std::cout << "Usage: " << programName << " -i <input_image_or_directory> [options]\n";
    std::cout << "  -i, --input <input_image_or_directory>   Input image or directory (required)\n";
    std::cout << "Options:\n";
    std::cout << "  -o, --output <output_image_or_directory> Output image or directory (optional, default: output)\n";
    std::cout << "  -r, --resize <width,height>              Resize shape as width,height (optional, default: 320,480)\n";
    std::cout << "  -c, --crop <x,y,w,h>                     Crop rectangle as x,y,w,h (optional, default: 50,150,400,300)\n";
    std::cout << "  -I, --interpolation <interpolation>      Interpolation type: 0=NEAREST, 1=LINEAR, 2=CUBIC (optional, default: LINEAR)\n";
    std::cout << "  -C, --cpu                                Use CPU for execution (optional, default: GPU)\n";
    std::cout << "  -d, --device <device_id>                 Device ID to use for execution (optional, default: 0)\n";
    std::cout << "  -h, --help                               Show this help message\n";
    std::cout << std::endl;
    // clang-format on
}

void ParseCropRectangle(const std::string& cropStr, Box_t& cropRect) {
    std::istringstream iss(cropStr);
    std::string token;
    getline(iss, token, ',');
    cropRect.x = std::stoi(token);
    getline(iss, token, ',');
    cropRect.y = std::stoi(token);
    getline(iss, token, ',');
    cropRect.width = std::stoi(token);
    getline(iss, token, ',');
    cropRect.height = std::stoi(token);
}

void ParseResizeShape(const std::string& resizeStr, Size2D& resizeShape) {
    std::istringstream iss(resizeStr);
    std::string token;
    getline(iss, token, ',');
    resizeShape.w = std::stoi(token);
    getline(iss, token, ',');
    resizeShape.h = std::stoi(token);
}

int main(int argc, char** argv) {
    Config config;
    static struct option longOptions[] = {{"input", required_argument, nullptr, 'i'},
                                          {"output", required_argument, nullptr, 'o'},
                                          {"resize", required_argument, nullptr, 'r'},
                                          {"crop", required_argument, nullptr, 'c'},
                                          {"interpolation", required_argument, nullptr, 'I'},
                                          {"cpu", no_argument, nullptr, 'C'},
                                          {"device", required_argument, nullptr, 'd'},
                                          {"help", no_argument, nullptr, 'h'},
                                          {nullptr, 0, nullptr, 0}};

    // Parse command line arguments
    int opt;
    while ((opt = getopt_long(argc, argv, "i:o:r:c:I:d:h:C", longOptions, nullptr)) != -1) {
        switch (opt) {
            case 'i':
                config.inputPath = optarg;
                break;
            case 'o':
                config.outputPath = optarg;
                break;
            case 'r':
                ParseResizeShape(optarg, config.resizeShape);
                break;
            case 'c':
                ParseCropRectangle(optarg, config.cropRect);
                break;
            case 'I':
                config.interpolation = static_cast<eInterpolationType>(std::stoi(optarg));
                break;
            case 'C':
                config.device = eDeviceType::CPU;
                break;
            case 'd':
                config.deviceId = std::stoi(optarg);
                break;
            case 'h':
                PrintUsage(argv[0]);
                return EXIT_SUCCESS;
            default:
                PrintUsage(argv[0]);
                return EXIT_FAILURE;
        }
    }

    if (config.inputPath.empty()) {
        std::cerr << "Error: Input path is required.\n\n";
        PrintUsage(argv[0]);
        return EXIT_FAILURE;
    }

    if (config.device == eDeviceType::GPU) {
        CHECK_HIP_ERROR(hipSetDevice(config.deviceId));
    }

    hipStream_t stream;
    CHECK_HIP_ERROR(hipStreamCreate(&stream));

    // Load batch of input images
    Tensor input = LoadImages(stream, config.inputPath, config.device);

    // Determine the batch size and channels from the input tensor
    int64_t batchSize = input.shape(input.layout().batch_index());
    int64_t channels = input.shape(input.layout().channels_index());

    // Create tensor for the cropped image
    Tensor cropTensor =
        Tensor(TensorShape(input.layout(), {batchSize, config.cropRect.height, config.cropRect.width, channels}),
               input.dtype(), config.device);

    // Create tensor for the resized image
    Tensor resizedTensor =
        Tensor(TensorShape(input.layout(), {batchSize, config.resizeShape.w, config.resizeShape.h, channels}),
               input.dtype(), config.device);

    // Create crop and resize operators
    CustomCrop cropOp;
    Resize resizeOp;

    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

    // Run the crop operation, writing results to the crop tensor
    cropOp(stream, input, cropTensor, config.cropRect, config.device);

    // Run the resize operation, writing results to the resized tensor
    resizeOp(stream, cropTensor, resizedTensor, config.interpolation, config.device);
    CHECK_HIP_ERROR(hipStreamSynchronize(stream));

    std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();

    // Report the duration of the crop and resize operation
    long executionTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Processed " << batchSize << " images in " << executionTime << "ms" << std::endl;

    // Write the cropped and resized images to disk
    WriteImages(stream, resizedTensor, config.outputPath);

    // Destroy the stream
    CHECK_HIP_ERROR(hipStreamDestroy(stream));

    return EXIT_SUCCESS;
}