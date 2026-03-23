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

#include <core/tensor.hpp>
#include <op_composite.hpp>
#include <opencv2/opencv.hpp>

#include "common/utils.hpp"

/** @file composite.cpp
 * @brief Sample application for the Composite operation.
 *
 * This sample application demonstrates the Composite operation, which blends the foreground image into the background
 * image using a grayscale alpha mask. It loads the background, foreground, and mask images, runs the Composite
 * operation, and writes the output images to disk.
 */

using namespace roccv;

struct Config {
    std::string backgroundPath;
    std::string foregroundPath;
    std::string maskPath;
    std::string outputPath = "output";
    eDeviceType device = eDeviceType::GPU;
    int deviceId = 0;
};

void PrintUsage(const char* programName) {
    // clang-format off
    std::cout << "Usage: " << programName << " -b <background_filename> -f <foreground_filename> -m <mask_filename> -o <output_filename> -d <device_id>" << std::endl;
    std::cout << "  -b, --background <background_filename> Background image filename (required)" << std::endl;
    std::cout << "  -f, --foreground <foreground_filename> Foreground image filename (required)" << std::endl;
    std::cout << "  -m, --mask <mask_filename> Mask image filename (required)" << std::endl;
    std::cout << "  -o, --output <output_filename> Output image filename (optional, default: output)" << std::endl;
    std::cout << "  -d, --device <device_id> Device ID to use for execution (optional, default: 0)" << std::endl;
    std::cout << "  -h, --help Show this help message" << std::endl;
    std::cout << std::endl;
    std::cout << "NOTE: If you use directories for the background, foreground, or mask inputs, each directory must contain the same number of images. Each image in the directories must be the same size." << std::endl;
    // clang-format on
}

/**
 * @brief Main function for the Composite operation.
 *
 * @param argc Number of command line arguments.
 * @param argv Command line arguments.
 * @return EXIT_SUCCESS if the operation completed successfully, EXIT_FAILURE otherwise.
 */
int main(int argc, char** argv) {
    Config config;
    static struct option longOptions[] = {{"background", required_argument, nullptr, 'b'},
                                          {"foreground", required_argument, nullptr, 'f'},
                                          {"mask", required_argument, nullptr, 'm'},
                                          {"output", required_argument, nullptr, 'o'},
                                          {"device", required_argument, nullptr, 'd'},
                                          {"help", no_argument, nullptr, 'h'},
                                          {nullptr, 0, nullptr, 0}};
    int opt;
    while ((opt = getopt_long(argc, argv, "b:f:m:o:d:h", longOptions, nullptr)) != -1) {
        switch (opt) {
            case 'b':
                config.backgroundPath = optarg;
                break;
            case 'f':
                config.foregroundPath = optarg;
                break;
            case 'm':
                config.maskPath = optarg;
                break;
            case 'o':
                config.outputPath = optarg;
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
    if (config.backgroundPath.empty() || config.foregroundPath.empty() || config.maskPath.empty()) {
        std::cerr << "Error: Background, foreground, and mask paths are required.\n\n";
        PrintUsage(argv[0]);
        return EXIT_FAILURE;
    }

    if (config.device == eDeviceType::GPU) {
        CHECK_HIP_ERROR(hipSetDevice(config.deviceId));
    }

    hipStream_t stream;
    CHECK_HIP_ERROR(hipStreamCreate(&stream));

    // Create required tensors for the Composite operation
    Tensor background = LoadImages(hipStreamPerThread, config.backgroundPath, config.device);
    Tensor foreground = LoadImages(hipStreamPerThread, config.foregroundPath, config.device);
    Tensor mask = LoadImages(hipStreamPerThread, config.maskPath, config.device, cv::IMREAD_GRAYSCALE);
    Tensor output = Tensor(background.shape(), background.dtype(), config.device);

    // Run the Composite operation
    Composite op;
    op(stream, foreground, background, mask, output, config.device);

    // Write output images to disk
    WriteImages(stream, output, config.outputPath);

    CHECK_HIP_ERROR(hipStreamDestroy(stream));

    return EXIT_SUCCESS;
}