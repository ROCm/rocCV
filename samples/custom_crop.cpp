/**
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <core/hip_assert.h>
#include <getopt.h>

#include <core/tensor.hpp>
#include <iostream>
#include <op_custom_crop.hpp>
#include <opencv2/opencv.hpp>

#include "common/utils.hpp"

using namespace roccv;

struct Config {
    std::string inputPath;
    std::string outputPath = "output";
    Box_t cropRect = {0, 0, 1, 1};
    eDeviceType device = eDeviceType::GPU;
    int deviceId = 0;
};

void PrintUsage(const char* programName) {
    // clang-format off
    std::cout << "Usage: " << programName << " -i <input_image> [-o <output_image>] [-crop <x,y,w,h>] [-d <device_id>] [-c]" << std::endl;
    std::cout << "  -i, --input <input_image>           Input image or directory containing images (required)" << std::endl;
    std::cout << "  -o, --output <output_image>         Output image or directory to save the results (optional, default: output)" << std::endl;
    std::cout << "  -c, --crop <x,y,w,h>                Crop rectangle as comma separated values (optional, default: 0,0,1,1)" << std::endl;
    std::cout << "  -d, --device <device_id>            Device ID to use for execution when using GPU (optional, default: 0)" << std::endl;
    std::cout << "  -C, --cpu                           Use CPU for execution (optional, default: GPU)" << std::endl;
    std::cout << "  -h, --help                          Show this help message" << std::endl;
    // clang-format on
}

bool ParseCropRectangle(const std::string& cropStr, Box_t& cropRect) {
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
    return true;
}

/**
 * @brief Custom crop operation example.
 */
int main(int argc, char** argv) {
    Config config;
    static struct option longOptions[] = {{"input", required_argument, nullptr, 'i'},
                                          {"output", required_argument, nullptr, 'o'},
                                          {"crop", required_argument, nullptr, 'c'},
                                          {"device", required_argument, nullptr, 'd'},
                                          {"cpu", no_argument, nullptr, 'C'},
                                          {"help", no_argument, nullptr, 'h'},
                                          {nullptr, 0, nullptr, 0}};
    int opt;
    while ((opt = getopt_long(argc, argv, "i:o:c:d:h:C", longOptions, nullptr)) != -1) {
        switch (opt) {
            case 'i':
                config.inputPath = optarg;
                break;
            case 'o':
                config.outputPath = optarg;
                break;
            case 'c':
                if (!ParseCropRectangle(optarg, config.cropRect)) {
                    std::cerr << "Invalid crop rectangle format. Use: x,y,w,h (e.g., 0,0,1,1)\n";
                    return EXIT_FAILURE;
                }
                break;
            case 'd':
                config.deviceId = std::stoi(optarg);
                break;
            case 'C':
                config.device = eDeviceType::CPU;
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

    CHECK_HIP_ERROR(hipSetDevice(config.deviceId));

    hipStream_t stream;
    CHECK_HIP_ERROR(hipStreamCreate(&stream));

    // Load input images
    Tensor input = LoadImages(stream, config.inputPath, config.device);

    int64_t batchSize = input.shape(input.layout().batch_index());
    int64_t channels = input.shape(input.layout().channels_index());

    // Create output tensor
    Tensor output(TensorShape(input.layout(), {batchSize, config.cropRect.height, config.cropRect.width, channels}),
                  input.dtype(), config.device);

    // Run custom crop operation
    CustomCrop op;
    op(stream, input, output, config.cropRect, config.device);

    // Write output image to disk
    WriteImages(stream, output, config.outputPath);

    CHECK_HIP_ERROR(hipStreamDestroy(stream));

    return EXIT_SUCCESS;
}