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
#include <op_copy_make_border.hpp>
#include <opencv2/opencv.hpp>

#include "common/utils.hpp"

using namespace roccv;

struct Config {
    std::string inputPath;
    std::string outputPath = "output";
    int32_t top = 10;
    int32_t left = 10;
    float r = 0.0f, g = 0.0f, b = 0.0f, a = 255.0f;
    eBorderType borderMode = eBorderType::BORDER_TYPE_CONSTANT;
    int deviceId = 0;
};

void PrintUsage(const char* programName) {
    std::cout << "Usage: " << programName << " -i <input_image_or_directory> [options]\n\n"
              << "Options:\n"
              << "  -i, --input <path>      Input image or directory (required)\n"
              << "  -o, --output <path>     Output file or directory (default: output/image_<index>.bmp)\n"
              << "  -t, --top <pixels>      Top/bottom border size (default: 10)\n"
              << "  -l, --left <pixels>     Left/right border size (default: 10)\n"
              << "  -c, --color <r,g,b,a>   Border color as r,g,b,a (default: 0,0,0,255)\n"
              << "  -m, --mode <mode>       Border mode: 0=constant, 1=replicate, 2=reflect (default: constant)\n"
              << "  -d, --device <id>       GPU device ID (default: 0)\n"
              << "  -h, --help              Show this help message\n\n"
              << "Example:\n"
              << "  " << programName << " -i image.jpg -o bordered.png -t 20 -l 15 -c 255,0,0,255\n";
}

bool ParseColor(const std::string& colorStr, float& r, float& g, float& b, float& a) {
    return sscanf(colorStr.c_str(), "%f,%f,%f,%f", &r, &g, &b, &a) == 4;
}

/**
 * @brief Copy make border operation example.
 *
 * This sample demonstrates the usage of the CopyMakeBorder operator. It accepts either a single image file or a
 * directory of images as the input path, and either a single file or a directory as the output path. If a directory is
 * provided for the input, all supported images within the directory will be processed as a batch. Similarly, if the
 * output path is a directory, all resulting images will be written to that directory. For each image, a border is
 * created based on the specified border mode and border value.
 *
 * If <image_or_directory> is a directory, a batch operation will occur on every image in the directory.
 * If <output_file_or_directory> is a directory, the output images will be written into that directory.
 */
int main(int argc, char** argv) {
    Config config;

    static struct option longOptions[] = {{"input", required_argument, nullptr, 'i'},
                                          {"output", required_argument, nullptr, 'o'},
                                          {"top", required_argument, nullptr, 't'},
                                          {"left", required_argument, nullptr, 'l'},
                                          {"color", required_argument, nullptr, 'c'},
                                          {"mode", required_argument, nullptr, 'm'},
                                          {"device", required_argument, nullptr, 'd'},
                                          {"help", no_argument, nullptr, 'h'},
                                          {nullptr, 0, nullptr, 0}};

    int opt;
    while ((opt = getopt_long(argc, argv, "i:o:t:l:c:m:d:h", longOptions, nullptr)) != -1) {
        switch (opt) {
            case 'i':
                config.inputPath = optarg;
                break;
            case 'o':
                config.outputPath = optarg;
                break;
            case 't':
                config.top = std::stoi(optarg);
                break;
            case 'l':
                config.left = std::stoi(optarg);
                break;
            case 'c':
                if (!ParseColor(optarg, config.r, config.g, config.b, config.a)) {
                    std::cerr << "Invalid color format. Use: r,g,b,a (e.g., 255,0,0,255)\n";
                    return EXIT_FAILURE;
                }
                break;
            case 'm':
                config.borderMode = static_cast<eBorderType>(std::stoi(optarg));
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

    CHECK_HIP_ERROR(hipSetDevice(config.deviceId));

    // Create stream
    hipStream_t stream;
    CHECK_HIP_ERROR(hipStreamCreate(&stream));

    // Load input image
    Tensor input = LoadImages(stream, config.inputPath.c_str());

    // Create output tensor
    int64_t outputHeight = input.shape(input.layout().height_index()) + config.top * 2;
    int64_t outputWidth = input.shape(input.layout().width_index()) + config.left * 2;
    TensorShape outputShape(input.layout(), {input.shape(input.layout().batch_index()), outputHeight, outputWidth,
                                             input.shape(input.layout().channels_index())});
    Tensor output(outputShape, input.dtype());

    // Create CopyMakeBorder operator
    CopyMakeBorder op;
    op(stream, input, output, config.top, config.left, config.borderMode, {config.b, config.g, config.r, config.a});

    WriteImages(stream, output, config.outputPath);

    CHECK_HIP_ERROR(hipStreamDestroy(stream));

    return EXIT_SUCCESS;
}