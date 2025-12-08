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
#include <op_bilateral_filter.hpp>
#include <opencv2/opencv.hpp>

#include "common/utils.hpp"

using namespace roccv;

struct Config {
    std::string inputPath;
    std::string outputPath = "output";
    int deviceId = 0;
    int diameter = 2;
    float sigmaSpace = 2.0f;
    float sigmaColor = 10.0f;
    eBorderType borderMode = eBorderType::BORDER_TYPE_REPLICATE;
    float4 borderColor = {0.0f, 0.0f, 0.0f, 0.0f};
    eDeviceType device = eDeviceType::GPU;
};

/**
 * @brief Bilateral filter operation example.
 */

void PrintUsage(const char* programName) {
    // clang-format off
    std::cout << "Usage: " << programName << " -i <input_image> [-o <output_image>] [-d <device_id>] [-D <diameter>] [-s <sigma_space>] [-c <sigma_color>] [-b <border_mode>] [-B <border_color>]" << std::endl;
    std::cout << "  -i, --input <input_image>           Input image or directory containing images (required)" << std::endl;
    std::cout << "  -o, --output <output_image>         Output image or directory to save the results (optional, default: output)" << std::endl;
    std::cout << "  -d, --device <device_id>            Device ID to use for execution (optional, default: 0)" << std::endl;
    std::cout << "  -D, --diameter <diameter>           Diameter of the filtering area (optional, default: 2)" << std::endl;
    std::cout << "  -s, --sigma_space <sigma_space>     Spatial parameter sigma of the Gaussian function (optional, default: 2.0f)" << std::endl;
    std::cout << "  -c, --sigma_color <sigma_color>     Range parameter sigma of the Gaussian function (optional, default: 10.0f)" << std::endl;
    std::cout << "  -B, --border_mode <border_mode>     Border mode at image boundary when work pixels are outside of the image (optional, default: 1 (replicate))" << std::endl;
    std::cout << "  -B, --border_color <border_color>   Border color for constant color border mode (optional, default: 0,0,0,0)" << std::endl;
    std::cout << "  -C, --cpu                           Use CPU for execution (optional, default: GPU)" << std::endl;
    std::cout << "  -h, --help                          Show this help message" << std::endl;
    // clang-format on
}

bool ParseBorderColor(const std::string& borderColorStr, float4& borderColor) {
    return sscanf(borderColorStr.c_str(), "%f,%f,%f,%f", &borderColor.x, &borderColor.y, &borderColor.z,
                  &borderColor.w) == 4;
}

int main(int argc, char** argv) {
    Config config;

    static struct option longOptions[] = {{"input", required_argument, nullptr, 'i'},
                                          {"output", required_argument, nullptr, 'o'},
                                          {"device", required_argument, nullptr, 'd'},
                                          {"diameter", required_argument, nullptr, 'D'},
                                          {"sigma_space", required_argument, nullptr, 's'},
                                          {"sigma_color", required_argument, nullptr, 'c'},
                                          {"border_mode", required_argument, nullptr, 'b'},
                                          {"border_color", required_argument, nullptr, 'B'},
                                          {"cpu", no_argument, nullptr, 'C'},
                                          {"help", no_argument, nullptr, 'h'},
                                          {nullptr, 0, nullptr, 0}};

    int opt;
    while ((opt = getopt_long(argc, argv, "i:o:d:D:s:c:b:B:h:C", longOptions, nullptr)) != -1) {
        switch (opt) {
            case 'i':
                config.inputPath = optarg;
                break;
            case 'o':
                config.outputPath = optarg;
                break;
            case 'd':
                config.deviceId = std::stoi(optarg);
                break;
            case 'D':
                config.diameter = std::stoi(optarg);
                break;
            case 's':
                config.sigmaSpace = std::stof(optarg);
                break;
            case 'c':
                config.sigmaColor = std::stof(optarg);
                break;
            case 'b':
                config.borderMode = static_cast<eBorderType>(std::stoi(optarg));
                break;
            case 'B':
                if (!ParseBorderColor(optarg, config.borderColor)) {
                    std::cerr << "Invalid border color format. Use: r,g,b,a (e.g., 0,0,0,0)\n";
                    return EXIT_FAILURE;
                }
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

    if (config.device == eDeviceType::GPU) {
        CHECK_HIP_ERROR(hipSetDevice(config.deviceId));
    }

    // Create stream
    hipStream_t stream;
    CHECK_HIP_ERROR(hipStreamCreate(&stream));

    // Load input image
    Tensor input = LoadImages(stream, config.inputPath.c_str(), config.device);

    // Create output tensor
    Tensor output(input.shape(), input.dtype(), config.device);

    // Create BilateralFilter operator
    BilateralFilter op;
    op(stream, input, output, config.diameter, config.sigmaColor, config.sigmaSpace, config.borderMode,
       config.borderColor, config.device);

    WriteImages(stream, output, config.outputPath);

    CHECK_HIP_ERROR(hipStreamDestroy(stream));

    return EXIT_SUCCESS;
}