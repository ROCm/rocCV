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
#include <op_warp_perspective.hpp>
#include <opencv2/opencv.hpp>

#include "common/utils.hpp"

using namespace roccv;

struct Config {
    std::string inputPath;
    std::string outputPath = "output";
    eInterpolationType interpolation = eInterpolationType::INTERP_TYPE_LINEAR;
    eBorderType borderMode = eBorderType::BORDER_TYPE_CONSTANT;
    int deviceId = 0;
};

void PrintUsage(const char* programName) {
    // clang-format off
    std::cerr << "Usage: " << programName << " -i <input_image> [-o <output_image>] [-p <interpolation>] [-b <border>] [-d <device_id>]" << std::endl;
    std::cerr << "  -i, --input <input_image>           Input image or directory containing images (required)" << std::endl;
    std::cerr << "  -o, --output <output_image>         Output image or directory to save the results (optional, default: output)" << std::endl;
    std::cerr << "  -I, --interpolation <interpolation> Interpolation type to use for output images [0: NEAREST, 1: LINEAR, 2: CUBIC] (optional, default: 1 (LINEAR))" << std::endl;
    std::cerr << "  -b, --border <border>               Border type for output images [0: CONSTANT, 1: REPLICATE, 2: REFLECT, 3: REFLECT101, 4: WRAP] (optional, default: 0 (CONSTANT))" << std::endl;
    std::cerr << "  -d, --device <device_id>            Device ID to use for execution (optional, default: 0)" << std::endl;
    // clang-format on
}

/**
 * @brief Warp perspective operation example.
 */
int main(int argc, char** argv) {
    Config config;

    static struct option longOptions[] = {{"input", required_argument, nullptr, 'i'},
                                          {"output", required_argument, nullptr, 'o'},
                                          {"interpolation", required_argument, nullptr, 'I'},
                                          {"border", required_argument, nullptr, 'b'},
                                          {"device", required_argument, nullptr, 'd'},
                                          {"help", no_argument, nullptr, 'h'},
                                          {nullptr, 0, nullptr, 0}};

    // Parse command line arguments
    int opt;
    while ((opt = getopt_long(argc, argv, "i:o:I:b:d:h", longOptions, nullptr)) != -1) {
        switch (opt) {
            case 'i':
                config.inputPath = optarg;
                break;
            case 'o':
                config.outputPath = optarg;
                break;
            case 'I':
                config.interpolation = static_cast<eInterpolationType>(std::stoi(optarg));
                break;
            case 'b':
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

    hipStream_t stream;
    CHECK_HIP_ERROR(hipStreamCreate(&stream));

    // Create input/output tensors for the image.
    Tensor input = LoadImages(stream, config.inputPath, eDeviceType::GPU);
    Tensor output(input.shape(), input.dtype(), eDeviceType::GPU);

    PerspectiveTransform transform_matrix = {1, 0, 0, 0, 1, 0, -0.001, 0, 1};

    roccv::WarpPerspective op;
    op(stream, input, output, transform_matrix, true, config.interpolation, config.borderMode, make_float4(0, 0, 0, 0));

    // Move image data back to device
    WriteImages(stream, output, config.outputPath);
    CHECK_HIP_ERROR(hipStreamDestroy(stream));

    return EXIT_SUCCESS;
}