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
#include <fstream>
#include <iostream>
#include <op_bnd_box.hpp>
#include <opencv2/opencv.hpp>

#include "common/utils.hpp"

using namespace roccv;

/**
 * @file bnd_box.cpp
 * @brief Example application for drawing bounding boxes on images using ROC CV.
 *
 * This file demonstrates how to parse a bounding box description file and apply bounding boxes to input images.
 * The bounding box file is expected to have the following format:
 * - The first line contains the number of images.
 * - Each image section starts with a line containing the number of boxes for that image.
 * - Each box is described over 13 subsequent lines as follows:
 *   - X coordinate of top-left corner
 *   - Y coordinate of top-left corner
 *   - Width
 *   - Height
 *   - Thickness of box boundary
 *   - B component of box border color
 *   - G component of box border color
 *   - R component of box border color
 *   - Alpha component of box border color
 *   - B component of box fill color
 *   - G component of box fill color
 *   - R component of box fill color
 *   - Alpha component of box fill color
 *
 * Example of bounding box list file content:
 * @code
 * 1
 * 2
 * 50
 * 50
 * 100
 * 50
 * 5
 * 0
 * 0
 * 255
 * 200
 * 0
 * 255
 * 0
 * 100
 * 250
 * 250
 * 50
 * 100
 * 10
 * 255
 * 0
 * 0
 * 200
 * 0
 * 0
 * 0
 * 0
 * @endcode
 */

struct Config {
    std::string inputPath;
    std::string outputPath = "output";
    eDeviceType device = eDeviceType::GPU;
    int deviceId = 0;
    std::string boundingBoxFilePath = "";
};

void PrintUsage(const char* programName) {
    // clang-format off
    std::cout << "Usage: " << programName << " -i <input_image> [-o <output_image>] [-b <bounding_box_file>] [-d <device_id>] [-C]" << std::endl;
    std::cout << "  -i, --input <input_image>           Input image or directory containing images (required)" << std::endl;
    std::cout << "  -o, --output <output_image>         Output image or directory to save the results (optional, default: output)" << std::endl;
    std::cout << "  -b, --box_file <bounding_box_file>  Bounding box list file (optional, default: use the set value in the app)" << std::endl;
    std::cout << "  -d, --device <device_id>            Device ID to use for execution (optional, default: 0)" << std::endl;
    std::cout << "  -C, --cpu                           Use CPU for execution (optional, default: GPU)" << std::endl;
    std::cout << "  -h, --help                          Show this help message" << std::endl;
    // clang-format on
}

/**
 * @brief Parse bounding box file and setup bounding box vector.
 *
 * @param[in] boundingBoxFilePath Path to bounding box list file.
 * @param[out] bbox_vector Vector of bounding box vectors to be filled.
 * @return void
 * @throws std::runtime_error if failed to open bounding box file.
 */
void ParseBoundingBoxFile(const std::string& boundingBoxFilePath, std::vector<std::vector<BndBox_t>>& bbox_vector) {
    std::ifstream file(boundingBoxFilePath);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open bounding box file " + boundingBoxFilePath);
    }

    int numImages;
    file >> numImages;
    bbox_vector.resize(numImages);
    for (int i = 0; i < numImages; i++) {
        int numBoxes;
        file >> numBoxes;

        bbox_vector[i].resize(numBoxes);
        for (int j = 0; j < numBoxes; j++) {
            // Parse each box from 13 lines, in order:
            //   1. x         (top-left corner)
            //   2. y         (top-left corner)
            //   3. width
            //   4. height
            //   5. thickness (box boundary)
            //   6. border B
            //   7. border G
            //   8. border R
            //   9. border A (alpha)
            //  10. fill B
            //  11. fill G
            //  12. fill R
            //  13. fill A (alpha)
            // (13 values total per box)

            // Read box dimensions
            file >> bbox_vector[i][j].box.x >> bbox_vector[i][j].box.y >> bbox_vector[i][j].box.width >>
                bbox_vector[i][j].box.height;
            file >> bbox_vector[i][j].thickness;

            // Read colors into temp ints
            int r, g, b, a;
            file >> b >> g >> r >> a;
            bbox_vector[i][j].borderColor = {static_cast<uint8_t>(r), static_cast<uint8_t>(g), static_cast<uint8_t>(b),
                                             static_cast<uint8_t>(a)};

            file >> b >> g >> r >> a;
            bbox_vector[i][j].fillColor = {static_cast<uint8_t>(r), static_cast<uint8_t>(g), static_cast<uint8_t>(b),
                                           static_cast<uint8_t>(a)};
        }
    }
}

/**
 * @brief Setup bounding box vector from file or default values.
 *
 * @param[in] batchSize Number of images in the batch.
 * @param[in] width Width of each image in the batch.
 * @param[in] height Height of each image in the batch.
 * @param[in] boundingBoxFilePath Path to bounding box list file.
 * @return Vector of bounding box vectors.
 */
std::vector<std::vector<BndBox_t>> SetupBoundingBoxVector(int64_t batchSize, int64_t width, int64_t height,
                                                          const std::string& boundingBoxFilePath) {
    std::vector<std::vector<BndBox_t>> bbox_vector;

    if (boundingBoxFilePath.empty()) {
        for (int64_t b = 0; b < batchSize; b++) {
            bbox_vector.push_back({
                {{width / 4, height / 4, width / 2, height / 2}, 5, {0, 0, 255, 200}, {0, 255, 0, 100}},
                {{width / 3, height / 3, width / 3 * 2, height / 4}, -1, {90, 16, 181, 50}, {0, 0, 0, 0}},
                {{-50, (height * 2) / 3, width + 50, height / 3 + 50}, 0, {0, 0, 0, 0}, {111, 159, 232, 150}},
            });
        }
    } else {
        ParseBoundingBoxFile(boundingBoxFilePath, bbox_vector);
    }

    return bbox_vector;
}

/**
 * @brief Main function to run the bounding box operation example.
 *
 * @param[in] argc Number of command line arguments.
 * @param[in] argv Command line arguments.
 * @return int Exit status.
 */
int main(int argc, char** argv) {
    Config config;
    static struct option longOptions[] = {{"input", required_argument, nullptr, 'i'},
                                          {"output", required_argument, nullptr, 'o'},
                                          {"box_file", required_argument, nullptr, 'b'},
                                          {"device", required_argument, nullptr, 'd'},
                                          {"cpu", no_argument, nullptr, 'C'},
                                          {"help", no_argument, nullptr, 'h'},
                                          {nullptr, 0, nullptr, 0}};
    int opt;
    while ((opt = getopt_long(argc, argv, "i:o:b:d:h:C", longOptions, nullptr)) != -1) {
        switch (opt) {
            case 'i':
                config.inputPath = optarg;
                break;
            case 'o':
                config.outputPath = optarg;
                break;
            case 'b':
                config.boundingBoxFilePath = optarg;
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

    if (config.device == eDeviceType::GPU) {
        CHECK_HIP_ERROR(hipSetDevice(config.deviceId));
    }

    hipStream_t stream;
    CHECK_HIP_ERROR(hipStreamCreate(&stream));

    // Load images
    Tensor input = LoadImages(stream, config.inputPath.c_str(), config.device);
    int64_t batchSize = input.shape()[input.shape().layout().batch_index()];
    int64_t height = input.shape()[input.shape().layout().height_index()];
    int64_t width = input.shape()[input.shape().layout().width_index()];

    // Setup bounding box vector
    std::vector<std::vector<BndBox_t>> bbox_vector =
        SetupBoundingBoxVector(batchSize, width, height, config.boundingBoxFilePath);

    BndBoxes bboxes(bbox_vector);

    // Create output tensor
    Tensor output(input.shape(), input.dtype(), config.device);

    // Create BndBox operator
    BndBox op;
    op(stream, input, output, bboxes, config.device);

    // Write output images to disk
    WriteImages(stream, output, config.outputPath);

    CHECK_HIP_ERROR(hipStreamDestroy(stream));
    return EXIT_SUCCESS;
}