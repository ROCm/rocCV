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

#include <core/tensor.hpp>
#include <iostream>
#include <op_copy_make_border.hpp>
#include <opencv2/opencv.hpp>

#include "common/utils.hpp"

using namespace roccv;

/**
 * @brief Copy make border operation example.
 *
 * This sample demonstrates the usage of the CopyMakeBorder operator. It loads an image from the specified input path,
 * creates a border around the image based on the specified border mode and border value, and writes the output image to
 * the specified output path.
 *
 * Usage:
 * ./copy_make_border <image_filename> <output_filename> <top> <left> <r> <g> <b> <a> <border_mode> <device_id>
 *
 */
int main(int argc, char** argv) {
    // Validate command line arguments
    if (argc != 11) {
        std::cerr << "Usage: " << argv[0]
                  << " <image_filename> <output_filename> <top> <left> <r> <g> <b> <a> <border_mode> <device_id>"
                  << std::endl;
        return EXIT_FAILURE;
    }

    CHECK_HIP_ERROR(hipSetDevice(std::stoi(argv[10])));

    // Parse command line arguments
    int32_t top = std::stoi(argv[3]);
    int32_t left = std::stoi(argv[4]);
    float r = std::stof(argv[5]);
    float g = std::stof(argv[6]);
    float b = std::stof(argv[7]);
    float a = std::stof(argv[8]);
    eBorderType border_mode = static_cast<eBorderType>(std::stoi(argv[9]));

    // Load input image
    Tensor input = LoadImages(argv[1]);

    // Create output tensor
    int64_t outputHeight = input.shape(input.layout().height_index()) + top * 2;
    int64_t outputWidth = input.shape(input.layout().width_index()) + left * 2;
    TensorShape outputShape(input.layout(), {input.shape(input.layout().batch_index()), outputHeight, outputWidth,
                                             input.shape(input.layout().channels_index())});
    Tensor output(outputShape, input.dtype());

    // Create stream
    hipStream_t stream;
    CHECK_HIP_ERROR(hipStreamCreate(&stream));

    // Create CopyMakeBorder operator
    CopyMakeBorder op;
    op(stream, input, output, top, left, border_mode, {b, g, r, a});

    CHECK_HIP_ERROR(hipStreamSynchronize(stream));

    WriteImages(output, argv[2]);

    CHECK_HIP_ERROR(hipStreamSynchronize(stream));

    return EXIT_SUCCESS;
}