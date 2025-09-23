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

#include <core/image_format.hpp>
#include <core/tensor.hpp>
#include <fstream>
#include <iostream>
#include <op_custom_crop.hpp>
#include <op_resize.hpp>
#include <string>

#include "common/utils.hpp"

/**
 * @brief Crop and Resize sample app.
 *
 * The Crop and Resize is a simple pipeline which demonstrates usage of
 * rocCV Tensor along with a few operators.
 *
 * Input Batch Tensor -> Crop -> Resize -> WriteImage
 */

/**
 * @brief Utility to show usage of sample app
 *
 **/
void showUsage() {
    std::cout << "usage: ./roccv_cropandresize_app -i <image file path or image directory> -b <batch size>"
              << std::endl;
}

/**
 * @brief Utility to parse the command line arguments
 *
 **/
int ParseArgs(int argc, char *argv[], std::string &imagePath, uint32_t &batchSize) {
    static struct option long_options[] = {{"help", no_argument, 0, 'h'},
                                           {"imagePath", required_argument, 0, 'i'},
                                           {"batch", required_argument, 0, 'b'},
                                           {0, 0, 0, 0}};

    int long_index = 0;
    int opt = 0;
    while ((opt = getopt_long(argc, argv, "hi:b:", long_options, &long_index)) != -1) {
        switch (opt) {
            case 'h':
                showUsage();
                return -1;
                break;
            case 'i':
                imagePath = optarg;
                break;
            case 'b':
                batchSize = std::stoi(optarg);
                break;
            case ':':
                showUsage();
                return -1;
            default:
                break;
        }
    }
    std::ifstream imageFile(imagePath);
    if (!imageFile.good()) {
        showUsage();
        std::cerr << "Image path '" + imagePath + "' does not exist" << std::endl;
        return -1;
    }
    return 0;
}

int main(int argc, char *argv[]) {
    // Default parameters
    // TODO: Default parameter for images cannot be added for now. Must specify a sample asset directory in the final
    // build which is relative to this executable.
    std::string imagePath = "none.jpg";
    uint32_t batchSize = 1;

    // Parse the command line paramaters to override the default parameters
    int retval = ParseArgs(argc, argv, imagePath, batchSize);
    if (retval != 0) {
        return retval;
    }

    // Note : The maximum input image dimensions needs to be updated in case
    // of testing with different test images

    int maxImageWidth = 720;
    int maxImageHeight = 480;
    int maxChannels = 3;

    // tag: Create the HIP stream
    hipStream_t stream;
    CHECK_HIP_ERROR(hipStreamCreate(&stream));

    // tag: Allocate input tensor
    // Allocating memory for RGBI input image batch of uint8_t data type.

    roccv::TensorDataStrided::Buffer inBuf;
    inBuf.strides[3] = sizeof(uint8_t);
    inBuf.strides[2] = maxChannels * inBuf.strides[3];
    inBuf.strides[1] = maxImageWidth * inBuf.strides[2];
    inBuf.strides[0] = maxImageHeight * inBuf.strides[1];
    CHECK_HIP_ERROR(hipMallocAsync(&inBuf.basePtr, batchSize * inBuf.strides[0], stream));

    // tag: Tensor Requirements
    // Calculate the requirements for the RGBI uint8_t Tensor which include
    // pitch bytes, alignment, shape  and tensor layout
    roccv::Tensor::Requirements inReqs =
        roccv::Tensor::CalcRequirements(batchSize, {maxImageWidth, maxImageHeight}, roccv::FMT_RGB8);

    // Create a tensor buffer to store the data pointer and pitch bytes for each plane
    roccv::TensorDataStrided inData(inReqs.shape, inReqs.dtype, inBuf);

    // Wrap tensor data in a rocCV tensor for use with the rocCV operators.
    roccv::Tensor inTensor = roccv::TensorWrapData(inData);

    // tag: Image Loading
    uint8_t *gpuInput = reinterpret_cast<uint8_t *>(inBuf.basePtr);
    // The total images is set to the same value as batch size for testing
    uint32_t totalImages = batchSize;

    // OpenCV is used to load the images, which gets copied into device memory.
    DecodeRGBIImage(imagePath, totalImages, gpuInput);

    // tag: The input buffer is now ready to be used by the operators

    // Set parameters for Crop and Resize
    // ROI dimensions to crop in the input image
    int cropX = 50;
    int cropY = 150;
    int cropWidth = 400;
    int cropHeight = 300;

    // Set the resize dimensions
    int resizeWidth = 320;
    int resizeHeight = 240;

    //  Create the crop rect for the cropping operator
    roccv::Box_t crpRect = {cropX, cropY, cropWidth, cropHeight};

    // tag: Allocate Tensors for Crop and Resize
    // Create a rocCV Tensor based on the crop window size.
    roccv::Tensor cropTensor(batchSize, {cropWidth, cropHeight}, roccv::FMT_RGB8);
    // Create a rocCV Tensor based on resize dimensions
    roccv::Tensor resizedTensor(batchSize, {resizeWidth, resizeHeight}, roccv::FMT_RGB8);

#ifdef PROFILE_SAMPLE
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);
    hipEventRecord(start);
#endif
    // tag: Initialize operators for Crop and Resize
    roccv::CustomCrop cropOp;
    roccv::Resize resizeOp;

    // tag: Executes the CustomCrop operation on the given HIP stream
    cropOp(stream, inTensor, cropTensor, crpRect);

    // Resize operator can now be enqueued into the same stream
    resizeOp(stream, cropTensor, resizedTensor, INTERP_TYPE_LINEAR);

    // tag: Profile section
#ifdef PROFILE_SAMPLE
    hipEventRecord(stop);
    hipEventSynchronize(stop);
    float operatorms = 0;
    hipEventElapsedTime(&operatorms, start, stop);
    std::cout << "Time for Crop and Resize : " << operatorms << " ms" << std::endl;
#endif

    // tag: Copy the buffer to CPU and write resized image into .bmp files
    WriteRGBITensor(resizedTensor, stream);

    // tag: Clean up
    CHECK_HIP_ERROR(hipStreamDestroy(stream));

    // tag: End of Sample
}