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
#include <op_custom_crop.hpp>
#include <opencv2/opencv.hpp>

using namespace roccv;

/**
 * @brief Custom crop operation example.
 */

void ShowHelpAndExit(const char *option = NULL) {
    std::cout << "Options: " << option << std::endl
    << "-i Input File Path - required" << std::endl
    << "-o Output File Path - optional; default: output.bmp" << std::endl
    << "-cpu Select CPU instead of GPU to perform operation - optional; default choice is GPU path" << std::endl
    << "-d GPU device ID (0 for the first device, 1 for the second, etc.) - optional; default: 0" << std::endl
    << "-crop Crop rectangle (top_left_corner_x, top_left_corner_y, width, height)- optional; default: use the set value in the app" << std::endl;
    exit(0);
}

int main(int argc, char** argv) {
    std::string input_file_path;
    std::string output_file_path = "output.bmp";
    bool gpuPath = true; // use GPU by default
    eDeviceType device = eDeviceType::GPU;
    int deviceId = 0;
    Box_t cropRect = {0, 0, 1, 1};
    bool cropSet = false;

    if(argc < 3) {
        ShowHelpAndExit("-h");
    }
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-h")) {
            ShowHelpAndExit("-h");
        }
        if (!strcmp(argv[i], "-i")) {
            if (++i == argc) {
                ShowHelpAndExit("-i");
            }
            input_file_path = argv[i];
            continue;
        }
        if (!strcmp(argv[i], "-o")) {
            if (++i == argc) {
                ShowHelpAndExit("-o");
            }
            output_file_path = argv[i];
            continue;
        }
        if (!strcmp(argv[i], "-crop")) {
            i++;
            if (i + 4 > argc) {
                ShowHelpAndExit("-crop");
            }
            cropRect.x = atoi(argv[i++]);
            cropRect.y = atoi(argv[i++]);
            cropRect.width = atoi(argv[i++]);
            cropRect.height = atoi(argv[i]);
            cropSet = true;
            continue;
        }
        if (!strcmp(argv[i], "-cpu")) {
            gpuPath = false;
            continue;
        }
    }

    if (gpuPath) {
        device = eDeviceType::GPU;
        HIP_VALIDATE_NO_ERRORS(hipSetDevice(deviceId));
    } else {
        device = eDeviceType::CPU;
    }

    cv::Mat imageData = cv::imread(input_file_path);
    if (imageData.empty()) {
        std::cerr << "Failed to read the input image file" << std::endl;
        exit(1);
    }
    if (!cropSet) {
        // Set a safe crop area if no user input
        cropRect = {(imageData.cols / 4), (imageData.rows / 4), (imageData.cols / 2), (imageData.rows / 2)};
    }

    // Create input/output tensors for the image.
    TensorShape inputShape(TensorLayout(eTensorLayout::TENSOR_LAYOUT_NHWC), {1, imageData.rows, imageData.cols, imageData.channels()});
    DataType dtype(eDataType::DATA_TYPE_U8);
    Tensor input(inputShape, dtype, device);

    TensorShape outShape(TensorLayout(eTensorLayout::TENSOR_LAYOUT_NHWC), {1, cropRect.height, cropRect.width, imageData.channels()});
    Tensor output(outShape, dtype, device);

    hipStream_t stream = nullptr;
    if (gpuPath) {
        HIP_VALIDATE_NO_ERRORS(hipStreamCreate(&stream));
    }

    // Move image data to input tensor
    size_t imageSizeInByte = input.shape().size() * input.dtype().size();
    auto input_data = input.exportData<TensorDataStrided>();
    if (gpuPath) {
        HIP_VALIDATE_NO_ERRORS(hipMemcpyAsync(input_data.basePtr(), imageData.data, imageSizeInByte, hipMemcpyHostToDevice, stream));
    } else {
        memcpy(input_data.basePtr(), imageData.data, imageSizeInByte);
    }

    CustomCrop op;
    op(nullptr, input, output, cropRect, device);

    // Move image data back to host
    size_t outputSize = output.shape().size() * output.dtype().size();
    auto outData = output.exportData<TensorDataStrided>();
    std::vector<uint8_t> h_output(outputSize);
    if (gpuPath) {
        HIP_VALIDATE_NO_ERRORS(hipMemcpyAsync(h_output.data(), outData.basePtr(), outputSize, hipMemcpyDeviceToHost, stream));
        HIP_VALIDATE_NO_ERRORS(hipStreamSynchronize(stream));
    } else {
        memcpy(h_output.data(), outData.basePtr(), outputSize);
    }

    // Write output image to disk
    cv::Mat output_imageData(cropRect.height, cropRect.width, imageData.type(), h_output.data());
    bool ret = cv::imwrite(output_file_path, output_imageData);
    if (!ret) {
        std::cerr << "Faild to save output image to the file" << std::endl;
    }

    std::cout << "Input image file: " << input_file_path << std::endl;
    std::cout << "Output image file: " << output_file_path << std::endl;
    if (gpuPath) {
        std::cout << "Operation on GPU device " << deviceId << std::endl;
    } else {
        std::cout << "Operation on CPU" << std::endl;
    }
    std::cout << "Input image size: width = " << imageData.cols << ", height = " << imageData.rows << std::endl;
    std::cout << "Cropping area: top-left corner = (" << cropRect.x << ", " << cropRect.y << "), width = " << cropRect.width << ", height = " << cropRect.height << std::endl;

    return EXIT_SUCCESS;
}