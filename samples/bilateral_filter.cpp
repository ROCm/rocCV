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
#include <fstream>
#include <op_bilateral_filter.hpp>
#include <opencv2/opencv.hpp>

using namespace roccv;

/**
 * @brief Bilateral filter operation example.
 */

 void ShowHelpAndExit(const char *option = NULL) {
    std::cout << "Options: " << option << std::endl
    << "-i Input File Path - required" << std::endl
    << "-o Output File Path - optional; default: output.bmp" << std::endl
    << "-cpu Select CPU instead of GPU to perform operation - optional; default choice is GPU path" << std::endl
    << "-d GPU device ID (0 for the first device, 1 for the second, etc.) - optional; default: 0" << std::endl
    << "-diameter Diameter of the filtering area - optional; default: 2" << std::endl
    << "-sigma_space Spatial parameter sigma of the Gaussian function - optional; default: 2.0f" << std::endl
    << "-sigma_color Range parameter sigma of the Gaussian function - optional; default: 10.0f" << std::endl
    << "-border_mode Border mode at image boundary when work pixels are outside of the image (0: constant color; 1: replicate; 2: reflect; 3: wrap) - optional; default: 1 (replicate)" << std::endl
    << "-border_color Border color for constant color border mode - optional; default: (0, 0, 0, 0)" << std::endl;
    exit(0);
}

int main(int argc, char** argv) {
    std::string input_file_path;
    std::string output_file_path = "output.bmp";
    bool gpuPath = true; // use GPU by default
    eDeviceType device = eDeviceType::GPU;
    int deviceId = 0;
    int diameter = 2;
    float sigmaSpace = 2.0f;
    float sigmaColor = 10.0f;
    eBorderType borderMode = BORDER_TYPE_REPLICATE;
    float4 borderColor = {0.0f, 0.0f, 0.0f, 0.0f};

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
        if (!strcmp(argv[i], "-diameter")) {
            if (++i == argc) {
                ShowHelpAndExit("-diameter");
            }
            diameter = std::atoi(argv[i]);
            continue;
        }
        if (!strcmp(argv[i], "-sigma_space")) {
            if (++i == argc) {
                ShowHelpAndExit("-sigma_space");
            }
            sigmaSpace = std::atof(argv[i]);
            continue;
        }
        if (!strcmp(argv[i], "-sigma_color")) {
            if (++i == argc) {
                ShowHelpAndExit("-sigma_color");
            }
            sigmaColor = std::atof(argv[i]);
            continue;
        }
        if (!strcmp(argv[i], "-border_mode")) {
            if (++i == argc) {
                ShowHelpAndExit("-border_mode");
            }
            borderMode = static_cast<eBorderType>(std::atoi(argv[i]));
            continue;
        }
        if (!strcmp(argv[i], "-border_color")) {
            i++;
            if (i + 4 > argc) {
                ShowHelpAndExit("-border_color");
            }
            borderColor.x = static_cast<float>(atoi(argv[i++]));
            borderColor.y = static_cast<float>(atoi(argv[i++]));
            borderColor.z = static_cast<float>(atoi(argv[i++]));
            borderColor.w = static_cast<float>(atoi(argv[i]));
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
    hipStream_t stream = nullptr;
    if (gpuPath) {
        HIP_VALIDATE_NO_ERRORS(hipStreamCreate(&stream));
    }

    cv::Mat imageData = cv::imread(input_file_path);
    if (imageData.empty()) {
        std::cerr << "Failed to read the input image file" << std::endl;
        exit(1);
    }

    // Create input/output tensors for the image.
    TensorShape imageShape(TensorLayout(eTensorLayout::TENSOR_LAYOUT_NHWC), {1, imageData.rows, imageData.cols, imageData.channels()});
    DataType dtype(eDataType::DATA_TYPE_U8);
    Tensor input(imageShape, dtype, device);
    Tensor output(imageShape, dtype, device);

    // Move image data to input tensor
    size_t imageSizeInByte = input.shape().size() * input.dtype().size();
    auto inputData = input.exportData<TensorDataStrided>();
    if (gpuPath) {
        HIP_VALIDATE_NO_ERRORS(hipMemcpyAsync(inputData.basePtr(), imageData.data, imageSizeInByte, hipMemcpyHostToDevice, stream));
    } else {
        memcpy(inputData.basePtr(), imageData.data, imageSizeInByte);
    }

    BilateralFilter op;
    op(stream, input, output, diameter, sigmaColor, sigmaSpace, borderMode, borderColor, device);

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
    cv::Mat outImageData(imageData.rows, imageData.cols, imageData.type(), h_output.data());
    bool ret = cv::imwrite(output_file_path, outImageData);
    if (!ret) {
        std::cerr << "Faild to save output image to the file" << std::endl;
        exit(1);
    }

    std::cout << "Input image file: " << input_file_path << std::endl;
    std::cout << "Output image file: " << output_file_path << std::endl;
    if (gpuPath) {
        std::cout << "Operation on GPU device " << deviceId << std::endl;
    } else {
        std::cout << "Operation on CPU" << std::endl;
    }
    std::cout << "Image size: width = " << imageData.cols << ", height = " << imageData.rows << std::endl;

    return EXIT_SUCCESS;
}