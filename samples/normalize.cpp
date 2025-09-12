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
#include <core/image_format.hpp>
#include <core/tensor.hpp>
#include <iostream>
#include <fstream>
#include <op_normalize.hpp>
#include <opencv2/opencv.hpp>

using namespace roccv;

/**
 * @brief Normalize operation example.
 */

/**
 * @brief Example shift base parameter file content
 * 1 <-- number of images
 * 1 <-- scalar base indicator for image 1 (1: scalar base; 0: per pixel per channel shift)
 * 120.0 <-- base for channel 0
 * 110.0 <-- base for channel 1
 * 115.0 <-- base for channel 2
 */

/**
 * @brief Example scale parameter file content
 * 1 <-- number of images
 * 1 <-- scalar scale indicator for image 1 (1: scalar scale; 0: per pixel per channel scaling)
 * 80.0 <-- scale for channel 0
 * 75.0 <-- scale for channel 1
 * 65.0 <-- scale for channel 2
 */

 void ShowHelpAndExit(const char *option = NULL) {
    std::cout << "Options: " << option << std::endl
    << "-i Input File Path - required" << std::endl
    << "-o Output File Path - optional; default: output.bmp" << std::endl
    << "-cpu Select CPU instead of GPU to perform operation - optional; default choice is GPU path" << std::endl
    << "-d GPU device ID (0 for the first device, 1 for the second, etc.) - optional; default: 0" << std::endl
    << "-global_shift Global shift parameter - optional; default: 0.0f" << std::endl
    << "-global_scale Global scale parameter - optional; default: 1.0f" << std::endl
    << "-base_file Shifting base parameter file - optional; default: use the set value in the app" << std::endl
    << "-scale_file Scaling parameter file - optional; default: use the set value in the app" << std::endl
    << "-stddev_scale Scaling parameter is standard deviation (0/1)- optional; default: 0 (false)" << std::endl
    << "-epsilon Epsilon parameter - optional; default: 0.1f" << std::endl;
    exit(0);
}

int main(int argc, char** argv) {
    std::string input_file_path;
    std::string base_file_path;
    std::string scale_file_path;
    std::string output_file_path = "output.bmp";
    bool gpuPath = true; // use GPU by default
    eDeviceType device = eDeviceType::GPU;
    int deviceId = 0;
    float globalScale = 1.0f;
    bool scaleSet = false; // User sets the scale parameters in a text file
    float globalShift = 0.0f;
    bool baseSet = false; // User sets the shift parameters in a text file
    uint32_t flags = 0;
    float epsilon = 0.1f;

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
        if (!strcmp(argv[i], "-global_shift")) {
            if (++i == argc) {
                ShowHelpAndExit("-global_shift");
            }
            globalShift = std::atof(argv[i]);
            continue;
        }
        if (!strcmp(argv[i], "-global_scale")) {
            if (++i == argc) {
                ShowHelpAndExit("-global_scale");
            }
            globalScale = std::atof(argv[i]);
            continue;
        }
        if (!strcmp(argv[i], "-base_file")) {
            if (++i == argc) {
                ShowHelpAndExit("-base_file");
            }
            base_file_path = argv[i];
            baseSet = true;
            continue;
        }
        if (!strcmp(argv[i], "-scale_file")) {
            if (++i == argc) {
                ShowHelpAndExit("-scale_file");
            }
            scale_file_path = argv[i];
            scaleSet = true;
            continue;
        }
        if (!strcmp(argv[i], "-stddev_scale")) {
            if (++i == argc) {
                ShowHelpAndExit("-stddev_scale");
            }
            flags = std::atoi(argv[i]) ? ROCCV_NORMALIZE_SCALE_IS_STDDEV : 0;
            continue;
        }
        if (!strcmp(argv[i], "-epsilon")) {
            if (++i == argc) {
                ShowHelpAndExit("-epsilon");
            }
            epsilon = std::atof(argv[i]);
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

    // Set up scale tensor
    int scaleBatchSize;
    Size2D scaleSize;
    roccv::ImageFormat scaleFormat = FMT_RGBf32;
    std::vector<float> scaleData;
    if (scaleSet) {
        std::ifstream scale_param_file(scale_file_path);
        if (scale_param_file.is_open()) {
            std::string line;
            std::getline(scale_param_file, line);
            scaleBatchSize = std::stoi(line.c_str());
            if (scaleBatchSize > 0) {
                for (int i = 0; i < scaleBatchSize; i++) {
                    std::getline(scale_param_file, line);
                    int scalarScale = std::stoi(line.c_str());
                    if (scalarScale) {
                        scaleSize = {1, 1};
                        int currIdx = scaleData.size();
                        scaleData.resize(currIdx + 3); // 3 channels
                        for (int b = currIdx; b < 3 + currIdx; b++) {
                            std::getline(scale_param_file, line);
                            scaleData[b] = std::atof(line.c_str());
                        }
                    } else {
                        std::cerr << "Per pixel scale is not supported in current sample." << std::endl;
                        exit(1);
                    }
                }
            } else {
                std::cerr << "Invalid scale batch size: " << scaleBatchSize << std::endl;
                exit(1);
            }
        } else {
            std::cerr << "Failed to open scale parameter file " << scale_file_path << std::endl;
            exit(1);
        }
    } else {
        // Use default scale params
        scaleBatchSize = 1;
        scaleSize = {1, 1};
        scaleData = {1.0, 1.0, 1.0};
    }
    Tensor scaleTensor(scaleBatchSize, scaleSize, scaleFormat, device);
    auto scaleTensorData = scaleTensor.exportData<TensorDataStrided>();
    if (gpuPath) {
        HIP_VALIDATE_NO_ERRORS(hipMemcpyAsync(scaleTensorData.basePtr(), scaleData.data(), scaleData.size() * sizeof(float), hipMemcpyHostToDevice, stream));
    } else {
        memcpy(scaleTensorData.basePtr(), scaleData.data(), scaleData.size() * sizeof(float));
    }

    // Set up base tensor
    int baseBatchSize;
    Size2D baseSize;
    roccv::ImageFormat baseFormat = FMT_RGBf32;
    std::vector<float> baseData;
    if (baseSet) {
        std::ifstream base_param_file(base_file_path);
        if (base_param_file.is_open()) {
            std::string line;
            std::getline(base_param_file, line);
            baseBatchSize = std::stoi(line.c_str());
            if (baseBatchSize > 0) {
                for (int i = 0; i < baseBatchSize; i++) {
                    std::getline(base_param_file, line);
                    int scalarBase = std::stoi(line.c_str());
                    if (scalarBase) {
                        baseSize = {1, 1};
                        int currIdx = baseData.size();
                        baseData.resize(currIdx + 3); // 3 channels
                        for (int b = currIdx; b < 3 + currIdx; b++) {
                            std::getline(base_param_file, line);
                            baseData[b] = std::atof(line.c_str());
                        }
                    } else {
                        std::cerr << "Per pixel shift is not supported in current sample." << std::endl;
                        exit(1);
                    }
                }
            } else {
                std::cerr << "Invalid base batch size: " << baseBatchSize << std::endl;
                exit(1);
            }
        } else {
            std::cerr << "Failed to open base parameter file " << base_file_path << std::endl;
            exit(1);
        }
    } else {
        // Use default base params
        baseBatchSize = 1;
        baseSize = {1, 1};
        baseData = {0.0, 0.0, 0.0};
    }
    Tensor baseTensor(baseBatchSize, baseSize, baseFormat, device);
    auto baseTensorData = baseTensor.exportData<TensorDataStrided>();
    if (gpuPath) {
        HIP_VALIDATE_NO_ERRORS(hipMemcpyAsync(baseTensorData.basePtr(), baseData.data(), baseData.size() * sizeof(float), hipMemcpyHostToDevice, stream));
    } else {
        memcpy(baseTensorData.basePtr(), baseData.data(), baseData.size() * sizeof(float));
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

    Normalize op;
    op(stream, input, baseTensor, scaleTensor, output, globalScale, globalShift, epsilon, flags, device);

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