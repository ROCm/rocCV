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
#include <fstream>
#include <iostream>
#include <op_bnd_box.hpp>
#include <opencv2/opencv.hpp>

using namespace roccv;

/**
 * @brief Bounding Box operation example.
 */

/**
 * @brief Example bounding box list file content
 * 1 <-- number of images
 * 2 <-- number of boxes for image 1
 * 50 <-- X coordinate of top-left corner of box 1
 * 50 <-- Y coordinate of top-left corner of box 1
 * 100 <-- width of box 1
 * 50 <-- height of box 1
 * 5 <-- thickness of box boundary of box 1
 * 0 <-- B component of box border color of box 1
 * 0 <-- G component of box border color of box 1
 * 255 <-- R component of box border color of box 1
 * 200 <-- alpha component of box border color of box 1
 * 0 <-- B component of box fill color of box 1
 * 255 <-- G component of box fill color of box 1
 * 0 <-- R component of box fill color of box 1
 * 100 <-- alpha component of box fill color of box 1
 * 250 <-- X coordinate of top-left corner of box 2
 * 250 <-- Y coordinate of top-left corner of box 2
 * 50 <-- width of box 2
 * 100 <-- height of box 2
 * 10 <-- thickness of box boundary of box 2
 * 255 <-- B component of box border color of box 2
 * 0 <-- G component of box border color of box 2
 * 0 <-- R component of box border color of box 2
 * 200 <-- alpha component of box border color of box 2
 * 0 <-- B component of box fill color of box 2
 * 0 <-- G component of box fill color of box 2
 * 0 <-- R component of box fill color of box 2
 * 0 <-- alpha component of box fill color of box 2
 */

void ShowHelpAndExit(const char* option = NULL) {
    std::cout << "Options: " << option << std::endl
              << "-i Input File Path - required" << std::endl
              << "-o Output File Path - optional; default: output.bmp" << std::endl
              << "-cpu Select CPU instead of GPU to perform operation - optional; default choice is GPU path"
              << std::endl
              << "-d GPU device ID (0 for the first device, 1 for the second, etc.) - optional; default: 0" << std::endl
              << "-box_file Bounding box list file - optional; default: use the set value in the app" << std::endl;
    exit(0);
}

int main(int argc, char** argv) {
    std::string input_file_path;
    std::string box_file_path;
    std::string output_file_path = "output.bmp";
    bool gpuPath = true;  // use GPU by default
    eDeviceType device = eDeviceType::GPU;
    int deviceId = 0;
    bool boxSet = false;  // User sets the bounding box list data in a text file

    if (argc < 3) {
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
        if (!strcmp(argv[i], "-box_file")) {
            if (++i == argc) {
                ShowHelpAndExit("-o");
            }
            box_file_path = argv[i];
            boxSet = true;
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

    int batchSize = 1;
    std::vector<std::vector<BndBox_t>> bbox_vector;
    if (boxSet) {
        std::ifstream box_list_file(box_file_path);
        if (box_list_file.is_open()) {
            std::string line;
            std::getline(box_list_file, line);
            batchSize = std::stoi(line.c_str());
            if (batchSize > 0) {
                bbox_vector.resize(batchSize);
                for (int i = 0; i < batchSize; i++) {
                    std::getline(box_list_file, line);
                    int numBoxes = std::stoi(line.c_str());
                    if (numBoxes > 0) {
                        for (int b = 0; b < numBoxes; b++) {
                            BndBox_t box;
                            std::getline(box_list_file, line);
                            box.box.x = std::atoi(line.c_str());
                            std::getline(box_list_file, line);
                            box.box.y = std::atoi(line.c_str());
                            std::getline(box_list_file, line);
                            box.box.width = std::atoi(line.c_str());
                            std::getline(box_list_file, line);
                            box.box.height = std::atoi(line.c_str());

                            std::getline(box_list_file, line);
                            box.thickness = std::atoi(line.c_str());

                            std::getline(box_list_file, line);
                            box.borderColor.r = std::atoi(line.c_str());
                            std::getline(box_list_file, line);
                            box.borderColor.g = std::atoi(line.c_str());
                            std::getline(box_list_file, line);
                            box.borderColor.b = std::atoi(line.c_str());
                            std::getline(box_list_file, line);
                            box.borderColor.a = std::atoi(line.c_str());

                            std::getline(box_list_file, line);
                            box.fillColor.r = std::atoi(line.c_str());
                            std::getline(box_list_file, line);
                            box.fillColor.g = std::atoi(line.c_str());
                            std::getline(box_list_file, line);
                            box.fillColor.b = std::atoi(line.c_str());
                            std::getline(box_list_file, line);
                            box.fillColor.a = std::atoi(line.c_str());

                            bbox_vector[i].push_back(box);
                        }
                    } else {
                        std::cerr << "Invalid number of boxes: " << numBoxes << "for image: " << i << std::endl;
                        exit(1);
                    }
                }
            } else {
                std::cerr << "Invalid batch size: " << batchSize << std::endl;
                exit(1);
            }
        } else {
            std::cerr << "Failed to open bounding box list file " << box_file_path << std::endl;
            exit(1);
        }
    } else {
        auto width = imageData.cols;
        auto height = imageData.rows;
        bbox_vector = {
            {
                {{width / 4, height / 4, width / 2, height / 2}, 5, {0, 0, 255, 200}, {0, 255, 0, 100}},
                {{width / 3, height / 3, width / 3 * 2, height / 4}, -1, {90, 16, 181, 50}, {0, 0, 0, 0}},
                {{-50, (height * 2) / 3, width + 50, height / 3 + 50}, 0, {0, 0, 0, 0}, {111, 159, 232, 150}},
            },
        };
    }
    BndBoxes bboxes(bbox_vector);

    // Create input/output tensors for the image.
    TensorShape imageShape(TensorLayout(eTensorLayout::TENSOR_LAYOUT_NHWC),
                           {1, imageData.rows, imageData.cols, imageData.channels()});
    DataType dtype(eDataType::DATA_TYPE_U8);
    Tensor input(imageShape, dtype, device);
    Tensor output(imageShape, dtype, device);

    hipStream_t stream = nullptr;
    if (gpuPath) {
        HIP_VALIDATE_NO_ERRORS(hipStreamCreate(&stream));
    }

    // Move image data to input tensor
    size_t imageSizeInByte = input.shape().size() * input.dtype().size();
    auto inputData = input.exportData<TensorDataStrided>();
    if (gpuPath) {
        HIP_VALIDATE_NO_ERRORS(
            hipMemcpyAsync(inputData.basePtr(), imageData.data, imageSizeInByte, hipMemcpyHostToDevice, stream));
    } else {
        memcpy(inputData.basePtr(), imageData.data, imageSizeInByte);
    }

    BndBox op;
    op(stream, input, output, bboxes, device);

    // Move image data back to host
    size_t outputSize = output.shape().size() * output.dtype().size();
    auto outData = output.exportData<TensorDataStrided>();
    std::vector<uint8_t> h_output(outputSize);
    if (gpuPath) {
        HIP_VALIDATE_NO_ERRORS(
            hipMemcpyAsync(h_output.data(), outData.basePtr(), outputSize, hipMemcpyDeviceToHost, stream));
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