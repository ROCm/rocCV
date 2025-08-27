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
    << "-d GPU device ID (0 for the first device, 1 for the second, etc.) - optional; default: 0" << std::endl
    << "-crop Crop rectangle (top_left_corner_x, top_left_corner_y, width, height)- optional; default: use the set value in the app" << std::endl;
    exit(0);
}

int main(int argc, char** argv) {
    std::string input_file_path;
    std::string output_file_path = "output.bmp";
    int deviceId = 0;
    Box_t cropRect = {0, 0, 1, 1};
    bool cropSet = false;

    if(argc < 3) {
        ShowHelpAndExit();
    }
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-h")) {
            ShowHelpAndExit();
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
    }

    HIP_VALIDATE_NO_ERRORS(hipSetDevice(deviceId));

    cv::Mat image_data = cv::imread(input_file_path);
    if (image_data.empty()) {
        std::cerr << "Failed to read the input image file" << std::endl;
        exit(1);
    }
    if (!cropSet) {
        cropRect = {(image_data.cols / 4), (image_data.rows / 4), (image_data.cols / 2), (image_data.rows / 2)};
    }

    // Create input/output tensors for the image.
    TensorShape inputShape(TensorLayout(eTensorLayout::TENSOR_LAYOUT_NHWC), {1, image_data.rows, image_data.cols, image_data.channels()});
    DataType dtype(eDataType::DATA_TYPE_U8);
    Tensor d_input(inputShape, dtype);

    TensorShape outShape(TensorLayout(eTensorLayout::TENSOR_LAYOUT_NHWC), {1, cropRect.height, cropRect.width, image_data.channels()});
    Tensor d_output(outShape, dtype);

    hipStream_t stream;
    HIP_VALIDATE_NO_ERRORS(hipStreamCreate(&stream));

    // Move image data to input tensor
    size_t image_size = d_input.shape().size() * d_input.dtype().size();
    auto d_input_data = d_input.exportData<TensorDataStrided>();
    HIP_VALIDATE_NO_ERRORS(hipMemcpyAsync(d_input_data.basePtr(), image_data.data, image_size, hipMemcpyHostToDevice, stream));

    CustomCrop op;
    op(nullptr, d_input, d_output, cropRect, eDeviceType::GPU);

    // Move image data back to host
    size_t outputSize = d_output.shape().size() * d_output.dtype().size();
    auto d_out_data = d_output.exportData<TensorDataStrided>();
    std::vector<uint8_t> h_output(outputSize);
    HIP_VALIDATE_NO_ERRORS(hipMemcpyAsync(h_output.data(), d_out_data.basePtr(), outputSize, hipMemcpyDeviceToHost, stream));

    HIP_VALIDATE_NO_ERRORS(hipStreamSynchronize(stream));

    // Write output image to disk
    cv::Mat output_image_data(cropRect.height, cropRect.width, image_data.type(), h_output.data());
    bool ret = cv::imwrite(output_file_path, output_image_data);
    if (!ret) {
        std::cerr << "Faild to save output image to the file" << std::endl;
    }

    return EXIT_SUCCESS;
}