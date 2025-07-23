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
#include <op_gamma_contrast.hpp>
#include <opencv2/opencv.hpp>

using namespace roccv;

/**
 * @brief Gamma contrast operator sample app.
 */
int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0]
                  << " <input_image> <output_image>"
                  << std::endl;
        return EXIT_FAILURE;
    }

    // Device to use in this sample will be the GPU
    eDeviceType device =  eDeviceType::GPU;
    
    // Load input image using the OpenCV library.
    // The Mat image_data will store all of the data of the image
    // Image width can be gotten with image_data.rows
    // Image height can be gotten with image_data.cols
    // The amount of channels can be gotten with image_data.channels()
    cv::Mat image_data = cv::imread(argv[1]);

    // Batch size is needed to create the input and output tensors
    int batchSize = 1;

    // A floating point gamma value to apply to the input image
    float gammaValue = 2.2;

    // Create input/output tensors
    // Tensor shape
    //      - Takes layout as input, in this case NHWC (N - batch size, H - image height, W - image width, C - number of channels)
    //      - Also takes the datatype, in this case U8 or an unsigned integer of 8 bits.
    TensorShape shape(
        TensorLayout(eTensorLayout::TENSOR_LAYOUT_NHWC),
        {batchSize, image_data.rows, image_data.cols, image_data.channels()});
    DataType dtype(eDataType::DATA_TYPE_U8);

    Tensor input(shape, dtype, device);
    Tensor output(shape, dtype, device);

    hipStream_t stream;
    HIP_VALIDATE_NO_ERRORS(hipStreamCreate(&stream));

    // imageSize is needed to know how much data needs to be copied to the GPU
    size_t imageSize = image_data.rows * image_data.cols * image_data.channels() * sizeof(uint8_t);
    
    auto input_data = input.exportData<TensorDataStrided>();
    HIP_VALIDATE_NO_ERRORS(
            hipMemcpy(static_cast<uint8_t *>(input_data.basePtr()), image_data.data, imageSize, hipMemcpyHostToDevice));
    
    // Apply gamma correction
    GammaContrast gamma_contrast;
    gamma_contrast(stream, input, output, gammaValue, device);

    // Move output data back to host
    auto output_data = output.exportData<TensorDataStrided>();
    std::vector<uint8_t> h_output(imageSize);
    HIP_VALIDATE_NO_ERRORS(hipMemcpy(h_output.data(), output_data.basePtr(), imageSize, hipMemcpyDeviceToHost));
    
    HIP_VALIDATE_NO_ERRORS(hipStreamSynchronize(stream));

    // Save the gamma-corrected image
    cv::Mat output_image(image_data.rows, image_data.cols, CV_8UC3, h_output.data());
    cv::imwrite(argv[2], output_image);

    std::cout << "Gamma correction applied successfully. Output saved to: " << argv[2] << std::endl;
    
    return EXIT_SUCCESS;
}
