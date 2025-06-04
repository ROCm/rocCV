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
 * @brief Gamma contrast operator example.
 */
int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <input_image> <output_image> <output image 2>"
                  << std::endl;
        return EXIT_FAILURE;
    }
    eDeviceType device =  eDeviceType::GPU;

    
    
    // Load input image
    cv::Mat image_data = cv::imread(argv[1]);

    int batchSize = 2;
    std::vector<float> gammaValues = {2.2, 0.8};

    Tensor gammaTensor(TensorShape(TensorLayout(eTensorLayout::TENSOR_LAYOUT_N), {2}),
                              DataType(eDataType::DATA_TYPE_F32), device);
    
    auto gammaTensorData = gammaTensor.exportData<TensorDataStrided>();

    HIP_VALIDATE_NO_ERRORS(hipMemcpy(gammaTensorData.basePtr(), gammaValues.data(),
                                             gammaTensor.shape().size() * gammaTensor.dtype().size(),
                                             hipMemcpyHostToDevice));

    // Create input/output tensors
    TensorShape shape(
        TensorLayout(eTensorLayout::TENSOR_LAYOUT_NHWC),
        {batchSize, image_data.rows, image_data.cols, image_data.channels()});
    DataType dtype(eDataType::DATA_TYPE_U8);

    Tensor d_in(shape, dtype, device);
    Tensor d_out(shape, dtype, device);

    hipStream_t stream;
    HIP_VALIDATE_NO_ERRORS(hipStreamCreate(&stream));

    // Move image data to input tensor
    size_t mem_offset = 0;
    size_t num_images = 2;
    size_t imageSize = image_data.rows * image_data.cols * image_data.channels() * sizeof(uint8_t);
    auto d_in_data = d_in.exportData<TensorDataStrided>();
    for (int b = 0; b < num_images; b++) {
        HIP_VALIDATE_NO_ERRORS(
            hipMemcpy(static_cast<uint8_t *>(d_in_data.basePtr()) + mem_offset, image_data.data, imageSize, hipMemcpyHostToDevice));
        mem_offset += imageSize;
    }

    // Apply gamma correction
    GammaContrast gamma_contrast;
    gamma_contrast(stream, d_in, d_out, gammaTensor, eDeviceType::GPU);

    // Move output data back to host
    auto d_out_data = d_out.exportData<TensorDataStrided>();
    std::vector<uint8_t> h_output(imageSize);
    HIP_VALIDATE_NO_ERRORS(hipMemcpy(h_output.data(), d_out_data.basePtr(), imageSize, hipMemcpyDeviceToHost));
    std::vector<uint8_t> h_output_2(imageSize);
    HIP_VALIDATE_NO_ERRORS(hipMemcpy(h_output_2.data(), static_cast<uint8_t *>(d_out_data.basePtr()) + imageSize, imageSize, hipMemcpyDeviceToHost));

    HIP_VALIDATE_NO_ERRORS(hipStreamSynchronize(stream));

    // Save the gamma-corrected image
    cv::Mat output_image(image_data.rows, image_data.cols, CV_8UC3, h_output.data());
    cv::Mat output_image2(image_data.rows, image_data.cols, CV_8UC3, h_output_2.data());
    cv::imwrite(argv[2], output_image);
    cv::imwrite(argv[3], output_image2);

    std::cout << "Gamma correction applied successfully. Output saved to: " << argv[2] << std::endl;
    std::cout << "Gamma correction applied successfully. Output saved to: " << argv[3] << std::endl;
    
    return EXIT_SUCCESS;
}
