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

using namespace roccv;

/**
 * @brief Normalize operation example.
 */
int main(int argc, char** argv) {
    if (argc != 11) {
        std::cerr << "Usage: " << argv[0]
                  << " <image_filename> <output_filename> <top> <left> <r> <g> <b> <a> <border_mode> <device_id>"
                  << std::endl;
        return EXIT_FAILURE;
    }

    HIP_VALIDATE_NO_ERRORS(hipSetDevice(std::stoi(argv[10])));

    int32_t top = std::stoi(argv[3]);
    int32_t left = std::stoi(argv[4]);
    float r = std::stof(argv[5]);
    float g = std::stof(argv[6]);
    float b = std::stof(argv[7]);
    float a = std::stof(argv[8]);
    eBorderType border_mode = static_cast<eBorderType>(std::stoi(argv[9]));

    cv::Mat image_data = cv::imread(argv[1]);

    // Create input/output tensors for the image.
    TensorShape shape(TensorLayout(eTensorLayout::TENSOR_LAYOUT_NHWC),
                      {1, image_data.rows, image_data.cols, image_data.channels()});
    DataType dtype(eDataType::DATA_TYPE_U8);

    TensorShape o_shape(TensorLayout(eTensorLayout::TENSOR_LAYOUT_NHWC),
                        {1, image_data.rows + top * 2, image_data.cols + left * 2, image_data.channels()});

    Tensor d_in(shape, dtype);
    Tensor d_out(o_shape, dtype);

    hipStream_t stream;
    HIP_VALIDATE_NO_ERRORS(hipStreamCreate(&stream));

    // Move image data to input tensor
    size_t image_size = d_in.shape().size() * d_in.dtype().size();
    auto d_input_data = d_in.exportData<TensorDataStrided>();
    HIP_VALIDATE_NO_ERRORS(
        hipMemcpyAsync(d_input_data.basePtr(), image_data.data, image_size, hipMemcpyHostToDevice, stream));

    CopyMakeBorder op;
    op(stream, d_in, d_out, top, left, border_mode, {b, g, r, a});

    // Move image data back to device
    auto d_out_data = d_out.exportData<TensorDataStrided>();
    size_t out_image_size = d_out.shape().size() * d_out.dtype().size();
    std::vector<uint8_t> h_output(out_image_size);
    HIP_VALIDATE_NO_ERRORS(
        hipMemcpyAsync(h_output.data(), d_out_data.basePtr(), out_image_size, hipMemcpyDeviceToHost, stream));

    HIP_VALIDATE_NO_ERRORS(hipStreamSynchronize(stream));

    cv::Mat output_image_data(image_data.rows + top * 2, image_data.cols + left * 2, CV_8UC3, h_output.data());
    cv::imwrite(argv[2], output_image_data);

    return EXIT_SUCCESS;
}