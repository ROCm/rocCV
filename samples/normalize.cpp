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
#include <op_normalize.hpp>
#include <opencv2/opencv.hpp>

using namespace roccv;

/**
 * @brief Normalize operation example.
 */
int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <image_filename> <output_filename> <device_id>" << std::endl;
        return EXIT_FAILURE;
    }

    HIP_VALIDATE_NO_ERRORS(hipSetDevice(std::stoi(argv[3])));

    cv::Mat image_data = cv::imread(argv[1]);

    // Create input/output tensors for the image.
    TensorShape shape(TensorLayout(eTensorLayout::TENSOR_LAYOUT_NHWC),
                      {1, image_data.rows, image_data.cols, image_data.channels()});
    DataType dtype(eDataType::DATA_TYPE_U8);

    Tensor d_in(shape, dtype);
    Tensor d_out(shape, dtype);

    // Create mean/stddev tensors. We store one value per channel per image, so
    // we only need a size of 3.
    TensorShape param_shape(TensorLayout(eTensorLayout::TENSOR_LAYOUT_NHWC), {1, 1, 1, image_data.channels()});
    DataType param_dtype(eDataType::DATA_TYPE_F32);
    Tensor d_mean(param_shape, param_dtype);
    Tensor d_stddev(param_shape, param_dtype);

    std::vector<float> mean_values = {121.816, 117.935, 98.395};
    std::vector<float> stddev_values = {82.195, 62.885, 61.023};
    hipMemcpy(d_mean.exportData<TensorDataStrided>().basePtr(), mean_values.data(), mean_values.size() * sizeof(float),
              hipMemcpyHostToDevice);
    hipMemcpy(d_stddev.exportData<TensorDataStrided>().basePtr(), stddev_values.data(),
              stddev_values.size() * sizeof(float), hipMemcpyHostToDevice);

    hipStream_t stream;
    HIP_VALIDATE_NO_ERRORS(hipStreamCreate(&stream));

    // Move image data to input tensor
    size_t image_size = d_in.shape().size() * d_in.dtype().size();
    auto d_input_data = d_in.exportData<TensorDataStrided>();
    HIP_VALIDATE_NO_ERRORS(
        hipMemcpyAsync(d_input_data.basePtr(), image_data.data, image_size, hipMemcpyHostToDevice, stream));

    Normalize op;
    op(stream, d_in, d_mean, d_stddev, d_out, 85.0f, 180.0f, 0.0f, ROCCV_NORMALIZE_SCALE_IS_STDDEV, eDeviceType::GPU);

    // Move image data back to device
    auto d_out_data = d_out.exportData<TensorDataStrided>();
    std::vector<uint8_t> h_output(image_size);
    HIP_VALIDATE_NO_ERRORS(
        hipMemcpyAsync(h_output.data(), d_out_data.basePtr(), image_size, hipMemcpyDeviceToHost, stream));

    HIP_VALIDATE_NO_ERRORS(hipStreamSynchronize(stream));

    // Write normalized image to disk
    cv::Mat output_image_data(image_data.rows, image_data.cols, CV_8UC3, h_output.data());
    cv::imwrite(argv[2], output_image_data);

    return EXIT_SUCCESS;
}