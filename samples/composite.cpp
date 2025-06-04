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

#include <core/hip_assert.h>

#include <core/tensor.hpp>
#include <op_composite.hpp>
#include <opencv2/opencv.hpp>

using namespace roccv;

int main(int argc, char** argv) {
    if (argc != 6) {
        std::cerr << "Usage: " << argv[0]
                  << " <background_filename> <foreground_filename> <mask_filename> <output_filename> <device_id>"
                  << std::endl;
        return EXIT_FAILURE;
    }

    hipStream_t stream;
    HIP_VALIDATE_NO_ERRORS(hipStreamCreate(&stream));

    HIP_VALIDATE_NO_ERRORS(hipSetDevice(std::stoi(argv[5])));

    cv::Mat background_data = cv::imread(argv[1]);
    cv::Mat foreground_data = cv::imread(argv[2]);
    cv::Mat mask_data = cv::imread(argv[3], cv::IMREAD_GRAYSCALE);

    // Create input/output tensors for the image.
    DataType dtype(eDataType::DATA_TYPE_U8);
    TensorShape background_shape(TensorLayout(eTensorLayout::TENSOR_LAYOUT_NHWC),
                                 {1, background_data.rows, background_data.cols, background_data.channels()});
    Tensor background_tensor(background_shape, dtype);

    TensorShape foreground_shape(TensorLayout(eTensorLayout::TENSOR_LAYOUT_NHWC),
                                 {1, foreground_data.rows, foreground_data.cols, foreground_data.channels()});
    Tensor foreground_tensor(foreground_shape, dtype);

    TensorShape mask_shape(TensorLayout(eTensorLayout::TENSOR_LAYOUT_NHWC),
                           {1, mask_data.rows, mask_data.cols, mask_data.channels()});
    Tensor mask_tensor(mask_shape, dtype);

    Tensor output_tensor(background_shape, dtype);

    auto bt_data = background_tensor.exportData<TensorDataStrided>();
    HIP_VALIDATE_NO_ERRORS(hipMemcpyAsync(bt_data.basePtr(), background_data.data,
                                          bt_data.shape().size() * bt_data.dtype().size(), hipMemcpyHostToDevice,
                                          stream));

    auto ft_data = foreground_tensor.exportData<TensorDataStrided>();
    HIP_VALIDATE_NO_ERRORS(hipMemcpyAsync(ft_data.basePtr(), foreground_data.data,
                                          foreground_tensor.shape().size() * foreground_tensor.dtype().size(),
                                          hipMemcpyHostToDevice, stream));

    auto m_data = mask_tensor.exportData<TensorDataStrided>();
    HIP_VALIDATE_NO_ERRORS(hipMemcpyAsync(m_data.basePtr(), mask_data.data,
                                          mask_tensor.shape().size() * mask_tensor.dtype().size(),
                                          hipMemcpyHostToDevice, stream));

    hipEvent_t begin, end;
    hipEventCreate(&begin);
    hipEventCreate(&end);

    hipEventRecord(begin, stream);
    roccv::Composite op;
    op(stream, foreground_tensor, background_tensor, mask_tensor, output_tensor);
    hipEventRecord(end, stream);
    hipEventSynchronize(end);

    float duration;
    hipEventElapsedTime(&duration, begin, end);
    printf("Kernel execution time: %fms\n", duration);

    hipEventDestroy(begin);
    hipEventDestroy(end);

    // Move image data back to device
    auto out_data = output_tensor.exportData<TensorDataStrided>();
    std::vector<uint8_t> out_h(output_tensor.shape().size());
    HIP_VALIDATE_NO_ERRORS(hipMemcpyAsync(out_h.data(), out_data.basePtr(),
                                          output_tensor.shape().size() * output_tensor.dtype().size(),
                                          hipMemcpyDeviceToHost, stream));

    HIP_VALIDATE_NO_ERRORS(hipStreamSynchronize(stream));

    // Write normalized image to disk
    cv::Mat output_image_data(background_data.rows, background_data.cols, CV_8UC3, out_h.data());
    cv::imwrite(argv[4], output_image_data);

    return EXIT_SUCCESS;
}