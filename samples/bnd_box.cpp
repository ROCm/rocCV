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

#include <op_bnd_box.hpp>
#include <core/tensor.hpp>
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace roccv;

/**
 * @brief Bounding Box operation example.
 */
int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <image_filename> <output_filename> <device_id>"
                  << std::endl;
        return EXIT_FAILURE;
    }
    
    HIP_VALIDATE_NO_ERRORS(hipSetDevice(std::stoi(argv[3])));

    cv::Mat image_data = cv::imread(argv[1]);

    // Create input/output tensors for the image.
    TensorShape shape(
        TensorLayout(eTensorLayout::TENSOR_LAYOUT_NHWC),
        {1, image_data.rows, image_data.cols, image_data.channels()});
    DataType dtype(eDataType::DATA_TYPE_U8);

    Tensor d_in(shape, dtype);
    Tensor d_out(shape, dtype);

    hipStream_t stream;
    HIP_VALIDATE_NO_ERRORS(hipStreamCreate(&stream));

    auto width = 100;
    auto height = 100;

    /*
     * Parameters
     */
    std::vector<int32_t> bboxes_size_vector(1, 3);
    std::vector<BndBox_t> bbox_vector(3);
    bbox_vector[0].box.x = width / 4;
    bbox_vector[0].box.y = height / 4;
    bbox_vector[0].box.width = width / 2;
    bbox_vector[0].box.height = height / 2;
    bbox_vector[0].thickness = 5;
    bbox_vector[0].borderColor = {0, 0, 255, 200};
    bbox_vector[0].fillColor = {0, 255, 0, 100};
    bbox_vector[1].box.x = width / 3;
    bbox_vector[1].box.y = height / 3;
    bbox_vector[1].box.width = width / 3 * 2;
    bbox_vector[1].box.height = height / 4;
    bbox_vector[1].thickness = -1;
    bbox_vector[1].borderColor = {90, 16, 181, 50};
    bbox_vector[2].box.x = -50;
    bbox_vector[2].box.y = (2 * height) / 3;
    bbox_vector[2].box.width = width + 50;
    bbox_vector[2].box.height = height / 3 + 50;
    bbox_vector[2].thickness = 0;
    bbox_vector[2].borderColor = {0, 0, 0, 50};
    bbox_vector[2].fillColor = {111, 159, 232, 150};
    BndBoxes_t bboxes{1, bboxes_size_vector, bbox_vector};

    // Move image data to input tensor
    size_t image_size = d_in.shape().size() * d_in.dtype().size();
    auto d_input_data = d_in.exportData<TensorDataStrided>();
    HIP_VALIDATE_NO_ERRORS(hipMemcpyAsync(d_input_data.basePtr(),
                                          image_data.data, image_size,
                                          hipMemcpyHostToDevice, stream));

    BndBox op;
    op(stream, d_in, d_out, bboxes);

    // Move image data back to host
    auto d_out_data = d_out.exportData<TensorDataStrided>();
    std::vector<uint8_t> h_output(image_size);
    HIP_VALIDATE_NO_ERRORS(hipMemcpyAsync(h_output.data(), d_out_data.basePtr(),
                                          image_size, hipMemcpyDeviceToHost,
                                          stream));

    HIP_VALIDATE_NO_ERRORS(hipStreamSynchronize(stream));

    // Write normalized image to disk
    cv::Mat output_image_data(image_data.rows, image_data.cols, CV_8UC3,
                              h_output.data());
    cv::imwrite(argv[2], output_image_data);

    return EXIT_SUCCESS;

}
