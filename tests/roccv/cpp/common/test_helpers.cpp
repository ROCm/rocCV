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

#include "test_helpers.hpp"

#include <core/hip_assert.h>

#include <core/exception.hpp>

namespace roccv {
namespace tests {
Tensor createTensorFromImage(const std::string& filename, DataType dtype, eDeviceType device, bool grayscale) {
    cv::Mat image = cv::imread(filename, grayscale ? cv::IMREAD_GRAYSCALE : 1);
    if (image.total() <= 0) {
        std::cerr << "Unable to read image: " << filename << std::endl;
        throw Exception(eStatusType::INVALID_VALUE);
    }

    std::vector<uint8_t> image_data;
    image_data.assign(image.data, image.data + image.total() * image.elemSize());

    TensorLayout layout(eTensorLayout::TENSOR_LAYOUT_NHWC);
    TensorShape shape(layout, {1, image.rows, image.cols, image.channels()});
    Tensor tensor(shape, dtype, device);

    copyData<uint8_t>(tensor, image_data, device);

    return tensor;
}

eTestStatusType compareImage(const Tensor& tensor, const std::string& filename, float error_threshold) {
    cv::Mat image = cv::imread(filename);

    if (image.total() <= 0) {
        std::cerr << "Unable to read image " << filename << std::endl;
        return eTestStatusType::TEST_FAILURE;
    }

    std::vector<uint8_t> image_data;
    image_data.assign(image.data, image.data + image.total() * image.elemSize());

    return compareArray<uint8_t>(tensor, image_data, error_threshold);
}

void writeTensor(const Tensor& tensor, const std::string& output_file) {
    size_t image_size = tensor.shape().size() * tensor.dtype().size();
    std::vector<uint8_t> host_data(tensor.shape().size());
    auto tensor_data = tensor.exportData<TensorDataStrided>();

    switch (tensor.device()) {
        case eDeviceType::GPU: {
            HIP_VALIDATE_NO_ERRORS(
                hipMemcpy(host_data.data(), tensor_data.basePtr(), image_size, hipMemcpyDeviceToHost));
            break;
        }

        case eDeviceType::CPU: {
            memcpy(host_data.data(), tensor_data.basePtr(), image_size);
        }
    }

    size_t image_width = tensor_data.shape(tensor.layout().width_index());
    size_t image_height = tensor_data.shape(tensor.layout().height_index());

    cv::Mat output_image_data(image_height, image_width, CV_8UC3, host_data.data());
    cv::imwrite(output_file, output_image_data);
}

}  // namespace tests
}  // namespace roccv