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
#pragma once

#include <core/exception.hpp>
#include <core/tensor.hpp>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <span>

namespace roccv {
namespace tests {

#define EXPECT_EXCEPTION(call_, expected_status_)                                                                   \
    try {                                                                                                           \
        call_;                                                                                                      \
        std::cerr << __FILE__ << "[" << __LINE__ << "]: "                                                           \
                  << "Expected (" << ExceptionMessage::getMessageByEnum(expected_status_)                           \
                  << ") but completed instead." << std::endl;                                                       \
        throw 1;                                                                                                    \
    } catch (Exception e) {                                                                                         \
        eStatusType received_ = e.getStatusEnum();                                                                  \
        if (received_ != expected_status_) {                                                                        \
            std::cerr << __FILE__ << "[" << __LINE__ << "]: "                                                       \
                      << "Expected (" << ExceptionMessage::getMessageByEnum(expected_status_) << ") but received (" \
                      << ExceptionMessage::getMessageByEnum(received_) << ") instead." << std::endl;                \
            throw 1;                                                                                                \
        }                                                                                                           \
    }

#define EXPECT_STATUS(call_, expected_status_)                                                                      \
    {                                                                                                               \
        eStatusType received_ = call_;                                                                              \
        if (received_ != expected_status_) {                                                                        \
            std::cerr << __FILE__ << "[" << __LINE__ << "]: "                                                       \
                      << "Expected (" << ExceptionMessage::getMessageByEnum(expected_status_) << ") but received (" \
                      << ExceptionMessage::getMessageByEnum(received_) << ") instead." << std::endl;                \
            exit(1);                                                                                                \
        }                                                                                                           \
    }

#define EXPECT_TEST_STATUS(call_, expected_status_)                                                                 \
    {                                                                                                               \
        eTestStatusType received_ = call_;                                                                          \
        if (received_ != expected_status_) {                                                                        \
            std::cerr << __FILE__ << "[" << __LINE__ << "]: "                                                       \
                      << "Expected (" << ExceptionMessage::getMessageByEnum(expected_status_) << ") but received (" \
                      << ExceptionMessage::getMessageByEnum(received_) << ") instead." << std::endl;                \
            exit(1);                                                                                                \
        }                                                                                                           \
    }

#define CHECK_ERROR(call_) EXPECT_STATUS(call_, SUCCESS)

/**
 * @brief Creates a NHWC tensor which contains data loaded from an image.
 *
 * @param filename A filename pointing to an image.
 * @param dtype The datatype of the requested tensor.
 * @param device The device which this tensor should be allocated on.
 * @param grayscale Loads image as grayscale if set to true.
 * @return A tensor containing image data.
 */
Tensor createTensorFromImage(const std::string& filename, DataType dtype, eDeviceType device, bool grayscale = false);

/**
 * @brief Compares tensor data with data within a vector.
 *
 * @tparam T The type for the Tensor/Vector data.
 * @param tensor The Tensor to compare.
 * @param expected_data A vector to compare tensor data with.
 * @param error_threshold Allowable absolute difference between data at
 * corresponding indices between the tensor and the vector.
 * @return A test status. Returns UNEXPECTED_VALUE if a comparison fails, or
 * SUCCESS on test success.
 */
template <typename T>
eTestStatusType compareArray(const Tensor& tensor, std::vector<T>& expected_data, float error_threshold) {
    // Flatten tensor data to a vector on the host (CPU)
    auto tensor_data = tensor.exportData<TensorDataStrided>();
    size_t tensor_data_size = tensor.shape().size() * tensor.dtype().size();
    std::vector<uint8_t> tensor_data_host(tensor.shape().size());

    switch (tensor.device()) {
        case eDeviceType::GPU: {
            hipMemcpy(tensor_data_host.data(), tensor_data.basePtr(), tensor_data_size, hipMemcpyDeviceToHost);
            break;
        }

        case eDeviceType::CPU: {
            memcpy(tensor_data_host.data(), tensor_data.basePtr(), tensor_data_size);
            break;
        }
    }

    // Compare data between tensor data and image
    for (int i = 0; i < expected_data.size(); i++) {
        T expected = static_cast<T>(expected_data[i]);
        T actual = static_cast<T>(tensor_data_host[i]);

        float difference = abs(actual - expected);
        if (difference > error_threshold) {
            fprintf(stderr, "Unexpected value at index %i: %i != %i\n", i, actual, expected);
            return eTestStatusType::UNEXPECTED_VALUE;
        }
    }

    return eTestStatusType::TEST_SUCCESS;
}

/**
 * @brief Compares tensor data to an image.
 *
 * @param tensor The tensor with image data.
 * @param filename A filename for the image.
 * @param error_threshold The amount of error allowed between two pixels during
 * comparison.
 * @return A test status. Returns UNEXPECTED_VALUE if a comparison fails, or
 * SUCCESS on test success.
 */
eTestStatusType compareImage(const Tensor& tensor, const std::string& filename, float error_threshold);

void writeTensor(const Tensor& tensor, const std::string& output_file);

template <typename T>
void copyData(const Tensor& input, const std::span<const T>& data, eDeviceType device) {
    auto tensor_data = input.exportData<TensorDataStrided>();

    switch (device) {
        case eDeviceType::GPU: {
            hipMemcpy(tensor_data.basePtr(), data.data(), data.size() * sizeof(T), hipMemcpyHostToDevice);
            break;
        }

        case eDeviceType::CPU: {
            memcpy(tensor_data.basePtr(), data.data(), data.size() * sizeof(T));
            break;
        }
    }
}
}  // namespace tests

}  // namespace roccv