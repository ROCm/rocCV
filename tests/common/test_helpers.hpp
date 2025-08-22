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

#include <operator_types.h>

#include <cmath>
#include <core/detail/casting.hpp>
#include <core/exception.hpp>
#include <core/image_format.hpp>
#include <core/tensor.hpp>
#include <cstdlib>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <random>
#include <span>

namespace roccv {
namespace tests {

#define NEAR_EQUAL_THRESHOLD 1e-6
#define ERROR_PREFIX ("[" __FILE__ ":" + std::to_string(__LINE__) + "] ")

/**
 * @brief Ensures that a correct roccv::Exception is thrown from a call. If the call is successful or the thrown
 * Exception does not match, will throw a runtime error to be caught by the test suite.
 *
 * @param[in] call_ The function call to test.
 * @param[in] expected_status_ eStatusType expected to be thrown from the call.
 * @throws std::runtime_error if no exception is thrown or if the resulting exception status does not match the provided
 * status.
 */
#define EXPECT_EXCEPTION(call_, expected_status_)                                                                     \
    try {                                                                                                             \
        call_;                                                                                                        \
        throw std::runtime_error(ERROR_PREFIX + "Expected (" + ExceptionMessage::getMessageByEnum(expected_status_) + \
                                 ") but completed successfully instead.");                                            \
    } catch (const Exception& e) {                                                                                    \
        eStatusType received_ = e.getStatusEnum();                                                                    \
        if (received_ != expected_status_) {                                                                          \
            throw std::runtime_error(ERROR_PREFIX + "Expected (" +                                                    \
                                     ExceptionMessage::getMessageByEnum(expected_status_) + ") but received (" +      \
                                     ExceptionMessage::getMessageByEnum(received_) + ") instead.");                   \
        }                                                                                                             \
    }

#define EXPECT_TEST_STATUS(call_, expected_status_)                                                           \
    {                                                                                                         \
        eTestStatusType received_ = call_;                                                                    \
        if (received_ != expected_status_) {                                                                  \
            std::cerr << ERROR_PREFIX << "Expected (" << ExceptionMessage::getMessageByEnum(expected_status_) \
                      << ") but received (" << ExceptionMessage::getMessageByEnum(received_) << ") instead."  \
                      << std::endl;                                                                           \
            exit(1);                                                                                          \
        }                                                                                                     \
    }

/**
 * @brief A macro to be placed at the beginning of a collection of test cases. This will define the global status of all
 * proceeding TEST_CASE macros in order to keep track of the overall final result.
 *
 */
#define TEST_CASES_BEGIN() eTestStatusType _testSuiteStatus = eTestStatusType::TEST_SUCCESS

/**
 * @brief A macro to be placed at the end of a collection of test cases. Will return the final result of the collected
 * test cases.
 *
 */
#define TEST_CASES_END() return _testSuiteStatus;

/**
 * @brief Defines a test case for a call. This catches any error thrown by the call and marks the entire test suite as
 * failed if one is thrown. TEST_CASE macros must be preceeded by a TEST_CASES_BEGIN() macro and proceeded by a
 * TEST_CASES_END() macro in order to function properly.
 *
 * @param[in] call The function call to test.
 */
#define TEST_CASE(call)                                                                                             \
    {                                                                                                               \
        try {                                                                                                       \
            call;                                                                                                   \
        } catch (const std::exception& e) {                                                                         \
            std::cerr << "Test Failed: " << #call << "\n    Line: " << ERROR_PREFIX << "\n    Reason: " << e.what() \
                      << "\n\n";                                                                                    \
            _testSuiteStatus = eTestStatusType::TEST_FAILURE;                                                       \
        }                                                                                                           \
    }

/**
 * @brief Compares two values to ensure they are equal.
 *
 * @param[in] v1 Left side value.
 * @param[in] v2 Right side value.
 * @throws std::runtime_error if the values are not equal.
 */
#define EXPECT_EQ(v1, v2)                                                                                         \
    {                                                                                                             \
        if (v1 != v2)                                                                                             \
            throw std::runtime_error(ERROR_PREFIX +                                                               \
                                     "Expected the following values to be equal, but they are not: " #v1 +        \
                                     " == " #v2 + " (" + std::to_string(v1) + " == " + std::to_string(v2) + ")"); \
    }

/**
 * @brief Compares the values of two STL containers. Throws an error if the values within are not identical or if the
 * sizes do not match. This a macro version of CompareVectors which additionally prints the offending line number upon a
 * failure for better bug tracing capabilities.
 *
 * @param[in] actual Actual container values to compare against.
 * @param[in] expected Expected container values to compare against.
 * @throws std::runtime_error If the sizes of the containers differ or if any of the values within the containers
 * differ.
 *
 */
#define EXPECT_VECTOR_EQ(actual, expected)                                                                        \
    {                                                                                                             \
        if (actual.size() != expected.size()) {                                                                   \
            throw std::runtime_error(ERROR_PREFIX + "Vectors " + #actual + " (" + std::to_string(actual.size()) + \
                                     " elements) and " + #expected + "(" + std::to_string(expected.size()) +      \
                                     " elements) differ in size.");                                               \
        }                                                                                                         \
                                                                                                                  \
        for (int i = 0; i < actual.size(); i++) {                                                                 \
            if (actual[i] != expected[i]) {                                                                       \
                throw std::runtime_error(ERROR_PREFIX + "Value at index " + std::to_string(i) +                   \
                                         " does not match! Actual value: " + std::to_string(actual[i]) +          \
                                         ", Expected value: " + std::to_string(expected[i]));                     \
            }                                                                                                     \
        }                                                                                                         \
    }

/**
 * @brief Compares two values to ensure they are not equal.
 *
 * @param[in] v1 Left side value.
 * @param[in] v2 Right side value.
 * @throws std::runtime_error if the values are equal.
 */
#define EXPECT_NE(v1, v2)                                                                                         \
    {                                                                                                             \
        if (v1 == v2)                                                                                             \
            throw std::runtime_error(ERROR_PREFIX +                                                               \
                                     "Expected the following values to not be equal, but they were: " #v1 +       \
                                     " != " #v2 + " (" + std::to_string(v1) + " != " + std::to_string(v2) + ")"); \
    }

/**
 * @brief Tests whether a comparison is true.
 *
 * @param[in] comparison The comparison to test.
 * @throws std::runtime_error if the comparison is false.
 */
#define EXPECT_TRUE(comparison)                                                                                 \
    {                                                                                                           \
        if (!(comparison))                                                                                      \
            throw std::runtime_error(ERROR_PREFIX + #comparison + " expected to be true, but returned false."); \
    }

/**
 * @brief Tests whether a comparison is false.
 *
 * @param[in] comparison The comparison to test.
 * @throws std::runtime_error if the comparison returns true.
 */
#define EXPECT_FALSE(comparison)                                                                                \
    {                                                                                                           \
        if (comparison)                                                                                         \
            throw std::runtime_error(ERROR_PREFIX + #comparison + " expected to be false, but returned true."); \
    }

#define EXPECT_NO_ERRORS(call)                                                                                   \
    try {                                                                                                        \
        call;                                                                                                    \
    } catch (const roccv::Exception& e) {                                                                        \
        throw std::runtime_error(ERROR_PREFIX + #call +                                                          \
                                 ". Expected no exceptions, but received the following exception: " + e.what()); \
    }

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
    std::vector<uint8_t> tensor_data_host(tensor_data_size);

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

/**
 * @brief Fills a vector with random values based on a provided seed.
 *
 * @tparam T The underlying type of the vector data.
 * @param[out] vec The vector to fill with random data.
 * @param[in] seed A random seed. (Defaults to 12345)
 */
template <typename T>
void FillVector(std::vector<T>& vec, uint32_t seed = 12345) {
    // Create random number generator with seed and distribution.
    std::mt19937 eng(seed);

    // Select real distribution if T is a floating point, integer distribution otherwise
    if constexpr (std::is_floating_point_v<T>) {
        std::uniform_real_distribution<T> dist(0.0f, 1.0f);
        for (int i = 0; i < vec.size(); i++) {
            vec[i] = dist(eng);
        }
    } else {
        std::uniform_int_distribution<T> dist(std::numeric_limits<T>().min(), std::numeric_limits<T>().max());
        for (int i = 0; i < vec.size(); i++) {
            vec[i] = dist(eng);
        }
    }
}

/**
 * @brief Fills a vector with randomly generated 0s and 1s based on a provided seed.
 *
 * @tparam T The underlying type of the vector data.
 * @param[out] vec The vector to fill with random data.
 * @param[in] seed A random seed. (Defaults to 12345)
 */
template <typename T>
void FillVectorMask(std::vector<T>& vec, uint32_t seed = 12345) {
    // Create random number generator with seed and distribution.
    std::mt19937 eng(seed);

    std::uniform_int_distribution<T> dist(0, 1);
    for (int i = 0; i < vec.size(); i++) {
        vec[i] = dist(eng);
    }
}

/**
 * @brief Compares a vector to a reference vector.
 *
 * @tparam T The base type of the vector data.
 * @param result A vector containing data of the actual result.
 * @param ref A vector containing data of the reference to compare against.
 * @throws std::runtime_error if the result vector does not match with the reference vector.
 */
template <typename T>
void CompareVectors(const std::vector<T>& result, const std::vector<T>& ref) {
    if (result.size() != ref.size()) {
        throw std::runtime_error("Result output size (" + std::to_string(result.size()) +
                                 ") does not match reference size (" + std::to_string(ref.size()) + ")");
    }

    for (size_t i = 0; i < ref.size(); ++i) {
        if (result[i] != ref[i]) {
            // Additional handling in case the datatype of T is uint8_t. Must be casted to int, otherwise the character
            // rather than the raw value will be printed.
            if constexpr (std::is_integral_v<T>) {
                throw std::runtime_error("Value at index " + std::to_string(i) + " does not match! Actual value: " +
                                         std::to_string(static_cast<int>(result[i])) +
                                         ", Expected value: " + std::to_string(static_cast<int>(ref[i])));
            } else {
                throw std::runtime_error("Value at index " + std::to_string(i) + " does not match! Actual value: " +
                                         std::to_string(result[i]) + ", Expected value: " + std::to_string(ref[i]));
            }
        }
    }
}

/**
 * @brief Compares values from two vectors, allowing a difference between values of at most delta.
 *
 * @tparam T Datatype of the underlying vector data.
 * @param result Vector containing actual results.
 * @param ref Reference vector to compare against.
 * @param delta The allowable error between two values in normalized [0-1.0] range.
 * @throws std::runtime_error if difference between vector values exceeds the given delta.
 */
template <typename T>
void CompareVectorsNear(const std::vector<T>& result, const std::vector<T>& ref, double delta = NEAR_EQUAL_THRESHOLD) {
    if (result.size() != ref.size()) {
        throw std::runtime_error("Result output size (" + std::to_string(result.size()) +
                                 ") does not match reference size (" + std::to_string(ref.size()) + ")");
    }

    for (size_t i = 0; i < ref.size(); ++i) {
        // Compute the absolute difference between reference and result vector values.
        T error = result[i] < ref[i] ? ref[i] - result[i] : result[i] - ref[i];
        T thresh = detail::RangeCast<T>(delta);
        // Clamp error threshold to at least 1 if the delta ends up being 0 after the RangeCast. This is to ensure that
        // integers which get rounded down to 0 still have some sort of proper error threshold to compare against.
        thresh = thresh == 0 ? 1 : thresh;

        if (error > thresh) {
            std::stringstream errorMsg;
            // Additional handling in case the datatype of T is uint8_t. Must be casted to int, otherwise the character
            // rather than the raw value will be printed.
            if constexpr (std::is_same_v<T, unsigned char> || std::is_same_v<T, signed char>) {
                errorMsg << "Value at index " << i << " does not match! Actual value: " << static_cast<int>(result[i])
                         << " Expected value: " << static_cast<int>(ref[i]) << ". Error: " << static_cast<int>(error)
                         << " > " << static_cast<int>(thresh);
            } else {
                errorMsg << "Value at index " << i << " does not match! Actual value: " << result[i]
                         << " Expected value: " << ref[i] << ". Error: " << error << " > " << thresh;
            }
            throw std::runtime_error(errorMsg.str());
        }
    }
}

/**
 * @brief Copies vector data into a roccv::Tensor. This will copy vector data into either GPU memory or CPU memory,
 * depending on the device specified in the roccv::Tensor's metadata.
 *
 * @tparam T The base datatype of the underlying data.
 * @param dst The destination roccv::Tensor to copy data into.
 * @param src A source vector containing data.
 * @throws std::runtime_error if the size of the dst and src do not match.
 */
template <typename T>
void CopyVectorIntoTensor(const Tensor& dst, std::vector<T>& src) {
    auto tensorData = dst.exportData<TensorDataStrided>();
    size_t dataSize = dst.shape().size() * dst.dtype().size();

    // Ensure source and destination have the same amount of memory allocated.
    if (dataSize != src.size() * sizeof(T)) {
        throw std::runtime_error(
            "Cannot copy source vector into destination tensor. Size of src vector and destination tensor do not "
            "match.");
    }

    switch (dst.device()) {
        case eDeviceType::GPU: {
            HIP_VALIDATE_NO_ERRORS(
                hipMemcpy(tensorData.basePtr(), src.data(), dataSize, hipMemcpyKind::hipMemcpyHostToDevice));
            break;
        }

        case eDeviceType::CPU: {
            HIP_VALIDATE_NO_ERRORS(
                hipMemcpy(tensorData.basePtr(), src.data(), dataSize, hipMemcpyKind::hipMemcpyHostToHost));
            break;
        }
    }
}

/**
 * @brief Copies roccv::Tensor data into a destination vector.
 *
 * @tparam T The base datatype of the underlying tensor data.
 * @param dst The destination vector which the data will be copied into.
 * @param src The roccv::Tensor containing the source data.
 * @throws std::runtime_error if the size of src and dst do not match.
 */
template <typename T>
void CopyTensorIntoVector(std::vector<T>& dst, const Tensor& src) {
    size_t size = src.shape().size() * src.dtype().size();
    auto tensorData = src.exportData<TensorDataStrided>();

    if (size != dst.size() * sizeof(T)) {
        throw std::runtime_error(
            "Cannot copy source tensor data into destination vector. Size of destination vector and source tensor do "
            "not match.");
    }

    switch (src.device()) {
        case eDeviceType::GPU: {
            HIP_VALIDATE_NO_ERRORS(
                hipMemcpy(dst.data(), tensorData.basePtr(), size, hipMemcpyKind::hipMemcpyDeviceToHost));
            break;
        }

        case eDeviceType::CPU: {
            HIP_VALIDATE_NO_ERRORS(
                hipMemcpy(dst.data(), tensorData.basePtr(), size, hipMemcpyKind::hipMemcpyHostToHost));
            break;
        }
    }
}

}  // namespace tests
}  // namespace roccv