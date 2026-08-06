/**
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

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

#include <atomic>
#include <cfenv>
#include <cmath>
#include <core/detail/casting.hpp>
#include <core/detail/type_traits.hpp>
#include <core/detail/vector_utils.hpp>
#include <core/wrappers/border_wrapper.hpp>
#include <core/wrappers/image_wrapper.hpp>
#include <op_average_blur.hpp>
#include <thread>

#include "test_helpers.hpp"

using namespace roccv;
using namespace roccv::detail;
using namespace roccv::tests;

namespace {

/**
 * @brief Golden model for the AverageBlur operation.
 *
 * @tparam T Vectorized datatype of the image's pixels.
 * @tparam BorderMode Border pixel extrapolation method.
 * @tparam BT Base type of the image's data.
 * @param[in] input Input tensor containing image data.
 * @param[in] batchSize The number of images in the batch.
 * @param[in] width Image width.
 * @param[in] height Image height.
 * @param[in] kernelWidth Kernel width.
 * @param[in] kernelHeight Kernel height.
 * @param[in] anchorX Kernel anchor in X direction.
 * @param[in] anchorY Kernel anchor in Y direction.
 * @return Vector containing the results of the operation.
 */
template <typename T, eBorderType BorderMode, typename BT = detail::BaseType<T>>
std::vector<BT> GenerateGoldenAverageBlur(std::vector<BT>& input, int32_t batchSize, int32_t width, int32_t height,
                                          int kernelWidth, int kernelHeight, int anchorX, int anchorY) {
    std::vector<BT> output(input.size());
    BorderWrapper<T, BorderMode> src(ImageWrapper<T>(input, batchSize, width, height), SetAll<T>(0));
    ImageWrapper<T> dst(output, batchSize, width, height);

    using namespace roccv::detail;
    using worktype = MakeType<float, NumElements<T>>;

    for (int b = 0; b < dst.batches(); b++) {
        for (int j = 0; j < dst.height(); j++) {
            for (int i = 0; i < dst.width(); i++) {
                worktype numerators = SetAll<worktype>(0.0f);
                float denominator = 0.0f;

                for (int y = j - anchorY; y <= j + (kernelHeight - 1 - anchorY); y++) {
                    for (int x = i - anchorX; x <= i + (kernelWidth - 1 - anchorX); x++) {
                        worktype workPixel = StaticCast<worktype>(src.at(b, y, x, 0));

                        denominator += 1.0f;
                        numerators += workPixel;
                    }
                }
                dst.at(b, j, i, 0) = SaturateCast<T>(numerators / denominator);
            }
        }
    }
    return output;
}

/**
 * @brief Tests correctness of the AverageBlur operator, comparing it against a generated golden result.
 *
 * @tparam T Underlying datatype of the image's pixels.
 * @tparam BorderMode Border pixel extrapolation method.
 * @tparam BT Base type of the image's data.
 * @param[in] batchSize Number of images in the batch.
 * @param[in] width Width of each image in the batch.
 * @param[in] height Height of each image in the batch.
 * @param[in] format Image format.
 * @param[in] kernelWidth Kernel width.
 * @param[in] kernelHeight Kernel height.
 * @param[in] anchorX Kernel anchor in X direction.
 * @param[in] anchorY Kernel anchor in Y direction.
 * @param[in] device Device this correctness test should be run on.
 */
template <typename T, eBorderType BorderMode, typename BT = detail::BaseType<T>>
void TestCorrectness(int batchSize, int width, int height, ImageFormat format, int kernelWidth, int kernelHeight,
                     int anchorX, int anchorY, eDeviceType device) {
    // Create input and output tensor based on test parameters
    Tensor input(batchSize, {width, height}, format, device);
    Tensor output(batchSize, {width, height}, format, device);

    // Create a vector and fill it with random data.
    std::vector<BT> inputData(input.shape().size());
    FillVector(inputData);
    if constexpr (std::is_floating_point_v<BT>) {
        for (size_t i = 0; i < inputData.size(); i++) {
            inputData[i] *= static_cast<BT>(std::numeric_limits<ushort>::max());
        }
    }

    // Copy generated input data into input tensor
    CopyVectorIntoTensor(input, inputData);

    // Infer anchor if (-1,-1)
    int effectiveAnchorX = (anchorX == -1) ? kernelWidth >> 1 : anchorX;
    int effectiveAnchorY = (anchorY == -1) ? kernelHeight >> 1 : anchorY;

    hipStream_t stream;
    HIP_VALIDATE_NO_ERRORS(hipStreamCreate(&stream));
    AverageBlur op(kernelWidth, kernelHeight);
    op(stream, input, output, kernelWidth, kernelHeight, anchorX, anchorY, BorderMode, device);
    HIP_VALIDATE_NO_ERRORS(hipStreamSynchronize(stream));
    HIP_VALIDATE_NO_ERRORS(hipStreamDestroy(stream));

    // Copy data from output tensor into a host allocated vector
    std::vector<BT> outputData(output.shape().size());
    CopyTensorIntoVector(outputData, output);

    // Calculate golden reference
    std::vector<BT> ref = GenerateGoldenAverageBlur<T, BorderMode>(inputData, batchSize, width, height, kernelWidth,
                                                                   kernelHeight, effectiveAnchorX, effectiveAnchorY);

    // Compare data in actual output versus the generated golden reference image
    // TODO check on delta, this matches bilateral filter and passes (looser does not)
    CompareVectorsNear(outputData, ref, 1);
}

/**
 * @brief Tests correctness of the AverageBlur operator when multiple threads concurrently use the same operator.
 * @tparam T Underlying datatype of the image's pixels.
 * @tparam BorderMode Border pixel extrapolation method.
 * @tparam BT Base type of the image's data.
 * @param[in] batchSize Number of images in the batch.
 * @param[in] width Width of each image in the batch.
 * @param[in] height Height of each image in the batch.
 * @param[in] format Image format.
 * @param[in] device Device this correctness test should be run on.
 */
template <typename T, eBorderType BorderMode, typename BT = detail::BaseType<T>>
void TestCorrectnessConcurrent(int batchSize, int width, int height, ImageFormat format, eDeviceType device) {
    constexpr int NUM_THREADS = 8;
    constexpr int TOTAL_TESTS = 80;

    struct ThreadTest {
        Tensor input;
        Tensor output;
        std::vector<BT> inputData;
        hipStream_t stream;
        int ksize;

        ThreadTest(int b, int w, int h, ImageFormat fmt, eDeviceType dev, int k, int id)
            : input(b, {w, h}, fmt, dev), output(b, {w, h}, fmt, dev), inputData(input.shape().size()), ksize(k) {
            HIP_VALIDATE_NO_ERRORS(hipStreamCreate(&stream));
            FillVector(inputData, id * 1000);
            CopyVectorIntoTensor(input, inputData);
        }

        ~ThreadTest() { (void)hipStreamDestroy(stream); }
    };

    std::vector<std::unique_ptr<ThreadTest>> threadTests;
    threadTests.reserve(TOTAL_TESTS);
    // each thread has different kernel sizes so different results
    const int kernelSizes[] = {3, 5, 7, 9, 11, 13, 15, 17};
    for (int i = 0; i < TOTAL_TESTS; ++i) {
        int kernelSize = kernelSizes[i % 8];
        threadTests.push_back(std::make_unique<ThreadTest>(batchSize, width, height, format, device, kernelSize, i));
    }

    AverageBlur op(17, 17);  // shared op for all threads
    std::vector<std::thread> threads;
    std::atomic<int> nextTestIndex{0};

    auto threadFunc = [&]() {
        while (true) {
            int idx = nextTestIndex.fetch_add(1);
            if (idx >= TOTAL_TESTS) break;
            ThreadTest* test = threadTests[idx].get();
            // All threads call the same operator instance concurrently
            op(test->stream, test->input, test->output, test->ksize, test->ksize, -1, -1, BorderMode, device);
        }
    };

    for (int i = 0; i < NUM_THREADS; i++) {
        threads.emplace_back(threadFunc);
    }
    for (auto& thread : threads) {
        thread.join();
    }
    for (auto& threadTest : threadTests) {
        HIP_VALIDATE_NO_ERRORS(hipStreamSynchronize(threadTest->stream));
    }

    for (auto& threadTest : threadTests) {
        std::vector<BT> outputData(threadTest->output.shape().size());
        CopyTensorIntoVector(outputData, threadTest->output);

        int anchor = threadTest->ksize >> 1;
        std::vector<BT> ref = GenerateGoldenAverageBlur<T, BorderMode>(
            threadTest->inputData, batchSize, width, height, threadTest->ksize, threadTest->ksize, anchor, anchor);

        CompareVectorsNear(outputData, ref, 1);
    }
}

/**
 * @brief Tests correctness of the AverageBlur operator when multiple threads concurrently use the same operator, but on
 * different devices. This tests that the synchronization correctly prevents the CPU path from modifying m_hostKernelMem
 * while the GPU is still asynchronously copying from it.
 * @tparam T Underlying datatype of the image's pixels.
 * @tparam BorderMode Border pixel extrapolation method.
 * @tparam BT Base type of the image's data.
 * @param[in] batchSize Number of images in the batch.
 * @param[in] width Width of each image in the batch.
 * @param[in] height Height of each image in the batch.
 * @param[in] format Image format.
 */
template <typename T, eBorderType BorderMode, typename BT = detail::BaseType<T>>
void TestCorrectnessConcurrentBothDevices(int batchSize, int width, int height, ImageFormat format) {
    constexpr int NUM_ITERATIONS = 100;
    constexpr int NUM_THREADS = 2;

    struct DeviceTest {
        Tensor input;
        Tensor output;
        std::vector<BT> inputData;
        hipStream_t stream;
        int ksize;
        eDeviceType device;

        DeviceTest(int b, int w, int h, ImageFormat fmt, eDeviceType dev, int k, int id)
            : input(b, {w, h}, fmt, dev),
              output(b, {w, h}, fmt, dev),
              inputData(input.shape().size()),
              ksize(k),
              device(dev) {
            HIP_VALIDATE_NO_ERRORS(hipStreamCreate(&stream));
            FillVector(inputData, id * 1000);
            CopyVectorIntoTensor(input, inputData);
        }

        ~DeviceTest() { (void)hipStreamDestroy(stream); }
    };

    std::vector<std::unique_ptr<DeviceTest>> tests;
    tests.reserve(NUM_ITERATIONS * NUM_THREADS);

    // Create alternating GPU and CPU tests with different kernel sizes
    for (int i = 0; i < NUM_ITERATIONS; ++i) {
        int gpuKernel = 3 + (i % 4) * 2;
        int cpuKernel = 5 + (i % 3) * 2;
        tests.push_back(
            std::make_unique<DeviceTest>(batchSize, width, height, format, eDeviceType::GPU, gpuKernel, i * 2));
        tests.push_back(
            std::make_unique<DeviceTest>(batchSize, width, height, format, eDeviceType::CPU, cpuKernel, i * 2 + 1));
    }

    AverageBlur op(11, 11);
    std::atomic<int> nextTestIndex{0};

    auto threadFunc = [&]() {
        while (true) {
            int idx = nextTestIndex.fetch_add(1);
            if (idx >= static_cast<int>(tests.size())) break;
            DeviceTest* test = tests[idx].get();
            op(test->stream, test->input, test->output, test->ksize, test->ksize, -1, -1, BorderMode, test->device);
        }
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < NUM_THREADS; i++) {
        threads.emplace_back(threadFunc);
    }
    for (auto& thread : threads) {
        thread.join();
    }

    for (auto& test : tests) {
        HIP_VALIDATE_NO_ERRORS(hipStreamSynchronize(test->stream));
    }

    // Verify all results
    for (auto& test : tests) {
        std::vector<BT> outputData(test->output.shape().size());
        CopyTensorIntoVector(outputData, test->output);

        int anchor = test->ksize >> 1;
        std::vector<BT> ref = GenerateGoldenAverageBlur<T, BorderMode>(test->inputData, batchSize, width, height,
                                                                       test->ksize, test->ksize, anchor, anchor);

        CompareVectorsNear(outputData, ref, 1);
    }
}

void TestNegativeAverageBlur() {
    TensorShape validShape(TensorLayout(eTensorLayout::TENSOR_LAYOUT_NHWC), {1, 1, 1, 1});
    Tensor validGPUTensor(validShape, DataType(eDataType::DATA_TYPE_U8), eDeviceType::GPU);
    Tensor validCPUTensor(validShape, DataType(eDataType::DATA_TYPE_U8), eDeviceType::CPU);

    AverageBlur op(3, 3);

    {
        // Test wrong device
        EXPECT_EXCEPTION(
            op(nullptr, validCPUTensor, validGPUTensor, 3, 3, 1, 1, BORDER_TYPE_CONSTANT, eDeviceType::GPU),
            eStatusType::INVALID_OPERATION);
        EXPECT_EXCEPTION(
            op(nullptr, validGPUTensor, validCPUTensor, 3, 3, 1, 1, BORDER_TYPE_CONSTANT, eDeviceType::GPU),
            eStatusType::INVALID_COMBINATION);
    }

    {
        // Test unsupported/mismatch input/output data type
        Tensor invalidTensor(validGPUTensor.shape(), DataType(eDataType::DATA_TYPE_U32), eDeviceType::GPU);
        EXPECT_EXCEPTION(
            op(nullptr, invalidTensor, validGPUTensor, 3, 3, -1, -1, BORDER_TYPE_REFLECT, eDeviceType::GPU),
            eStatusType::NOT_IMPLEMENTED);
        EXPECT_EXCEPTION(
            op(nullptr, validCPUTensor, invalidTensor, 3, 3, -1, -1, BORDER_TYPE_REFLECT, eDeviceType::CPU),
            eStatusType::INVALID_COMBINATION);
        Tensor validGPUS16Tensor(validShape, DataType(eDataType::DATA_TYPE_S16), eDeviceType::GPU);
        EXPECT_EXCEPTION(
            op(nullptr, validGPUTensor, validGPUS16Tensor, 3, 3, -1, -1, BORDER_TYPE_REFLECT, eDeviceType::GPU),
            eStatusType::INVALID_COMBINATION);
    }

    {
        // Test unsupported input/output layout
        TensorShape invalidLayoutShape(TensorLayout(eTensorLayout::TENSOR_LAYOUT_NC), {1, 1});
        Tensor invalidTensor(invalidLayoutShape, DataType(eDataType::DATA_TYPE_U8), eDeviceType::GPU);
        EXPECT_EXCEPTION(op(nullptr, invalidTensor, validGPUTensor, 3, 3, -1, -1, BORDER_TYPE_WRAP, eDeviceType::GPU),
                         eStatusType::INVALID_COMBINATION);
        EXPECT_EXCEPTION(op(nullptr, validGPUTensor, invalidTensor, 3, 3, -1, -1, BORDER_TYPE_WRAP, eDeviceType::GPU),
                         eStatusType::INVALID_COMBINATION);
    }

    {
        // Test input/output shape mismatch
        Tensor invalidTensor(TensorShape(validGPUTensor.layout(), {2, 2, 2, 2}), DataType(eDataType::DATA_TYPE_U8),
                             eDeviceType::GPU);
        EXPECT_EXCEPTION(
            op(nullptr, invalidTensor, validGPUTensor, 3, 3, -1, -1, BORDER_TYPE_REPLICATE, eDeviceType::GPU),
            eStatusType::INVALID_COMBINATION);
    }

    {
        // Test bad op construction (bad max ksize)
        EXPECT_EXCEPTION(AverageBlur opBadW(0, 3), eStatusType::INVALID_VALUE);
        EXPECT_EXCEPTION(AverageBlur opBadH(3, -3), eStatusType::INVALID_VALUE);
    }

    {
        // Test bad kernel size
        // exceeding max
        EXPECT_EXCEPTION(
            op(nullptr, validGPUTensor, validGPUTensor, 3, 5, -1, -1, BORDER_TYPE_CONSTANT, eDeviceType::GPU),
            eStatusType::INVALID_VALUE);
        EXPECT_EXCEPTION(
            op(nullptr, validGPUTensor, validGPUTensor, 5, 3, -1, -1, BORDER_TYPE_CONSTANT, eDeviceType::GPU),
            eStatusType::INVALID_VALUE);
        // not odd
        EXPECT_EXCEPTION(
            op(nullptr, validGPUTensor, validGPUTensor, 1, 2, -1, -1, BORDER_TYPE_CONSTANT, eDeviceType::GPU),
            eStatusType::INVALID_VALUE);
        EXPECT_EXCEPTION(
            op(nullptr, validGPUTensor, validGPUTensor, 2, 1, -1, -1, BORDER_TYPE_CONSTANT, eDeviceType::GPU),
            eStatusType::INVALID_VALUE);
        // not positive
        EXPECT_EXCEPTION(
            op(nullptr, validGPUTensor, validGPUTensor, 0, 2, -1, -1, BORDER_TYPE_CONSTANT, eDeviceType::GPU),
            eStatusType::INVALID_VALUE);
        EXPECT_EXCEPTION(
            op(nullptr, validGPUTensor, validGPUTensor, 2, -1, -1, -1, BORDER_TYPE_CONSTANT, eDeviceType::GPU),
            eStatusType::INVALID_VALUE);
    }

    {
        // Test bad anchor
        // >= ksize
        EXPECT_EXCEPTION(
            op(nullptr, validGPUTensor, validGPUTensor, 3, 3, 4, -1, BORDER_TYPE_CONSTANT, eDeviceType::GPU),
            eStatusType::INVALID_VALUE);
        EXPECT_EXCEPTION(
            op(nullptr, validGPUTensor, validGPUTensor, 3, 3, -1, 3, BORDER_TYPE_CONSTANT, eDeviceType::GPU),
            eStatusType::INVALID_VALUE);
        // Negative and not -1
        EXPECT_EXCEPTION(
            op(nullptr, validGPUTensor, validGPUTensor, 3, 3, 2, -2, BORDER_TYPE_CONSTANT, eDeviceType::GPU),
            eStatusType::INVALID_VALUE);
        EXPECT_EXCEPTION(
            op(nullptr, validGPUTensor, validGPUTensor, 3, 3, -2, 2, BORDER_TYPE_CONSTANT, eDeviceType::GPU),
            eStatusType::INVALID_VALUE);
    }
}
}  // namespace

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;
    TEST_CASES_BEGIN();

    // Test negative operator cases
    TEST_CASE(TestNegativeAverageBlur());

    // Test concurrency control
    TEST_CASE((TestCorrectnessConcurrent<float1, BORDER_TYPE_REFLECT>(1, 64, 64, FMT_F32, eDeviceType::GPU)));
    TEST_CASE((TestCorrectnessConcurrent<float1, BORDER_TYPE_REFLECT>(1, 64, 64, FMT_F32, eDeviceType::CPU)));
    TEST_CASE((TestCorrectnessConcurrentBothDevices<float1, BORDER_TYPE_REFLECT>(1, 64, 64, FMT_F32)));

    // GPU correctness tests
    TEST_CASE((TestCorrectness<uchar1, BORDER_TYPE_CONSTANT>(1, 20, 20, FMT_U8, 3, 3, -1, -1, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<ushort1, BORDER_TYPE_WRAP>(1, 20, 20, FMT_U16, 3, 3, 0, 0, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<ushort1, BORDER_TYPE_REFLECT>(2, 20, 20, FMT_U16, 5, 5, 2, 2, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<short1, BORDER_TYPE_REPLICATE>(1, 20, 20, FMT_S16, 3, 3, 1, 1, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<int1, BORDER_TYPE_CONSTANT>(1, 32, 32, FMT_S32, 5, 3, 4, 0, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<int1, BORDER_TYPE_WRAP>(2, 32, 32, FMT_S32, 3, 3, -1, -1, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<float1, BORDER_TYPE_REPLICATE>(1, 24, 24, FMT_F32, 7, 7, 3, 6, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<float1, BORDER_TYPE_WRAP>(2, 24, 24, FMT_F32, 5, 5, 0, 0, eDeviceType::GPU)));

    TEST_CASE((TestCorrectness<uchar3, BORDER_TYPE_REPLICATE>(2, 20, 20, FMT_RGB8, 5, 5, -1, -1, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<uchar3, BORDER_TYPE_CONSTANT>(1, 20, 20, FMT_RGB8, 3, 3, 2, 2, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<ushort3, BORDER_TYPE_CONSTANT>(1, 20, 20, FMT_RGB16, 3, 3, 0, 1, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<short3, BORDER_TYPE_CONSTANT>(1, 20, 20, FMT_RGBs16, 7, 3, 6, 1, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<short3, BORDER_TYPE_REFLECT101>(2, 20, 20, FMT_RGBs16, 3, 3, 1, 0, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<int3, BORDER_TYPE_REPLICATE>(1, 24, 24, FMT_RGBs32, 5, 5, 2, 4, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<float3, BORDER_TYPE_WRAP>(2, 24, 24, FMT_RGBf32, 3, 3, -1, -1, eDeviceType::GPU)));

    TEST_CASE((TestCorrectness<uchar4, BORDER_TYPE_WRAP>(1, 10, 10, FMT_RGBA8, 3, 3, 0, 2, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<uchar4, BORDER_TYPE_REPLICATE>(5, 64, 64, FMT_RGBA8, 7, 7, 3, 3, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<ushort4, BORDER_TYPE_REFLECT>(1, 20, 20, FMT_RGBA16, 3, 3, -1, -1, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<ushort4, BORDER_TYPE_WRAP>(2, 20, 20, FMT_RGBA16, 5, 3, 2, 0, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<short4, BORDER_TYPE_REFLECT101>(2, 20, 20, FMT_RGBAs16, 3, 3, 1, 1, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<int4, BORDER_TYPE_REPLICATE>(1, 24, 24, FMT_RGBAs32, 5, 5, 0, 4, eDeviceType::GPU)));
    TEST_CASE((TestCorrectness<float4, BORDER_TYPE_WRAP>(2, 24, 24, FMT_RGBAf32, 5, 3, 2, 1, eDeviceType::GPU)));

    // CPU correctness tests
    TEST_CASE((TestCorrectness<uchar1, BORDER_TYPE_CONSTANT>(1, 20, 20, FMT_U8, 3, 3, -1, -1, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<ushort1, BORDER_TYPE_WRAP>(1, 20, 20, FMT_U16, 3, 3, 0, 0, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<ushort1, BORDER_TYPE_REFLECT>(2, 20, 20, FMT_U16, 5, 5, 2, 2, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<short1, BORDER_TYPE_REPLICATE>(1, 20, 20, FMT_S16, 3, 3, 1, 1, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<int1, BORDER_TYPE_CONSTANT>(1, 32, 32, FMT_S32, 5, 3, 4, 0, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<int1, BORDER_TYPE_WRAP>(2, 32, 32, FMT_S32, 3, 3, -1, -1, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<float1, BORDER_TYPE_REPLICATE>(1, 24, 24, FMT_F32, 7, 7, 3, 6, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<float1, BORDER_TYPE_WRAP>(2, 24, 24, FMT_F32, 5, 5, 0, 0, eDeviceType::CPU)));

    TEST_CASE((TestCorrectness<uchar3, BORDER_TYPE_REPLICATE>(2, 20, 20, FMT_RGB8, 5, 5, -1, -1, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<uchar3, BORDER_TYPE_CONSTANT>(1, 20, 20, FMT_RGB8, 3, 3, 2, 2, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<ushort3, BORDER_TYPE_CONSTANT>(1, 20, 20, FMT_RGB16, 3, 3, 0, 1, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<short3, BORDER_TYPE_CONSTANT>(1, 20, 20, FMT_RGBs16, 7, 3, 6, 1, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<short3, BORDER_TYPE_REFLECT101>(2, 20, 20, FMT_RGBs16, 3, 3, 1, 0, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<int3, BORDER_TYPE_REPLICATE>(1, 24, 24, FMT_RGBs32, 5, 5, 2, 4, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<float3, BORDER_TYPE_WRAP>(2, 24, 24, FMT_RGBf32, 3, 3, -1, -1, eDeviceType::CPU)));

    TEST_CASE((TestCorrectness<uchar4, BORDER_TYPE_WRAP>(1, 10, 10, FMT_RGBA8, 3, 3, 0, 2, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<uchar4, BORDER_TYPE_REPLICATE>(5, 64, 64, FMT_RGBA8, 7, 7, 3, 3, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<ushort4, BORDER_TYPE_REFLECT>(1, 20, 20, FMT_RGBA16, 3, 3, -1, -1, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<ushort4, BORDER_TYPE_WRAP>(2, 20, 20, FMT_RGBA16, 5, 3, 2, 0, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<short4, BORDER_TYPE_REFLECT101>(2, 20, 20, FMT_RGBAs16, 3, 3, 1, 1, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<int4, BORDER_TYPE_REPLICATE>(1, 24, 24, FMT_RGBAs32, 5, 5, 0, 4, eDeviceType::CPU)));
    TEST_CASE((TestCorrectness<float4, BORDER_TYPE_WRAP>(2, 24, 24, FMT_RGBAf32, 5, 3, 2, 1, eDeviceType::CPU)));

    TEST_CASES_END();
}
