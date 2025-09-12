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

#pragma once

#include <hip/hip_runtime.h>

#include <vector>

#include "core/detail/type_traits.hpp"
#include "core/tensor.hpp"

namespace roccv {

/**
 * @brief ImageWrapper is a non-owning wrapper for roccv::Tensors with a NHWC/NCHW/HWC layout. It provides
 * methods for accessing the underlying data within HIP kernels.
 *
 * @tparam T The datatype of the underlying tensor data.
 */
template <typename T>
class ImageWrapper {
   public:
    using ValueType = T;
    using BaseType = detail::BaseType<T>;

    /**
     * @brief Creates an ImageWrapper from a Tensor.
     *
     * @param tensor The Tensor to be represented by the ImageWrapper.
     */
    ImageWrapper(const Tensor& tensor) {
        if (tensor.layout() != eTensorLayout::TENSOR_LAYOUT_NHWC &&
            tensor.layout() != eTensorLayout::TENSOR_LAYOUT_NCHW &&
            tensor.layout() != eTensorLayout::TENSOR_LAYOUT_HWC) {
            throw Exception("The given tensor layout is not supported for ImageWrapper", eStatusType::NOT_IMPLEMENTED);
        }

        // Copy tensor data into image tensor descriptor
        TensorDataStrided tdata = tensor.exportData<TensorDataStrided>();
        ImageShape indexes = {tensor.layout().batch_index(), tensor.layout().height_index(),
                              tensor.layout().width_index(), tensor.layout().channels_index()};

        // Handle HWC layout, which doesn't have shapes/strides for the batch dimension. We set the batch shape to 1 and
        // the strides to 0.
        int64_t num_batches = indexes.n < 0 ? 1 : tensor.shape(indexes.n);
        int64_t batch_stride = indexes.n < 0 ? 0 : tdata.stride(indexes.n);

        shape = {num_batches, tdata.shape(indexes.h), tdata.shape(indexes.w), tdata.shape(indexes.c)};
        stride = {batch_stride, tdata.stride(indexes.h), tdata.stride(indexes.w), tdata.stride(indexes.c)};
        data = static_cast<unsigned char*>(tdata.basePtr());
    }

    /**
     * @brief Creates an ImageWrapper from a vector.
     *
     * @param input The input vector to wrap.
     * @param batchSize The number of images within the batch.
     * @param width The width of each image within the batch.
     * @param height The height of each image within the batch.
     */
    ImageWrapper(std::vector<BaseType>& input, int32_t batchSize, int32_t width, int32_t height) {
        // Calculate strides based on input (byte-wise strides)
        stride.c = sizeof(BaseType);
        stride.w = stride.c * detail::NumElements<T>;
        stride.h = stride.w * width;
        stride.n = stride.h * height;

        // Copy shape information
        shape.c = detail::NumElements<T>;
        shape.w = width;
        shape.h = height;
        shape.n = batchSize;

        // Copy data pointer from input vector
        data = reinterpret_cast<unsigned char*>(input.data());
    }

    /**
     * @brief Creates an ImageWrapper from a pointer.
     *
     * @param input The input pointer to wrap.
     * @param batchSize The number of images within the batch.
     * @param width The width of each image within the batch.
     * @param height The height of each image within the batch.
     */
    ImageWrapper(void* input, int32_t batchSize, int32_t width, int32_t height) {
        // Calculate strides based on input (byte-wise strides)
        stride.c = sizeof(BaseType);
        stride.w = stride.c * detail::NumElements<T>;
        stride.h = stride.w * width;
        stride.n = stride.h * height;

        // Copy shape information
        shape.c = detail::NumElements<T>;
        shape.w = width;
        shape.h = height;
        shape.n = batchSize;

        data = reinterpret_cast<unsigned char*>(input);
    }

    /**
     * @brief Returns a reference to data given coordinates within an image tensor.
     *
     * @param n Batch coordinates.
     * @param h Height coordinates.
     * @param w Width coordinates.
     * @param c Channel coordinates.
     * @return A reference to the underlying data at given coordinates.
     */
    __device__ __host__ T& at(int64_t n, int64_t h, int64_t w, int64_t c) {
        return *(reinterpret_cast<T*>(data + (stride.n * n) + (stride.h * h) + (stride.w * w) + (stride.c * c)));
    }

    __device__ __host__ const T at(int64_t n, int64_t h, int64_t w, int64_t c) const {
        return *(reinterpret_cast<T*>(data + (stride.n * n) + (stride.h * h) + (stride.w * w) + (stride.c * c)));
    }

    /**
     * @brief Retrives the height of the images.
     *
     * @return Image height.
     */
    __device__ __host__ inline int64_t height() const { return shape.h; }

    /**
     * @brief Retrieves the width of the image.
     *
     * @return Image width.
     */
    __device__ __host__ inline int64_t width() const { return shape.w; }

    /**
     * @brief Retrieves the number of batches in the image tensor.
     *
     * @return Number of batches.
     */
    __device__ __host__ inline int64_t batches() const { return shape.n; }

    /**
     * @brief Retries the number of channels in the image.
     *
     * @return Image channels.
     */
    __device__ __host__ inline int64_t channels() const { return shape.c; }

   private:
    struct ImageShape {
        int64_t n, h, w, c;
    };

    ImageShape shape;

    /**
     * @brief Describes the number of bytes to move in order to access the next index of the shape.
     *
     * stride.n: Number of bytes to move to the next image in the batch.
     * stride.h: Number of bytes to access the next row in the image.
     * stride.w: Number of bytes to access the next pixel in the image.
     * stride.c: Number of bytes to access the next channel in a pixel.
     */
    ImageShape stride;

    unsigned char* data;
};
}  // namespace roccv