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

#include "core/tensor.hpp"

namespace roccv {
template <typename T>
class GenericTensorWrapper {
   public:
    /**
     * @brief Constructs a generic tensor descriptor from a roccv::Tensor.
     *
     * @param tensor A roccv::Tensor to wrap.
     */
    GenericTensorWrapper(const Tensor& tensor) {
        auto tensorData = tensor.exportData<TensorDataStrided>();

        for (int i = 0; i < tensor.rank(); i++) {
            shape[i] = tensor.shape(i);
            strides[i] = tensorData.stride(i);
        }

        rank = tensor.rank();
        data = static_cast<unsigned char*>(tensorData.basePtr());
    }

    template <typename... ARGS>
    __device__ __host__ T& at(ARGS... idx) {
        int64_t coords[] = {idx...};
        size_t index = 0;

        for (int i = 0; i < rank; i++) {
            index += coords[i] * strides[i];
        }

        return *(reinterpret_cast<T*>(data + index));
    }

    std::array<int64_t, ROCCV_TENSOR_MAX_RANK> shape;
    std::array<int64_t, ROCCV_TENSOR_MAX_RANK> strides;
    int rank;
    unsigned char* data;
};
}  // namespace roccv