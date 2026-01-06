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

#include "roccv_bench_helpers.hpp"

#include <core/hip_assert.h>

#include <roccvbench/utils.hpp>
#include <vector>

struct MemcpyParams {
    void* basePtr = nullptr;  // Base pointer to the tensor data
    size_t rowPitch = 0;      // Number of bytes per row, including padding
    size_t rowBytes = 0;      // Number of bytes per row, not including padding
    size_t imageBytes = 0;    // Number of bytes per image, including padding (rowBytes * height)
};

/**
 * @brief Gets the memcpy parameters for a tensor to perform a memcpy2D operation.
 *
 * @param tensor The tensor to get the memcpy parameters for.
 * @return The memcpy parameters to perform a memcpy2D operation.
 */
inline MemcpyParams GetMemcpyParams(const roccv::Tensor& tensor) {
    MemcpyParams params;

    roccv::TensorDataStrided tensorData = tensor.exportData<roccv::TensorDataStrided>();
    params.rowPitch = tensorData.stride(tensor.layout().height_index());
    params.rowBytes = tensor.shape(tensor.layout().width_index()) * tensor.shape(tensor.layout().channels_index()) *
                      tensor.dtype().size();
    params.imageBytes = params.rowPitch * tensor.shape(tensor.layout().height_index());
    params.basePtr = tensorData.basePtr();

    return params;
}

template <typename T>
void MoveToTensor(const roccv::Tensor& tensor, const std::vector<T>& vec) {
    const hipMemcpyKind kind = (tensor.device() == eDeviceType::GPU) ? hipMemcpyHostToDevice : hipMemcpyHostToHost;

    if (tensor.isContiguous()) {
        // Contiguous data, so we can use a simple memcpy.
        const size_t totalBytes = tensor.dataSize();
        void* basePtr = tensor.exportData<roccv::TensorDataStrided>().basePtr();
        HIP_VALIDATE_NO_ERRORS(hipMemcpy(basePtr, vec.data(), totalBytes, kind));
    } else {
        // Data is padded, so we need to use a memcpy2D.
        const MemcpyParams params = GetMemcpyParams(tensor);
        const size_t batchSize = tensor.shape(tensor.layout().batch_index());
        const size_t totalRows = batchSize * tensor.shape(tensor.layout().height_index());
        HIP_VALIDATE_NO_ERRORS(hipMemcpy2D(params.basePtr, params.rowPitch, vec.data(), params.rowBytes,
                                           params.rowBytes, totalRows, kind));
    }
}

void FillTensor(const roccv::Tensor& tensor) {
    switch (tensor.dtype().etype()) {
        case DATA_TYPE_U8: {
            std::vector<uint8_t> vec = roccvbench::RandVector<uint8_t>(tensor.shape().size());
            MoveToTensor<uint8_t>(tensor, vec);
            break;
        }

        case DATA_TYPE_S8: {
            std::vector<int8_t> vec = roccvbench::RandVector<int8_t>(tensor.shape().size());
            MoveToTensor<int8_t>(tensor, vec);
            break;
        }

        case DATA_TYPE_F32: {
            std::vector<float> vec = roccvbench::RandVector<float>(tensor.shape().size());
            MoveToTensor<float>(tensor, vec);
            break;
        }

        case DATA_TYPE_F64: {
            std::vector<double> vec = roccvbench::RandVector<double>(tensor.shape().size());
            MoveToTensor<double>(tensor, vec);
            break;
        }

        case DATA_TYPE_S32: {
            std::vector<int32_t> vec = roccvbench::RandVector<int32_t>(tensor.shape().size());
            MoveToTensor<int32_t>(tensor, vec);
            break;
        }

        case DATA_TYPE_U32: {
            std::vector<uint32_t> vec = roccvbench::RandVector<uint32_t>(tensor.shape().size());
            MoveToTensor<uint32_t>(tensor, vec);
            break;
        }

        default: {
            throw std::runtime_error("Unsupported tensor data type.");
            break;
        }
    }
}