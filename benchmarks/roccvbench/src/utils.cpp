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

#include "roccvbench/utils.hpp"

#include <core/hip_assert.h>

#include <core/tensor.hpp>
#include <random>
#include <vector>

namespace roccvbench {

template <typename T>
std::vector<T> RandVector(size_t size) {
    std::random_device dev;
    std::mt19937 gen(dev());
    std::vector<T> result(size);

    if constexpr (std::is_floating_point_v<T>) {
        std::uniform_real_distribution<T> dist(0.0f, 1.0f);
        for (size_t i = 0; i < size; i++) {
            result[i] = dist(gen);
        }
    } else if constexpr (std::is_integral_v<T>) {
        std::uniform_int_distribution<int64_t> dist(std::numeric_limits<T>().min(), std::numeric_limits<T>().max());
        for (size_t i = 0; i < size; i++) {
            result[i] = static_cast<T>(dist(gen));
        }
    } else {
        static_assert(false, "Unsupported data type for random vector fill.\n");
    }

    return result;
}

template <typename T>
void MoveToTensor(const roccv::Tensor& tensor, const std::vector<T>& vec) {
    auto tensor_data = tensor.exportData<roccv::TensorDataStrided>();
    switch (tensor.device()) {
        case eDeviceType::GPU: {
            HIP_VALIDATE_NO_ERRORS(hipMemcpy(tensor_data.basePtr(), vec.data(),
                                             tensor.shape().size() * tensor.dtype().size(), hipMemcpyHostToDevice));
            break;
        }

        case eDeviceType::CPU: {
            HIP_VALIDATE_NO_ERRORS(hipMemcpy(tensor_data.basePtr(), vec.data(),
                                             tensor.shape().size() * tensor.dtype().size(), hipMemcpyHostToHost));
            break;
        }
    }
}

void FillTensor(const roccv::Tensor& tensor) {
    switch (tensor.dtype().etype()) {
        case DATA_TYPE_U8: {
            std::vector<uint8_t> vec = RandVector<uint8_t>(tensor.shape().size());
            MoveToTensor<uint8_t>(tensor, vec);
            break;
        }

        case DATA_TYPE_S8: {
            std::vector<int8_t> vec = RandVector<int8_t>(tensor.shape().size());
            MoveToTensor<int8_t>(tensor, vec);
            break;
        }

        case DATA_TYPE_F32: {
            std::vector<float> vec = RandVector<float>(tensor.shape().size());
            MoveToTensor<float>(tensor, vec);
            break;
        }

        case DATA_TYPE_F64: {
            std::vector<double> vec = RandVector<double>(tensor.shape().size());
            MoveToTensor<double>(tensor, vec);
            break;
        }

        case DATA_TYPE_S32: {
            std::vector<int32_t> vec = RandVector<int32_t>(tensor.shape().size());
            MoveToTensor<int32_t>(tensor, vec);
            break;
        }

        case DATA_TYPE_U32: {
            std::vector<uint32_t> vec = RandVector<uint32_t>(tensor.shape().size());
            MoveToTensor<uint32_t>(tensor, vec);
            break;
        }

        default: {
            throw std::runtime_error("Unsupported tensor data type.");
            break;
        }
    }
}
}  // namespace roccvbench