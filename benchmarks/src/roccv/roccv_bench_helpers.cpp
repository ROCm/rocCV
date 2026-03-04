/*
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
#include <rocrand/rocrand.h>

#include <roccvbench/utils.hpp>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>

namespace {

class RandomGenerator {
   public:
    RandomGenerator(eDeviceType device) {
        switch (device) {
            case eDeviceType::GPU: {
                rocrand_create_generator(&m_gen, ROCRAND_RNG_PSEUDO_DEFAULT);
                break;
            }
            case eDeviceType::CPU: {
                rocrand_create_generator_host_blocking(&m_gen, ROCRAND_RNG_PSEUDO_DEFAULT);
                break;
            }
            default: {
                throw std::runtime_error("Unsupported device type.");
            }
        }
    }

    /**
     * @brief Generates random data into a tensor.
     *
     * @tparam T The type of the data to generate.
     * @param tensor The tensor to generate data into.
     */
    template <typename T>
    void generate(const roccv::Tensor& tensor) {
        const auto tensor_data = tensor.exportData<roccv::TensorDataStrided>();

        const size_t numElements = tensor.dataSize() / tensor.dtype().size();

        if constexpr (std::is_integral_v<T>) {
            rocrand_generate_char(m_gen, static_cast<unsigned char*>(tensor_data.basePtr()), numElements);
        } else if constexpr (std::is_same_v<T, float>) {
            rocrand_generate_uniform(m_gen, static_cast<float*>(tensor_data.basePtr()), numElements);
        } else if constexpr (std::is_same_v<T, double>) {
            rocrand_generate_uniform_double(m_gen, static_cast<double*>(tensor_data.basePtr()), numElements);
        } else {
            throw std::runtime_error("Unsupported data type.");
        }

        if (tensor.device() == eDeviceType::GPU) {
            HIP_VALIDATE_NO_ERRORS(hipDeviceSynchronize());
        }
    }

    ~RandomGenerator() { rocrand_destroy_generator(m_gen); }

   private:
    rocrand_generator m_gen;
};

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
void FillTensorImpl(const roccv::Tensor& tensor) {
    RandomGenerator generator(tensor.device());
    generator.generate<T>(tensor);
}
}  // namespace

void FillTensor(const roccv::Tensor& tensor) {
    static const std::unordered_map<eDataType, void (*)(const roccv::Tensor&)> fillTensorImpls = {
        {eDataType::DATA_TYPE_U8, FillTensorImpl<uint8_t>},   {eDataType::DATA_TYPE_S8, FillTensorImpl<int8_t>},
        {eDataType::DATA_TYPE_U16, FillTensorImpl<uint16_t>}, {eDataType::DATA_TYPE_S16, FillTensorImpl<int16_t>},
        {eDataType::DATA_TYPE_U32, FillTensorImpl<uint32_t>}, {eDataType::DATA_TYPE_S32, FillTensorImpl<int32_t>},
        {eDataType::DATA_TYPE_F32, FillTensorImpl<float>},    {eDataType::DATA_TYPE_F64, FillTensorImpl<double>},
    };

    if (!fillTensorImpls.contains(tensor.dtype().etype())) {
        throw std::runtime_error("Unsupported data type.");
    }
    fillTensorImpls.at(tensor.dtype().etype())(tensor);
}