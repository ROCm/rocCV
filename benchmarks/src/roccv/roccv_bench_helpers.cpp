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
#define ROCRAND_CHECK(status)                                                 \
    if (status != rocrand_status::ROCRAND_STATUS_SUCCESS) {                   \
        throw std::runtime_error("ROCRAND error: " + std::to_string(status)); \
    }

class RandomGenerator {
   public:
    RandomGenerator(eDeviceType device) {
        switch (device) {
            case eDeviceType::GPU: {
                ROCRAND_CHECK(rocrand_create_generator(&m_gen, ROCRAND_RNG_PSEUDO_DEFAULT));
                break;
            }
            case eDeviceType::CPU: {
                ROCRAND_CHECK(rocrand_create_generator_host_blocking(&m_gen, ROCRAND_RNG_PSEUDO_DEFAULT));
                break;
            }
            default: {
                throw std::runtime_error("Unsupported device type.");
            }
        }
        ROCRAND_CHECK(rocrand_set_seed(m_gen, roccvbench::kBenchSeed));
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

        if constexpr (std::is_integral_v<T>) {
            rocrand_generate_char(m_gen, static_cast<unsigned char*>(tensor_data.basePtr()), tensor.dataSize());
        } else if constexpr (std::is_same_v<T, float>) {
            size_t num_elements = tensor.dataSize() / sizeof(float);
            rocrand_generate_uniform(m_gen, static_cast<float*>(tensor_data.basePtr()), num_elements);
        } else if constexpr (std::is_same_v<T, double>) {
            size_t num_elements = tensor.dataSize() / sizeof(double);
            rocrand_generate_uniform_double(m_gen, static_cast<double*>(tensor_data.basePtr()), num_elements);
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