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
#include <rocrand/rocrand.h>

#include <roccvbench/utils.hpp>

namespace {
static rocrand_generator GetGenerator(eDeviceType device) {
    rocrand_generator gen;
    switch (device) {
        case eDeviceType::GPU: {
            rocrand_create_generator(&gen, ROCRAND_RNG_PSEUDO_DEFAULT);
            break;
        }
        case eDeviceType::CPU: {
            rocrand_create_generator_host_blocking(&gen, ROCRAND_RNG_PSEUDO_DEFAULT);
            break;
        }

        default: {
            throw std::runtime_error("Unsupported device type.");
        }
    }

    return gen;
}

template <typename T>
void FillTensorImpl(const roccv::Tensor& tensor) {
    const auto tensor_data = tensor.exportData<roccv::TensorDataStrided>();
    const auto generator = GetGenerator(tensor.device());

    if constexpr (std::is_integral_v<T>) {
        rocrand_generate_char(generator, static_cast<unsigned char*>(tensor_data.basePtr()),
                              tensor.shape().size() * tensor.dtype().size());
    } else if constexpr (std::is_same_v<T, float>) {
        rocrand_generate_uniform(generator, static_cast<float*>(tensor_data.basePtr()), tensor.shape().size());
    } else if constexpr (std::is_same_v<T, double>) {
        rocrand_generate_uniform_double(generator, static_cast<double*>(tensor_data.basePtr()), tensor.shape().size());
    } else {
        throw std::runtime_error("Unsupported data type.");
    }
    HIP_VALIDATE_NO_ERRORS(hipDeviceSynchronize());
}
}  // namespace

void FillTensor(const roccv::Tensor& tensor) {
    static const std::unordered_map<eDataType, void (*)(const roccv::Tensor&)> fillTensorImpls = {
        {eDataType::DATA_TYPE_U8, FillTensorImpl<uint8_t>},   {eDataType::DATA_TYPE_S8, FillTensorImpl<int8_t>},
        {eDataType::DATA_TYPE_U16, FillTensorImpl<uint16_t>}, {eDataType::DATA_TYPE_S16, FillTensorImpl<int16_t>},
        {eDataType::DATA_TYPE_U32, FillTensorImpl<uint32_t>}, {eDataType::DATA_TYPE_S32, FillTensorImpl<int32_t>},
        {eDataType::DATA_TYPE_F32, FillTensorImpl<float>},    {eDataType::DATA_TYPE_F64, FillTensorImpl<double>},
    };

    fillTensorImpls.at(tensor.dtype().etype())(tensor);
}