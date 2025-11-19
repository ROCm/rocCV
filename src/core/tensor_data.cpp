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

#include "core/tensor_data.hpp"

#include "core/data_type.hpp"
#include "core/tensor_buffer.hpp"
#include "core/util_enums.h"

namespace roccv {

int TensorData::rank() const { return m_shape.layout().rank(); }

const TensorShape& TensorData::shape() const& { return m_shape; }

const int64_t TensorData::shape(int d) const& { return m_shape[d]; }

const DataType& TensorData::dtype() const { return m_dtype; }

const eDeviceType TensorData::device() const { return m_deviceType; }

TensorData::TensorData(const TensorShape& tshape, const DataType& dtype, const TensorBuffer& buffer)
    : m_shape(tshape), m_dtype(dtype), m_bufferType(TensorBufferType::TENSOR_BUFFER_NONE), m_buffer(buffer) {}

bool TensorData::IsCompatibleKind(TensorBufferType bufferType) {
    return bufferType != TensorBufferType::TENSOR_BUFFER_NONE;
}

TensorDataStrided::TensorDataStrided(const TensorShape& tshape, const DataType& dtype, const TensorBuffer& buffer)
    : TensorData(tshape, dtype, buffer) {}

bool TensorDataStrided::IsCompatibleKind(TensorBufferType bufferType) {
    return bufferType == TensorBufferType::TENSOR_BUFFER_STRIDED_HIP ||
           bufferType == TensorBufferType::TENSOR_BUFFER_STRIDED_HOST;
}

void* roccv::TensorDataStrided::basePtr() const { return m_buffer.strided.basePtr; }

const int64_t TensorDataStrided::stride(int d) const { return m_buffer.strided.strides[d]; }

TensorDataStridedHip::TensorDataStridedHip(const TensorShape& shape, const DataType& dtype, const TensorBuffer& buffer)
    : TensorDataStrided(shape, dtype, buffer) {
    m_bufferType = TensorBufferType::TENSOR_BUFFER_STRIDED_HIP;
    m_deviceType = eDeviceType::GPU;
}

TensorDataStridedHip::TensorDataStridedHip(const TensorShape& shape, const DataType& dtype,
                                           const TensorDataStridedHip::Buffer& buffer)
    : TensorDataStridedHip(shape, dtype, {.strided = buffer}) {}

bool TensorDataStridedHip::IsCompatibleKind(TensorBufferType bufferType) {
    return bufferType == TensorBufferType::TENSOR_BUFFER_STRIDED_HIP;
}

TensorDataStridedHost::TensorDataStridedHost(const TensorShape& shape, const DataType& dtype,
                                             const TensorBuffer& buffer)
    : TensorDataStrided(shape, dtype, buffer) {
    m_bufferType = TensorBufferType::TENSOR_BUFFER_STRIDED_HOST;
    m_deviceType = eDeviceType::CPU;
}

TensorDataStridedHost::TensorDataStridedHost(const TensorShape& shape, const DataType& dtype, const Buffer& buffer)
    : TensorDataStridedHost(shape, dtype, {.strided = buffer}) {}

bool TensorDataStridedHost::IsCompatibleKind(TensorBufferType bufferType) {
    return bufferType == TensorBufferType::TENSOR_BUFFER_STRIDED_HOST;
}

}  // namespace roccv