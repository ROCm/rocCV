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

#include "core/tensor.hpp"

#include <array>

#include "core/data_type.hpp"
#include "core/detail/context.hpp"
#include "core/exception.hpp"
#include "core/image_format.hpp"
#include "core/mem_alignment.hpp"
#include "core/status_type.h"
#include "core/tensor_data.hpp"
#include "core/tensor_layout.hpp"
#include "core/tensor_requirements.hpp"
#include "core/tensor_shape.hpp"
#include "core/util_enums.h"
#include "operator_types.h"

namespace roccv {

// Constructor definitions
Tensor::Tensor(const Tensor::Requirements& reqs, const IAllocator& alloc) : m_requirements(reqs), m_allocator(alloc) {
    size_t numBytes = reqs.device == eDeviceType::GPU ? reqs.res.deviceMem.bytes : reqs.res.hostMem.bytes;
    m_data = std::make_shared<TensorStorage>(numBytes, reqs.device, alloc);
}

Tensor::Tensor(const Tensor::Requirements& reqs, std::shared_ptr<TensorStorage> data)
    : m_requirements(reqs), m_data(data), m_allocator(data->allocator()) {}

Tensor::Tensor(const TensorShape& shape, DataType dtype, const eDeviceType device)
    : Tensor(shape, dtype, {}, GlobalContext().getDefaultAllocator(), device) {}

Tensor::Tensor(const TensorShape& shape, DataType dtype, const MemAlignment& bufAlign, const IAllocator& alloc,
               const eDeviceType device)
    : Tensor(CalcRequirements(shape, dtype, device), alloc) {}

Tensor::Tensor(int num_images, Size2D image_size, ImageFormat fmt, eDeviceType device)
    : Tensor(num_images, image_size, fmt, {}, GlobalContext().getDefaultAllocator(), device) {}

Tensor::Tensor(int num_images, Size2D image_size, ImageFormat fmt, const MemAlignment& bufAlign,
               const IAllocator& alloc, eDeviceType device)
    : Tensor(CalcRequirements(num_images, image_size, fmt, device), alloc) {}

Tensor::Tensor(Tensor&& other)
    : m_requirements(std::move(other.m_requirements)),
      m_data(std::move(other.m_data)),
      m_allocator(other.m_allocator) {}

// Member definitions
int Tensor::rank() const { return m_requirements.rank; }

eDeviceType Tensor::device() const { return m_requirements.device; }

TensorShape Tensor::shape() const { return TensorShape(m_requirements.shape, m_requirements.rank, layout()); }

int64_t Tensor::shape(int d) const& { return shape()[d]; }

DataType Tensor::dtype() const { return DataType(m_requirements.dtype); }

TensorLayout Tensor::layout() const { return TensorLayout(m_requirements.layout); }

TensorData Tensor::exportData() const {
    TensorBufferStrided buffer;
    buffer.basePtr = m_data->data();
    buffer.strides = m_requirements.strides;

    switch (device()) {
        case eDeviceType::GPU: {
            return TensorDataStridedHip(shape(), dtype(), buffer);
        }

        case eDeviceType::CPU: {
            return TensorDataStridedHost(shape(), dtype(), buffer);
        }

        default: {
            throw Exception("Unsupported device type in Tensor::exportData().", eStatusType::INVALID_VALUE);
        }
    }
}

Tensor Tensor::reshape(const TensorShape& new_shape) const {
    // New tensor shape must have the same number of elements
    if (new_shape.size() != this->shape().size()) {
        throw Exception("New tensor shape does not match the number of elements of the old shape.",
                        eStatusType::INVALID_VALUE);
    }

    Tensor::Requirements reqs = CalcRequirements(new_shape, this->dtype(), this->device());
    return Tensor(reqs, m_data);
}

Tensor Tensor::reshape(const TensorShape& new_shape, const DataType& new_dtype) const {
    if (new_shape.size() * new_dtype.size() != this->shape().size() * this->dtype().size()) {
        throw Exception("New tensor view must have the same underlying number of bytes.", eStatusType::INVALID_VALUE);
    }

    Tensor::Requirements reqs = CalcRequirements(new_shape, new_dtype, this->device());
    return Tensor(reqs, m_data);
}

Tensor& Tensor::operator=(const Tensor& other) {
    this->m_requirements = other.m_requirements;
    this->m_data = other.m_data;
    return *this;
}

Tensor::Requirements Tensor::CalcRequirements(const TensorShape& shape, const DataType& dtype,
                                              const eDeviceType device) {
    return CalcRequirements(shape, dtype, (MemAlignment){}, device);
}

Tensor::Requirements Tensor::CalcRequirements(const TensorShape& shape, const DataType& dtype,
                                              const MemAlignment& bufAlign, const eDeviceType device) {
    std::array<int64_t, ROCCV_TENSOR_MAX_RANK> strides = CalcStrides(shape, dtype);
    Tensor::Requirements reqs = CalcRequirements(shape, dtype, strides, device);
    return reqs;
}

Tensor::Requirements Tensor::CalcRequirements(const TensorShape& shape, const DataType& dtype,
                                              std::array<int64_t, ROCCV_TENSOR_MAX_RANK> strides, eDeviceType device) {
    Tensor::Requirements reqs;

    reqs.shape = shape.shape();
    reqs.rank = shape.layout().rank();
    reqs.layout = shape.layout().elayout();
    reqs.strides = strides;
    reqs.dtype = dtype.etype();
    reqs.alignBytes = 0;  // TODO: Must be specified later
    reqs.device = device;

    // Determine resource usage
    size_t numBytes = reqs.strides[0] * reqs.shape[0];
    switch (reqs.device) {
        case eDeviceType::GPU: {
            reqs.res.deviceMem.bytes = numBytes;
            break;
        }

        case eDeviceType::CPU: {
            reqs.res.hostMem.bytes = numBytes;
            break;
        }

        default: {
            throw Exception("Unsupported device when calling Tensor::CalcRequirements().", eStatusType::INVALID_VALUE);
        }
    }

    return reqs;
}

Tensor::Requirements Tensor::CalcRequirements(int num_images, Size2D image_size, ImageFormat fmt, eDeviceType device) {
    return CalcRequirements(num_images, image_size, fmt, (MemAlignment){}, device);
}

Tensor::Requirements Tensor::CalcRequirements(int num_images, Size2D image_size, ImageFormat fmt,
                                              const MemAlignment& bufAlign, eDeviceType device) {
    // TODO: Need to support different types of tensor layouts. This will happen once more image formats are supported
    // first.
    TensorShape shape(TensorLayout(eTensorLayout::TENSOR_LAYOUT_NHWC),
                      {num_images, image_size.h, image_size.w, fmt.channels()});
    return CalcRequirements(shape, DataType(fmt.dtype()), bufAlign, device);
}

std::array<int64_t, ROCCV_TENSOR_MAX_RANK> Tensor::CalcStrides(const TensorShape& shape, const DataType& dtype) {
    // TODO: Support memory alignment and padding in stride calculations

    // Calculate strides based on the given tensor shape. Strides are byte-wise.
    std::array<int64_t, ROCCV_TENSOR_MAX_RANK> strides;
    strides[shape.layout().rank() - 1] = dtype.size();
    for (int i = shape.layout().rank() - 2; i >= 0; i--) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    return strides;
}

Tensor TensorWrapData(const TensorData& tensor_data) {
    auto tensorDataStrided = tensor_data.cast<TensorDataStrided>();
    if (!tensorDataStrided.has_value()) {
        throw Exception("TensorData could not be cast to TensorDataStrided. Tensors can only wrap strided tensor data.",
                        eStatusType::INVALID_VALUE);
    }

    std::array<int64_t, ROCCV_TENSOR_MAX_RANK> strides;
    for (int i = 0; i < tensorDataStrided->rank(); i++) {
        strides[i] = tensorDataStrided->stride(i);
    }

    Tensor::Requirements reqs = Tensor::CalcRequirements(tensorDataStrided->shape(), tensorDataStrided->dtype(),
                                                         strides, tensorDataStrided->device());

    auto data =
        std::make_shared<TensorStorage>(tensorDataStrided->basePtr(), tensorDataStrided->device(), eOwnership::OWNING);
    return Tensor(reqs, data);
}

}  // namespace roccv