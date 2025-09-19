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

#include "core/detail/context.hpp"
#include "core/exception.hpp"
#include "core/hip_assert.h"
#include "core/image_format.hpp"
#include "operator_types.h"

namespace roccv {

// Constructor definitions
Tensor::Tensor(const TensorRequirements& reqs) : Tensor(reqs, GlobalContext().getDefaultAllocator()) {}

Tensor::Tensor(const TensorRequirements& reqs, const IAllocator& alloc) : m_requirements(reqs), m_allocator(alloc) {
    m_data = std::make_shared<TensorStorage>(reqs.shape.size() * reqs.dtype.size(), reqs.device, alloc);
}
Tensor::Tensor(const TensorRequirements& reqs, std::shared_ptr<TensorStorage> data)
    : Tensor(reqs, data, GlobalContext().getDefaultAllocator()) {}

Tensor::Tensor(const TensorRequirements& reqs, std::shared_ptr<TensorStorage> data, const IAllocator& alloc)
    : m_requirements(reqs), m_data(data), m_allocator(alloc) {}

Tensor::Tensor(const TensorShape& shape, DataType dtype, const eDeviceType device)
    : Tensor(shape, dtype, GlobalContext().getDefaultAllocator(), device) {}

Tensor::Tensor(const TensorShape& shape, DataType dtype, const IAllocator& alloc, const eDeviceType device)
    : Tensor(CalcRequirements(shape, dtype, device), alloc) {}

Tensor::Tensor(int num_images, Size2D image_size, ImageFormat fmt, eDeviceType device)
    : Tensor(num_images, image_size, fmt, GlobalContext().getDefaultAllocator(), device) {}

Tensor::Tensor(int num_images, Size2D image_size, ImageFormat fmt, const IAllocator& alloc, eDeviceType device)
    : Tensor(CalcRequirements(num_images, image_size, fmt, device), alloc) {}

Tensor::Tensor(Tensor&& other)
    : m_requirements(other.m_requirements), m_data(other.m_data), m_allocator(other.m_allocator) {}

// Member definitions
int Tensor::rank() const { return m_requirements.shape.layout().rank(); }

const eDeviceType Tensor::device() const { return m_requirements.device; }

const TensorShape& Tensor::shape() const { return m_requirements.shape; }

const int64_t Tensor::shape(int d) const& { return m_requirements.shape[d]; }

const DataType& Tensor::dtype() const { return m_requirements.dtype; }

const TensorLayout& Tensor::layout() const { return m_requirements.shape.layout(); }

TensorData Tensor::exportData() const {
    TensorBufferStrided buffer;
    buffer.basePtr = m_data->data();
    buffer.strides = m_requirements.strides;
    TensorDataStrided data(shape(), dtype(), buffer, device());
    return data;
}

Tensor Tensor::reshape(const TensorShape& new_shape) const {
    // New tensor shape must have the same number of elements
    if (new_shape.size() != this->shape().size()) {
        throw Exception("New tensor shape does not match the number of elements of the old shape.",
                        eStatusType::INVALID_VALUE);
    }

    TensorRequirements reqs = CalcRequirements(new_shape, this->dtype(), this->device());
    return Tensor(reqs, m_data);
}

Tensor Tensor::reshape(const TensorShape& new_shape, const DataType& new_dtype) const {
    if (new_shape.size() * new_dtype.size() != this->shape().size() * this->dtype().size()) {
        throw Exception("New tensor view must have the same underlying number of bytes.", eStatusType::INVALID_VALUE);
    }

    TensorRequirements reqs = CalcRequirements(new_shape, new_dtype, this->device());
    return Tensor(reqs, m_data);
}

Tensor& Tensor::operator=(const Tensor& other) {
    this->m_requirements = other.m_requirements;
    this->m_data = other.m_data;
    return *this;
}

TensorRequirements Tensor::CalcRequirements(const TensorShape& shape, DataType dtype, const eDeviceType device) {
    // Calculate strides based on the given tensor shape. Strides are byte-wise.
    std::array<int64_t, ROCCV_TENSOR_MAX_RANK> strides;
    strides[shape.layout().rank() - 1] = dtype.size();
    for (int i = shape.layout().rank() - 2; i >= 0; i--) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }

    TensorRequirements reqs = {dtype, device, shape, strides};

    return reqs;
}

TensorRequirements Tensor::CalcRequirements(int num_images, Size2D image_size, ImageFormat fmt, eDeviceType device) {
    // TODO: Need to support different types of tensor layouts. This will happen once more image formats are supported
    // first.
    TensorShape shape(TensorLayout(eTensorLayout::TENSOR_LAYOUT_NHWC),
                      {num_images, image_size.h, image_size.w, fmt.channels()});
    return CalcRequirements(shape, DataType(fmt.dtype()), device);
}

Tensor TensorWrapData(const TensorData& tensor_data) {
    TensorRequirements req = Tensor::CalcRequirements(tensor_data.shape(), tensor_data.dtype(), tensor_data.device());
    auto data = std::make_shared<TensorStorage>(tensor_data.basePtr(), tensor_data.device(), eOwnership::OWNING);
    return Tensor(req, data);
}

}  // namespace roccv