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
#include <numeric>

#include "core/data_type.hpp"
#include "core/detail/context.hpp"
#include "core/exception.hpp"
#include "core/hip_assert.h"
#include "core/image_format.hpp"
#include "core/mem_alignment.hpp"
#include "core/status_type.h"
#include "core/tensor_data.hpp"
#include "core/tensor_layout.hpp"
#include "core/tensor_requirements.hpp"
#include "core/tensor_shape.hpp"
#include "core/util_enums.h"
#include "core/utils.hpp"
#include "operator_types.h"

namespace roccv {

// Constructor definitions
Tensor::Tensor(const Tensor::Requirements& reqs, const IAllocator& alloc) : m_requirements(reqs) {
    m_data = std::make_shared<TensorStorage>(this->dataSize(), reqs.device, alloc);
}

Tensor::Tensor(const Tensor::Requirements& reqs, std::shared_ptr<TensorStorage> data)
    : m_requirements(reqs), m_data(data) {}

Tensor::Tensor(const TensorShape& shape, DataType dtype, const eDeviceType device)
    : Tensor(shape, dtype, {}, GlobalContext().getDefaultAllocator(), device) {}

Tensor::Tensor(const TensorShape& shape, DataType dtype, const MemAlignment& bufAlign, const IAllocator& alloc,
               const eDeviceType device)
    : Tensor(CalcRequirements(shape, dtype, bufAlign, device), alloc) {}

Tensor::Tensor(int num_images, Size2D image_size, ImageFormat fmt, eDeviceType device)
    : Tensor(num_images, image_size, fmt, {}, GlobalContext().getDefaultAllocator(), device) {}

Tensor::Tensor(int num_images, Size2D image_size, ImageFormat fmt, const MemAlignment& bufAlign,
               const IAllocator& alloc, eDeviceType device)
    : Tensor(CalcRequirements(num_images, image_size, fmt, bufAlign, device), alloc) {}

// Move constructor
Tensor::Tensor(Tensor&& other) : m_requirements(std::move(other.m_requirements)), m_data(std::move(other.m_data)) {}

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
    if (!isContiguous()) {
        throw Exception("Tensor is not contiguous. Reshape can only be performed on contiguous tensors.",
                        eStatusType::INVALID_VALUE);
    }

    // New tensor shape must have the same number of elements
    if (new_shape.size() != this->shape().size()) {
        throw Exception("New tensor shape does not match the number of elements of the old shape.",
                        eStatusType::INVALID_VALUE);
    }

    Tensor::Requirements reqs = CalcRequirements(new_shape, this->dtype(), this->device());
    return Tensor(reqs, m_data);
}

Tensor Tensor::reshape(const TensorShape& new_shape, const DataType& new_dtype) const {
    if (!isContiguous()) {
        throw Exception("Tensor is not contiguous. Reshape can only be performed on contiguous tensors.",
                        eStatusType::INVALID_VALUE);
    }

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

size_t Tensor::dataSize() const { return m_requirements.strides[0] * m_requirements.shape[0]; }

bool Tensor::isContiguous() const { return dataSize() == shape().size() * dtype().size(); }

Tensor::Requirements Tensor::CalcRequirements(const TensorShape& shape, const DataType& dtype,
                                              const eDeviceType device) {
    return CalcRequirements(shape, dtype, (MemAlignment){}, device);
}

Tensor::Requirements Tensor::CalcRequirements(const TensorShape& shape, const DataType& dtype,
                                              const MemAlignment& bufAlign, const eDeviceType device) {
    int dev;
    HIP_VALIDATE_NO_ERRORS(hipGetDevice(&dev));

    // Validate memory alignment, set default alignment if set to 0.
    // TODO: Must be supported for CPU as well.
    int rowAlign;
    if (bufAlign.rowAddr() == 0) {
        HIP_VALIDATE_NO_ERRORS(hipDeviceGetAttribute(&rowAlign, hipDeviceAttributeTexturePitchAlignment, dev));
        rowAlign = std::lcm(rowAlign, detail::NextPowerOfTwo(dtype.size()));
    } else {
        if (!detail::IsPowerOfTwo(bufAlign.rowAddr())) {
            throw Exception("Row address alignment must be a power of two.", eStatusType::INVALID_VALUE);
        }
        rowAlign = std::lcm(bufAlign.rowAddr(), detail::NextPowerOfTwo(dtype.size()));
    }

    int baseAlign;
    if (bufAlign.baseAddr() == 0) {
        HIP_VALIDATE_NO_ERRORS(hipDeviceGetAttribute(&baseAlign, hipDeviceAttributeTextureAlignment, dev));
        baseAlign = std::lcm(baseAlign, detail::NextPowerOfTwo(dtype.size()));
    } else {
        if (!detail::IsPowerOfTwo(bufAlign.baseAddr())) {
            throw Exception("Base address alignment must be a power of two.", eStatusType::INVALID_VALUE);
        }
        baseAlign = std::lcm(bufAlign.baseAddr(), detail::NextPowerOfTwo(dtype.size()));
    }

    std::array<int64_t, ROCCV_TENSOR_MAX_RANK> strides = CalcStrides(shape, dtype, rowAlign);
    Tensor::Requirements reqs = CalcRequirements(shape, dtype, strides, baseAlign, device);
    return reqs;
}

Tensor::Requirements Tensor::CalcRequirements(const TensorShape& shape, const DataType& dtype,
                                              const std::array<int64_t, ROCCV_TENSOR_MAX_RANK> strides,
                                              int32_t baseAlign, eDeviceType device) {
    Tensor::Requirements reqs;

    reqs.shape = shape.shape();
    reqs.rank = shape.layout().rank();
    reqs.layout = shape.layout().elayout();
    reqs.strides = strides;
    reqs.dtype = dtype.etype();
    reqs.alignBytes = baseAlign;
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

std::array<int64_t, ROCCV_TENSOR_MAX_RANK> Tensor::CalcStrides(const TensorShape& shape, const DataType& dtype,
                                                               int32_t rowAlign) {
    // Calculate strides based on the given tensor shape. Strides are byte-wise.
    std::array<int64_t, ROCCV_TENSOR_MAX_RANK> strides;
    strides[shape.layout().rank() - 1] = dtype.size();
    for (int i = shape.layout().rank() - 2; i >= 0; i--) {
        // Ensure strides for the row are padded to the next multiple of the alignment.
        if (i == shape.layout().height_index()) {
            strides[i] = detail::AlignUp(strides[i + 1] * shape[i + 1], rowAlign);
        } else {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
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
                                                         strides, 0, tensorDataStrided->device());

    auto data =
        std::make_shared<TensorStorage>(tensorDataStrided->basePtr(), tensorDataStrided->device(), eOwnership::OWNING);
    return Tensor(reqs, data);
}

}  // namespace roccv