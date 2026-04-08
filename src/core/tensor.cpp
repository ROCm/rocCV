/**
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

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

namespace {

/**
 * @brief Returns the index of the first packed dimension in the given tensor layout.
 *
 * In most cases, the first packed dimension is the last dimension. However, for layouts ending in WC, the first packed
 * dimension is the second to last dimension.
 *
 * @param[in] layout The tensor layout to get the first packed dimension for.
 * @return The index of the first packed dimension in the given tensor layout.
 */
static int GetFirstPackedDimension(const TensorLayout& layout) {
    const int rank = layout.rank();
    switch (layout.elayout()) {
        case eTensorLayout::TENSOR_LAYOUT_NHWC:
        case eTensorLayout::TENSOR_LAYOUT_LNHWC:
        case eTensorLayout::TENSOR_LAYOUT_HWC:
        case eTensorLayout::TENSOR_LAYOUT_NWC:
            return std::max(0, rank - 2);
        default:
            return rank - 1;
    }
}

static bool ReshapeSimplified(int inRank, const std::array<int64_t, ROCCV_TENSOR_MAX_RANK>& inShape,
                              const std::array<int64_t, ROCCV_TENSOR_MAX_RANK>& inStrides, int targetRank,
                              const std::array<int64_t, ROCCV_TENSOR_MAX_RANK>& targetShape,
                              std::array<int64_t, ROCCV_TENSOR_MAX_RANK>& outStrides) {
    int i = 0, j = 0;
    for (; i < inRank && j < targetRank; i++) {
        int64_t inE = inShape[i];
        int64_t outV = 1;
        int group_start = j;
        while (j < targetRank && (outV * targetShape[j]) <= inE) outV *= targetShape[j++];

        if (outV != inE) {
            return false;
        }

        int64_t s = inStrides[i];
        for (int d = j - 1; d >= group_start; d--) {
            outStrides[d] = s;
            s *= targetShape[d];
        }
    }
    return true;
}

/**
 * @brief Simplifies a tensor shape and strides into their canonical form.
 *
 * @param[in] rank The rank of the tensor.
 * @param[in] shape The shape of the tensor.
 * @param[in] strides The strides of the tensor.
 * @param[out] outShape The simplified shape of the tensor.
 * @param[out] outStrides The simplified strides of the tensor.
 * @return The rank of the simplified tensor.
 */
static int Simplify(int rank, const std::array<int64_t, ROCCV_TENSOR_MAX_RANK>& shape,
                    const std::array<int64_t, ROCCV_TENSOR_MAX_RANK>& strides,
                    std::array<int64_t, ROCCV_TENSOR_MAX_RANK>& outShape,
                    std::array<int64_t, ROCCV_TENSOR_MAX_RANK>& outStrides) {
    if (rank <= 1) {
        if (rank == 1) {
            outShape[0] = shape[0];
            outStrides[0] = strides[0];
        }
        return rank;
    }

    int outRank = 0;
    int64_t vol = shape[0];
    for (int d = 1; d < rank; d++) {
        if (strides[d - 1] != shape[d] * strides[d]) {
            outStrides[outRank] = strides[d - 1];
            outShape[outRank] = vol;
            vol = shape[d];
            outRank++;
        } else {
            vol *= shape[d];
        }
    }
    outStrides[outRank] = strides[rank - 1];
    outShape[outRank] = vol;
    outRank++;
    return outRank;
}

/**
 * @brief Computes copy parameters for host-tensor copy.
 * @return (row_width_bytes, num_rows, tensor_pitch). If contiguous, returns (total_size, 1, total_size).
 */
static std::tuple<size_t, size_t, size_t> ComputeCopyParams(int rank,
                                                            const std::array<int64_t, ROCCV_TENSOR_MAX_RANK>& shape,
                                                            const std::array<int64_t, ROCCV_TENSOR_MAX_RANK>& strides,
                                                            size_t dtypeSize, bool contiguous) {
    if (contiguous) {
        size_t totalSize = dtypeSize;
        for (int i = 0; i < rank; ++i) {
            totalSize *= static_cast<size_t>(shape[i]);
        }
        return {totalSize, 1, totalSize};
    }

    int paddedDim = 0;
    for (int i = 0; i < rank - 1; i++) {
        if (strides[i] != shape[i + 1] * strides[i + 1]) {
            paddedDim = i;
            break;
        }
    }

    size_t rowWidth = dtypeSize;
    for (int i = paddedDim + 1; i < rank; ++i) {
        rowWidth *= static_cast<size_t>(shape[i]);
    }

    size_t numRows = 1;
    for (int i = 0; i <= paddedDim; ++i) {
        numRows *= static_cast<size_t>(shape[i]);
    }

    size_t tensorPitch = static_cast<size_t>(strides[paddedDim]);
    return {rowWidth, numRows, tensorPitch};
}

/**
 * @brief Computes the memory alignment for a tensor based on the device, data type, and user provided buffer alignment.
 * @param[in] device The device the tensor is to be allocated on.
 * @param[in] dtype The datatype of the tensor.
 * @param[in] bufAlign The memory alignment to use.
 * @return The memory alignment for the tensor.
 */
MemAlignment ComputeMemAlignment(eDeviceType device, const DataType& dtype, const MemAlignment& bufAlign) {
    int dev = 0;
    if (device == eDeviceType::GPU) {
        HIP_VALIDATE_NO_ERRORS(hipGetDevice(&dev));
    }

    int rowAlign;
    if (bufAlign.rowAddr() == 0) {
        if (device == eDeviceType::GPU) {
            HIP_VALIDATE_NO_ERRORS(hipDeviceGetAttribute(&rowAlign, hipDeviceAttributeTexturePitchAlignment, dev));
        } else {
            rowAlign = ROCCV_CPU_DEFAULT_ALIGNMENT;
        }
        rowAlign = std::lcm(rowAlign, detail::NextPowerOfTwo(dtype.size()));
    } else {
        if (!detail::IsPowerOfTwo(bufAlign.rowAddr())) {
            throw Exception("Row address alignment must be a power of two.", eStatusType::INVALID_VALUE);
        }
        rowAlign = std::lcm(bufAlign.rowAddr(), detail::NextPowerOfTwo(dtype.size()));
    }

    int baseAlign;
    if (bufAlign.baseAddr() == 0) {
        if (device == eDeviceType::GPU) {
            HIP_VALIDATE_NO_ERRORS(hipDeviceGetAttribute(&baseAlign, hipDeviceAttributeTextureAlignment, dev));
        } else {
            baseAlign = ROCCV_CPU_DEFAULT_ALIGNMENT;
        }
        baseAlign = std::lcm(baseAlign, detail::NextPowerOfTwo(dtype.size()));
    } else {
        if (!detail::IsPowerOfTwo(bufAlign.baseAddr())) {
            throw Exception("Base address alignment must be a power of two.", eStatusType::INVALID_VALUE);
        }
        baseAlign = std::lcm(bufAlign.baseAddr(), detail::NextPowerOfTwo(dtype.size()));
    }

    return MemAlignment().baseAddr(baseAlign).rowAddr(rowAlign);
}
}  // namespace

// Constructor definitions
Tensor::Tensor(const Tensor::Requirements& reqs, const IAllocator& alloc) : m_requirements(reqs) {
    m_data = std::make_shared<TensorStorage>(this->dataSize(), reqs.device, alloc);
}

Tensor::Tensor(const Tensor::Requirements& reqs, std::shared_ptr<TensorStorage> data)
    : m_requirements(reqs), m_data(data) {}

Tensor::Tensor(const TensorShape& shape, DataType dtype, eDeviceType device)
    : Tensor(shape, dtype, {}, GlobalContext().getDefaultAllocator(), device) {}

Tensor::Tensor(const TensorShape& shape, DataType dtype, const MemAlignment& bufAlign, const IAllocator& alloc,
               eDeviceType device)
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

int64_t Tensor::shape(std::string_view dimension) const& { return shape()[dimension]; }

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

Tensor Tensor::reshape(const TensorShape& newShape) const { return reshape(dtype(), newShape); }

Tensor Tensor::reshape(const DataType& newDtype, const TensorShape& newShape) const {
    if (newShape.size() * newDtype.size() != this->shape().size() * this->dtype().size()) {
        throw Exception("New tensor view must have the same underlying number of bytes.", eStatusType::INVALID_VALUE);
    }

    const int oldRank = m_requirements.rank;
    const int newRank = newShape.layout().rank();

    if (m_requirements.strides[oldRank - 1] != static_cast<int64_t>(dtype().size())) {
        throw Exception("Cannot reshape tensor: innermost dimension is not element-contiguous.",
                        eStatusType::INVALID_VALUE);
    }

    // Convert to a byte-level view by expanding the innermost dimension size
    // by the element size and setting its stride to 1.
    std::array<int64_t, ROCCV_TENSOR_MAX_RANK> byteShape = m_requirements.shape;
    std::array<int64_t, ROCCV_TENSOR_MAX_RANK> byteStrides = m_requirements.strides;
    byteShape[oldRank - 1] *= dtype().size();
    byteStrides[oldRank - 1] = 1;

    std::array<int64_t, ROCCV_TENSOR_MAX_RANK> simpleShape, simpleStrides;
    int simpleRank = Simplify(oldRank, byteShape, byteStrides, simpleShape, simpleStrides);

    std::array<int64_t, ROCCV_TENSOR_MAX_RANK> targetByteShape = newShape.shape();
    targetByteShape[newRank - 1] *= newDtype.size();

    std::array<int64_t, ROCCV_TENSOR_MAX_RANK> targetByteStrides;
    bool result =
        ReshapeSimplified(simpleRank, simpleShape, simpleStrides, newRank, targetByteShape, targetByteStrides);
    if (!result) {
        throw Exception("Cannot reshape tensor into requested shape and data type.", eStatusType::INVALID_VALUE);
    }

    // The byte-level reshape produces stride 1 for the innermost dimension;
    // scale it back to the new element size.
    std::array<int64_t, ROCCV_TENSOR_MAX_RANK> newStrides = targetByteStrides;
    newStrides[newRank - 1] = newDtype.size();

    Tensor::Requirements reqs =
        CalcRequirements(newShape, newDtype, newStrides, m_requirements.alignBytes, this->device());
    return Tensor(reqs, m_data);
}

Tensor& Tensor::operator=(const Tensor& other) {
    this->m_requirements = other.m_requirements;
    this->m_data = other.m_data;
    return *this;
}

size_t Tensor::dataSize() const { return m_requirements.strides[0] * m_requirements.shape[0]; }

bool Tensor::isContiguous() const { return dataSize() == shape().size() * dtype().size(); }

void Tensor::copyFromHostAsync(const void* src, hipStream_t stream) const {
    auto [rowWidth, numRows, tensorPitch] = ComputeCopyParams(m_requirements.rank, m_requirements.shape,
                                                              m_requirements.strides, dtype().size(), isContiguous());

    const size_t srcPitch = rowWidth;
    hipMemcpyKind kind = (device() == eDeviceType::GPU) ? hipMemcpyHostToDevice : hipMemcpyHostToHost;

    HIP_VALIDATE_NO_ERRORS(
        hipMemcpy2DAsync(m_data->data(), tensorPitch, src, srcPitch, rowWidth, numRows, kind, stream));
}

void Tensor::copyToHostAsync(void* dst, hipStream_t stream) const {
    auto [rowWidth, numRows, tensorPitch] = ComputeCopyParams(m_requirements.rank, m_requirements.shape,
                                                              m_requirements.strides, dtype().size(), isContiguous());

    const size_t dstPitch = rowWidth;
    hipMemcpyKind kind = (device() == eDeviceType::GPU) ? hipMemcpyDeviceToHost : hipMemcpyHostToHost;

    HIP_VALIDATE_NO_ERRORS(
        hipMemcpy2DAsync(dst, dstPitch, m_data->data(), tensorPitch, rowWidth, numRows, kind, stream));
}

Tensor::Requirements Tensor::CalcRequirements(const TensorShape& shape, const DataType& dtype, eDeviceType device) {
    return CalcRequirements(shape, dtype, (MemAlignment){}, device);
}

Tensor::Requirements Tensor::CalcRequirements(const TensorShape& shape, const DataType& dtype,
                                              const MemAlignment& bufAlign, eDeviceType device) {
    MemAlignment newAlign = ComputeMemAlignment(device, dtype, bufAlign);

    std::array<int64_t, ROCCV_TENSOR_MAX_RANK> strides = CalcStrides(shape, dtype, newAlign.rowAddr());
    Tensor::Requirements reqs = CalcRequirements(shape, dtype, strides, newAlign.baseAddr(), device);
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

    const int firstPackedDim = GetFirstPackedDimension(shape.layout());

    strides[shape.layout().rank() - 1] = dtype.size();
    for (int i = shape.layout().rank() - 2; i >= 0; i--) {
        // The stride dimension preceeding the first packed dimension is padded to the next multiple of the row
        // alignment.
        if (i == firstPackedDim - 1) {
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