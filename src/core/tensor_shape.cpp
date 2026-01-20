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

#include "core/tensor_shape.hpp"

#include <algorithm>
#include <vector>

#include "core/exception.hpp"
#include "core/status_type.h"
#include "core/tensor_layout.hpp"

namespace roccv {

static const std::unordered_map<std::string, eTensorLayout> layoutDescToEnum = {
    {"N", eTensorLayout::TENSOR_LAYOUT_N},       {"NC", eTensorLayout::TENSOR_LAYOUT_NC},
    {"NW", eTensorLayout::TENSOR_LAYOUT_NW},     {"NWC", eTensorLayout::TENSOR_LAYOUT_NWC},
    {"HWC", eTensorLayout::TENSOR_LAYOUT_HWC},   {"NMC", eTensorLayout::TENSOR_LAYOUT_NMC},
    {"NMD", eTensorLayout::TENSOR_LAYOUT_NMD},   {"NHWC", eTensorLayout::TENSOR_LAYOUT_NHWC},
    {"NCHW", eTensorLayout::TENSOR_LAYOUT_NCHW}, {"LNHWC", eTensorLayout::TENSOR_LAYOUT_LNHWC},
};

const TensorLayout GetLayoutFromString(const std::string &desc) {
    if (!layoutDescToEnum.contains(desc)) {
        throw Exception("Invalid string descriptor for tensor layout.", eStatusType::INVALID_VALUE);
    }

    return TensorLayout(layoutDescToEnum.at(desc));
}

TensorShape::TensorShape(const TensorLayout &layout, const std::span<const int64_t> shape)
    : TensorShape(shape, shape.size(), layout) {}

TensorShape::TensorShape(const TensorLayout &layout, const std::initializer_list<const int64_t> shape)
    : TensorShape(layout, std::span<const int64_t>(shape.begin(), shape.end())) {}

TensorShape::TensorShape(const std::initializer_list<const int64_t> shape, const std::string &layoutDesc)
    : TensorShape(GetLayoutFromString(layoutDesc), shape) {}

TensorShape::TensorShape(const std::span<const int64_t> shape, const std::string &layoutDesc)
    : TensorShape(GetLayoutFromString(layoutDesc), shape) {}

TensorShape::TensorShape(const std::span<const int64_t> shape, int rank, eTensorLayout layout)
    : TensorShape(shape, rank, TensorLayout(layout)) {}

TensorShape::TensorShape(const std::span<const int64_t> shape, int rank, const TensorLayout &layout)
    : m_layout(layout) {
    if (rank < 0) {
        throw Exception("Rank must be a non-negative integer.", eStatusType::OUT_OF_BOUNDS);
    }

    if (rank != layout.rank()) {
        throw Exception(
            "Invalid shape size: The size of the shape must match the rank of "
            "the provided layout.",
            eStatusType::OUT_OF_BOUNDS);
    }

    if (shape.size() < static_cast<size_t>(rank)) {
        throw Exception("Size of the input shape data is less than the rank provided.", eStatusType::OUT_OF_BOUNDS);
    }

    for (int i = 0; i < rank; i++) {
        if (shape[i] <= 0) {
            throw Exception(
                "Invalid shape dimension: values of elements in the "
                "shape array must be > 0.",
                eStatusType::OUT_OF_BOUNDS);
        }
    }

    // Copy the std::span shape into the internal shape array.
    std::copy_n(shape.begin(), rank, m_shape.begin());

    // Calculate shape size
    m_size = 1;
    for (int i = 0; i < rank; i++) {
        m_size *= m_shape[i];
    }
}

TensorShape &TensorShape::operator=(const TensorShape &other) {
    if (this != &other) {
        m_layout = other.m_layout;
        m_shape = other.m_shape;
        m_size = other.m_size;
    }
    return *this;
}

int64_t TensorShape::operator[](int32_t i) const {
    if (i < 0 || i >= this->m_layout.rank()) {
        throw Exception("TensorShape index out of bounds: " + std::to_string(i) + ". Dimension must be >= 0 and < " +
                            std::to_string(this->m_layout.rank()),
                        eStatusType::OUT_OF_BOUNDS);
    }
    return m_shape[i];
}

int64_t TensorShape::operator[](std::string_view dimension) const {
    int32_t index = m_layout.indexOf(dimension);
    if (index == -1) {
        throw Exception("Invalid dimension: " + std::string(dimension) + ". Dimension must be in the layout.",
                        eStatusType::OUT_OF_BOUNDS);
    }
    return operator[](index);
}

bool TensorShape::operator==(const TensorShape &rhs) const {
    if (this->m_layout != rhs.m_layout) {
        return false;
    }

    if (this->m_size != rhs.m_size) {
        return false;
    }

    for (int32_t i = 0; i < this->m_layout.rank(); i++) {
        if (this->m_shape[i] != rhs.m_shape[i]) {
            return false;
        }
    }

    return true;
}

bool TensorShape::operator!=(const TensorShape &rhs) const { return !(*this == rhs); }

size_t TensorShape::size() const { return m_size; }

const TensorLayout &TensorShape::layout() const { return m_layout; }

const std::array<int64_t, ROCCV_TENSOR_MAX_RANK> &TensorShape::shape() const { return m_shape; }

TensorShape TensorShape::permute(const TensorLayout &layout) const {
    std::vector<int64_t> permutedShape(layout.rank());
    for (int32_t i = 0; i < layout.rank(); i++) {
        permutedShape[i] = operator[](layout.dimAt(i));
    }
    return TensorShape(layout, permutedShape);
}

}  // namespace roccv