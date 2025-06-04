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

#include "core/tensor_shape.hpp"

#include <algorithm>

#include "core/tensor_layout.hpp"

namespace roccv {
TensorShape::TensorShape(const TensorLayout &layout, const std::span<const int64_t> shape) : m_layout(layout) {
    if (shape.size() != layout.rank()) {
        throw Exception(
            "Invalid shape size: The size of the shape must match the rank of "
            "the provided layout.",
            eStatusType::OUT_OF_BOUNDS);
    }

    for (int i = 0; i < layout.rank(); i++) {
        if (shape[i] <= 0) {
            throw Exception(
                "Invalid shape dimension: values of elements in the "
                "shape array must be > 0.",
                eStatusType::OUT_OF_BOUNDS);
        }
    }

    // Copy the std::span shape into the internal shape array.
    std::copy(shape.begin(), shape.end(), m_shape.begin());

    // Calculate shape size
    m_size = 1;
    for (int64_t dim_size : shape) {
        m_size *= dim_size;
    }
}

TensorShape::TensorShape(const TensorLayout &layout, const std::initializer_list<const int64_t> shape)
    : TensorShape(layout, std::span<const int64_t>(shape.begin(), shape.end())) {}

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
        throw Exception("Invalid parameter: Index must be >= 0 and < rank.", eStatusType::OUT_OF_BOUNDS);
    }
    return m_shape[i];
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

int TensorShape::size() const { return m_size; }

const TensorLayout &TensorShape::layout() const { return m_layout; }
}  // namespace roccv