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

#include "core/tensor_storage.hpp"

#include "core/detail/context.hpp"
#include "core/hip_assert.h"

namespace roccv {
TensorStorage::TensorStorage(void* data, eDeviceType device, eOwnership ownership)
    : TensorStorage(data, device, GlobalContext().getDefaultAllocator(), ownership) {}

TensorStorage::TensorStorage(void* data, eDeviceType device, const IAllocator& alloc, eOwnership ownership)
    : m_device(device), m_ownership(ownership), m_data(data), m_allocator(alloc) {}

TensorStorage::TensorStorage(size_t bytes, eDeviceType device)
    : TensorStorage(bytes, device, GlobalContext().getDefaultAllocator()) {}

TensorStorage::TensorStorage(size_t bytes, eDeviceType device, const IAllocator& alloc)
    : m_device(device), m_ownership(eOwnership::OWNING), m_allocator(alloc) {
    switch (m_device) {
        case eDeviceType::GPU:
            m_data = m_allocator.allocHipMem(bytes);
            break;
        case eDeviceType::CPU:
            m_data = m_allocator.allocHostMem(bytes);
            break;
    }
}

TensorStorage::~TensorStorage() {
    if (m_ownership == eOwnership::VIEW) return;

    switch (m_device) {
        case eDeviceType::GPU:
            m_allocator.freeHipMem(m_data);
            break;
        case eDeviceType::CPU:
            m_allocator.freeHostMem(m_data);
            break;
    }
}

void* TensorStorage::data() const { return m_data; }

const eDeviceType TensorStorage::device() const { return m_device; }
}  // namespace roccv
