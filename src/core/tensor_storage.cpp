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

#include <iostream>

#include "core/detail/context.hpp"

namespace roccv {
TensorStorage::TensorStorage(void* data, TensorStorageCleanupFunc cleanup)
    : m_data(data), m_cleanup(std::move(cleanup)) {}

TensorStorage::TensorStorage(size_t bytes, eDeviceType device, int32_t alignment)
    : TensorStorage(bytes, device, GlobalContext().getDefaultAllocator(), alignment) {}

TensorStorage::TensorStorage(size_t bytes, eDeviceType device, const IAllocator& alloc, int32_t alignment) {
    switch (device) {
        case eDeviceType::GPU:
            m_data = alloc.allocHipMem(bytes, alignment);
            m_cleanup = [&alloc](void* data) { alloc.freeHipMem(data); };
            break;
        case eDeviceType::CPU:
            m_data = alloc.allocHostMem(bytes, alignment);
            m_cleanup = [&alloc](void* data) { alloc.freeHostMem(data); };
            break;
    }
}

TensorStorage::~TensorStorage() {
    if (!m_cleanup) return;
    try {
        m_cleanup(m_data);
    } catch (const std::exception& e) {
        std::cerr << "Warning: TensorStorage cleanup function threw an exception: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "Warning: TensorStorage cleanup function threw an unknown exception." << std::endl;
    }
}

void* TensorStorage::data() const { return m_data; }
}  // namespace roccv
