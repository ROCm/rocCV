/*
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
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

#include "core/detail/var_shape_descriptor_table.hpp"

#include <algorithm>

#include "core/exception.hpp"
#include "core/hip_assert.h"

namespace roccv::detail {

VarShapeDescriptorTable::VarShapeDescriptorTable(int32_t capacity, eDeviceType device, const IAllocator& alloc)
    : m_device(device), m_allocator(alloc) {
    if (capacity <= 0) {
        throw Exception("ImageBatchVarShape capacity must be positive", eStatusType::INVALID_VALUE);
    }

    const size_t imagesBytes = sizeof(ImageBufferStrided) * capacity;
    const size_t formatsBytes = sizeof(ImageFormat) * capacity;

    try {
        if (m_device == eDeviceType::GPU) {
            // Device descriptor table + pinned host mirrors, kept in sync by the
            // lazy H2D copy in sync() and guarded by m_fence.
            m_devImages = static_cast<ImageBufferStrided*>(m_allocator.allocHipMem(imagesBytes));
            m_devFormats = static_cast<ImageFormat*>(m_allocator.allocHipMem(formatsBytes));
            m_hostImages = static_cast<ImageBufferStrided*>(m_allocator.allocHostPinnedMem(imagesBytes));
            m_hostFormats = static_cast<ImageFormat*>(m_allocator.allocHostPinnedMem(formatsBytes));

            HIP_VALIDATE_NO_ERRORS(hipEventCreateWithFlags(&m_fence, hipEventDisableTiming));
        } else {
            // A single host-resident table handed straight to host kernels: no
            // device buffers, no pinned memory, no fence.
            m_hostImages = static_cast<ImageBufferStrided*>(m_allocator.allocHostMem(imagesBytes));
            m_hostFormats = static_cast<ImageFormat*>(m_allocator.allocHostMem(formatsBytes));
        }
    } catch (...) {
        freeAll();
        throw;
    }
}

VarShapeDescriptorTable::~VarShapeDescriptorTable() {
    if (m_fencePending && m_fence != nullptr) {
        // Drain any in-flight H2D copy before freeing the host mirrors it reads
        // from. (void) — destructors must not throw.
        (void)hipEventSynchronize(m_fence);
    }
    if (m_fence != nullptr) {
        (void)hipEventDestroy(m_fence);
    }
    freeAll();
}

VarShapeDescriptorTable::VarShapeDescriptorTable(VarShapeDescriptorTable&& other) noexcept
    : m_device(other.m_device),
      m_allocator(other.m_allocator),
      m_dirtyStartingFromIndex(other.m_dirtyStartingFromIndex),
      m_fencePending(other.m_fencePending),
      m_devImages(other.m_devImages),
      m_devFormats(other.m_devFormats),
      m_hostImages(other.m_hostImages),
      m_hostFormats(other.m_hostFormats),
      m_fence(other.m_fence) {
    other.m_dirtyStartingFromIndex = 0;
    other.m_fencePending = false;
    other.m_devImages = nullptr;
    other.m_devFormats = nullptr;
    other.m_hostImages = nullptr;
    other.m_hostFormats = nullptr;
    other.m_fence = nullptr;
}

void VarShapeDescriptorTable::freeAll() noexcept {
    // The host mirrors are pinned for a GPU table and plain host memory for a CPU
    // table, so only their free path differs. The device frees are null-guarded,
    // so a CPU table (whose device pointers are null) skips them.
    if (m_device == eDeviceType::GPU) {
        if (m_hostFormats != nullptr) m_allocator.freeHostPinnedMem(m_hostFormats);
        if (m_hostImages != nullptr) m_allocator.freeHostPinnedMem(m_hostImages);
    } else {
        if (m_hostFormats != nullptr) m_allocator.freeHostMem(m_hostFormats);
        if (m_hostImages != nullptr) m_allocator.freeHostMem(m_hostImages);
    }
    if (m_devFormats != nullptr) m_allocator.freeHipMem(m_devFormats);
    if (m_devImages != nullptr) m_allocator.freeHipMem(m_devImages);
}

void VarShapeDescriptorTable::writeSlot(int32_t index, const ImageBufferStrided& slot, ImageFormat format) {
    if (m_fencePending) {
        HIP_VALIDATE_NO_ERRORS(hipEventSynchronize(m_fence));
        m_fencePending = false;
    }
    m_hostImages[index] = slot;
    m_hostFormats[index] = format;
}

void VarShapeDescriptorTable::onShrink(int32_t newNumImages) noexcept {
    m_dirtyStartingFromIndex = std::min(m_dirtyStartingFromIndex, newNumImages);
}

VarShapeDescriptorTable::Snapshot VarShapeDescriptorTable::sync(hipStream_t stream, int32_t numImages) {
    // CPU tables have a single host table — nothing to copy. Only a GPU table with
    // a dirty suffix issues an H2D copy and records the fence.
    if (m_device == eDeviceType::GPU && m_dirtyStartingFromIndex < numImages) {
        const int32_t dirtyCount = numImages - m_dirtyStartingFromIndex;

        if (m_fencePending) {
            HIP_VALIDATE_NO_ERRORS(hipStreamWaitEvent(stream, m_fence, /*flags=*/0));
        }

        HIP_VALIDATE_NO_ERRORS(hipMemcpyAsync(m_devImages + m_dirtyStartingFromIndex,
                                              m_hostImages + m_dirtyStartingFromIndex,
                                              sizeof(ImageBufferStrided) * dirtyCount, hipMemcpyHostToDevice, stream));
        HIP_VALIDATE_NO_ERRORS(hipMemcpyAsync(m_devFormats + m_dirtyStartingFromIndex,
                                              m_hostFormats + m_dirtyStartingFromIndex,
                                              sizeof(ImageFormat) * dirtyCount, hipMemcpyHostToDevice, stream));

        HIP_VALIDATE_NO_ERRORS(hipEventRecord(m_fence, stream));
        m_fencePending = true;
    }
    m_dirtyStartingFromIndex = numImages;

    if (m_device == eDeviceType::GPU) {
        return Snapshot{m_devImages, m_devFormats, m_hostFormats};
    }
    // CPU: imageList/formatList are the host table; hostFormatList aliases it.
    return Snapshot{m_hostImages, m_hostFormats, m_hostFormats};
}

}  // namespace roccv::detail
