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

#pragma once

#include <hip/hip_runtime.h>
#include <stdint.h>

#include "core/detail/allocators/i_allocator.hpp"
#include "core/image_buffer.hpp"
#include "core/image_format.hpp"
#include "core/util_enums.h"

namespace roccv::detail {

/**
 * @brief Owns the per-image descriptor table (strided buffer + format arrays)
 * for an ImageBatchVarShape, along with its residency-specific lifecycle.
 *
 * GPU: a device-resident table mirrored by pinned host buffers, brought up to
 * date by a lazy H2D copy guarded by a hipEvent. CPU: a single host-resident
 * table handed straight to host kernels — no device buffers, no fence, no sync.
 *
 * All device dispatch lives here so ImageBatchVarShape stays device-agnostic and
 * gets a defaulted destructor and trivial move. Move-only: the moved-from table
 * is left empty (every pointer null) so its destructor is a no-op.
 */
class VarShapeDescriptorTable {
   public:
    /**
     * @brief The kernel-facing pointer set for one exported snapshot. `imageList`
     * and `formatList` are device pointers for a GPU table and host pointers for a
     * CPU table; `hostFormatList` is always host-resident (it aliases `formatList`
     * for a CPU table).
     */
    struct Snapshot {
        ImageBufferStrided* imageList;
        ImageFormat* formatList;
        const ImageFormat* hostFormatList;
    };

    /**
     * @brief Allocate a table sized for `capacity` images on `device`. Throws
     * INVALID_VALUE if capacity is not positive.
     */
    VarShapeDescriptorTable(int32_t capacity, eDeviceType device, const IAllocator& alloc);
    ~VarShapeDescriptorTable();

    VarShapeDescriptorTable(const VarShapeDescriptorTable&) = delete;
    VarShapeDescriptorTable& operator=(const VarShapeDescriptorTable&) = delete;
    VarShapeDescriptorTable(VarShapeDescriptorTable&&) noexcept;
    VarShapeDescriptorTable& operator=(VarShapeDescriptorTable&&) = delete;

    eDeviceType device() const noexcept { return m_device; }

    /** Host-resident mirrors, always valid (both devices) for cache rebuilds. */
    const ImageBufferStrided* hostImages() const noexcept { return m_hostImages; }
    const ImageFormat* hostFormats() const noexcept { return m_hostFormats; }

    /**
     * @brief Write descriptor slot `index` from already-validated image data. For
     * a GPU table, drains any in-flight H2D copy first so the host mirror a
     * consumer is reading never tears.
     */
    void writeSlot(int32_t index, const ImageBufferStrided& slot, ImageFormat format);

    /**
     * @brief Adjust dirty tracking after the live image count shrinks (popBack /
     * clear). Pass the new image count (0 for clear).
     */
    void onShrink(int32_t newNumImages) noexcept;

    /**
     * @brief Flush the dirty suffix [dirtyStart, numImages) to the device on
     * `stream` (GPU) or do nothing (CPU), then return the kernel-facing pointers.
     */
    Snapshot sync(hipStream_t stream, int32_t numImages);

   private:
    void freeAll() noexcept;

    eDeviceType m_device;
    const IAllocator& m_allocator;
    int32_t m_dirtyStartingFromIndex = 0;
    bool m_fencePending = false;

    ImageBufferStrided* m_devImages = nullptr;
    ImageFormat* m_devFormats = nullptr;
    ImageBufferStrided* m_hostImages = nullptr;
    ImageFormat* m_hostFormats = nullptr;
    hipEvent_t m_fence = nullptr;
};

}  // namespace roccv::detail
