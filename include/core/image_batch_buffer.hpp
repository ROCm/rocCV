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

#include <stdint.h>

#include "core/image_buffer.hpp"
#include "core/image_format.hpp"

namespace roccv {

/**
 * @brief Pitch-linear descriptor table for a variable-shape image batch.
 *
 * Each entry of `imageList` is a full per-image strided buffer descriptor —
 * reusing `ImageBufferStrided` keeps the per-image shape (multi-plane-capable,
 * one base pointer per plane, per-plane row stride) identical to what a single
 * `Image` carries today.
 *
 * Pointer residency:
 *  - `imageList` is the descriptor table read by GPU kernels. For a GPU-resident
 *    batch this points into device memory; for a hypothetical CPU-resident
 *    batch it would point into host memory. The producing batch class owns the
 *    allocation and decides residency.
 *  - `formatList` mirrors `imageList`'s residency and holds one ImageFormat per
 *    image (so kernels can branch on per-image format without dereferencing the
 *    descriptor table).
 *  - `hostFormatList` is always host-resident. It exists so host-side validation
 *    code can read per-image formats without paying a D->H copy. For a
 *    CPU-resident batch this MAY alias `formatList`; for a GPU-resident batch
 *    it is a separate host mirror kept in sync by the producer.
 *
 * `uniqueFormat` is the common ImageFormat across all images, or FMT_NONE if
 * formats are heterogeneous or the batch is empty. Cached to fast-path the
 * homogeneous case.
 *
 * `maxWidth` / `maxHeight` are the bounding box across all images. Used by
 * operators to size launch grids. Both are 0 when the batch is empty.
 *
 * The struct is intentionally trivially copyable so it can ride inside
 * `ImageBatchBuffer` without an allocation, mirroring `ImageBufferStrided`'s
 * relationship to `ImageBuffer`.
 */
struct ImageBatchVarShapeBufferStrided {
    /** Common format across all images in the batch, or a default-constructed
     *  ImageFormat if formats are heterogeneous or the batch is empty. */
    ImageFormat uniqueFormat;

    /** Bounding box across all images, in pixels. Both 0 when empty. */
    int32_t maxWidth;
    int32_t maxHeight;

    /** Per-image format array, length == numImages. Residency matches
     *  `imageList` (device for GPU batches, host for CPU batches). */
    ImageFormat* formatList;

    /** Host-resident mirror of `formatList`. May alias `formatList` for
     *  CPU-resident batches. Length == numImages. */
    const ImageFormat* hostFormatList;

    /** Per-image descriptor table, length == numImages. The kernel-facing
     *  pointer; residency determines which device the batch lives on. */
    ImageBufferStrided* imageList;
};

/**
 * @brief An image-batch buffer. Currently only the variable-shape strided
 * variant is supported. Shaped as a tagged-union-style aggregate so additional
 * batch buffer kinds can be added later (e.g. tensor-backed batches) without
 * changing the public type.
 */
struct ImageBatchBuffer {
    ImageBatchVarShapeBufferStrided varShapeStrided;
};

}  // namespace roccv
