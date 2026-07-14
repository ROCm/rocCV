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

/** Maximum number of data planes an image can have. */
#define ROCCV_MAX_IMAGE_PLANES (6)

namespace roccv {

/**
 * @brief Describes a single pitch-linear image plane.
 *
 * For interleaved-channel formats there is exactly one plane covering the whole
 * image. For planar formats (e.g. NV12, YUV420) each channel/plane carries its
 * own width, height, and row stride and lives in its own buffer.
 */
struct ImagePlaneStrided {
    /** Width of this plane in pixels. Must be >= 1. */
    int32_t width;

    /** Height of this plane in pixels. Must be >= 1. */
    int32_t height;

    /** Distance in bytes between the start of consecutive rows. Must be at
     *  least `(width * bits-per-pixel + 7) / 8`. */
    int64_t rowStride;

    /** Pointer to the first byte of plane data. Validity (device vs host) is
     *  determined by the enclosing data type. */
    void* basePtr;
};

/**
 * @brief A pitch-linear image buffer: one or more `ImagePlaneStrided` entries.
 *
 * Only the first `numPlanes` entries carry valid data; the remainder of the
 * fixed-size `planes` array is unused. Capping the array size keeps the buffer
 * trivially copyable so it can ride inside `ImageBuffer` without an
 * allocation.
 */
struct ImageBufferStrided {
    /** Number of valid planes. Must be >= 1. */
    int32_t numPlanes;

    /** Per-plane descriptors. Only the first `numPlanes` are valid. */
    ImagePlaneStrided planes[ROCCV_MAX_IMAGE_PLANES];
};

/**
 * @brief An image buffer. Currently only the strided variant is supported.
 * Mirrors the role `TensorBuffer` plays for tensors and is intentionally
 * shaped as a tagged-union-style aggregate so additional buffer kinds can be
 * added later (e.g. HIP textures) without changing the public type.
 */
struct ImageBuffer {
    ImageBufferStrided strided;
};

}  // namespace roccv
