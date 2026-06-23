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

#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>
#include <type_traits>

#include "core/detail/type_traits.hpp"

namespace Kernels::Device {

/**
 * @brief Wide store type used to write a run of output pixels in a single transaction.
 *
 * A 3-element pixel (e.g. uchar3) is written as a uint3 (12 bytes); everything else uses uint4 (16 bytes). This packs
 * several small pixels (uchar1/uchar3/uchar4/float1) into one aligned vector store, which removes the sub-word store
 * penalty those types otherwise pay. Wide pixels (float3/float4) already fill a transaction, so their pack collapses
 * to a single pixel (see ResizeNIX).
 */
template <typename T>
using ResizePack = std::conditional_t<roccv::detail::NumElements<T> == 3, uint3, uint4>;

/**
 * @brief Number of output pixels each thread produces: how many pixels of type T fit in one ResizePack.
 */
template <typename T>
constexpr int ResizeNIX = sizeof(ResizePack<T>) / sizeof(T);

/**
 * @brief Byte-alignment required to issue a ResizePack store. A uint3 is only 4-byte aligned (three uints), while a
 * uint4 is 16-byte aligned.
 */
template <typename T>
constexpr uintptr_t ResizePackAlignMask =
    (sizeof(ResizePack<T>) == sizeof(uint3) ? sizeof(unsigned int) : sizeof(ResizePack<T>)) - 1;

/**
 * @brief Resizes an image using the interpolation/border logic baked into the source wrapper.
 *
 * Each thread emits a contiguous run of NIX output pixels along x. When the run is fully in-bounds and the destination
 * row is suitably aligned, the run is written with a single ResizePack vector store; otherwise it falls back to
 * per-pixel stores. Reads and interpolation are unchanged from the scalar path, so results are bit-identical.
 *
 * @tparam NIX Output pixels produced per thread (sizeof(DstPack)/sizeof(T)).
 * @tparam DstPack Vector store type used for the coalesced write.
 */
template <int NIX, typename DstPack, typename SrcWrapper, typename DstWrapper>
__global__ void resize(SrcWrapper input, DstWrapper output, float scaleX, float scaleY) {
    using T = typename DstWrapper::ValueType;

    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch = blockIdx.z;
    if (y >= output.height()) return;

    const int x0 = (blockIdx.x * blockDim.x + threadIdx.x) * NIX;
    if (x0 >= output.width()) return;

    const float srcY = fmaf(y + 0.5f, scaleY, -0.5f);

    if (x0 + NIX - 1 < output.width()) {
        // Fast path: the whole NIX-wide run is in bounds.
        T pack[NIX];
#pragma unroll
        for (int i = 0; i < NIX; i++) {
            const float srcX = fmaf(x0 + i + 0.5f, scaleX, -0.5f);
            pack[i] = input.at(batch, srcY, srcX, 0);
        }

        T* dstRow = &output.at(batch, y, x0, 0);
        // Alignment is identical for every thread writing a given row, so this branch is warp-uniform.
        if ((reinterpret_cast<uintptr_t>(dstRow) & ResizePackAlignMask<T>) == 0) {
            *reinterpret_cast<DstPack*>(dstRow) = reinterpret_cast<const DstPack&>(pack);
        } else {
#pragma unroll
            for (int i = 0; i < NIX; i++) dstRow[i] = pack[i];
        }
    } else {
        // Tail: ragged right edge where fewer than NIX pixels remain.
#pragma unroll
        for (int i = 0; i < NIX; i++) {
            const int x = x0 + i;
            if (x >= output.width()) break;
            const float srcX = fmaf(x + 0.5f, scaleX, -0.5f);
            output.at(batch, y, x, 0) = input.at(batch, srcY, srcX, 0);
        }
    }
}
}  // namespace Kernels::Device