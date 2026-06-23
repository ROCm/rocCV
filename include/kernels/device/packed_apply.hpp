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

/**
 * @file packed_apply.hpp
 * @brief Wide-vector store helpers for device kernels.
 *
 * Each thread handles a PackWidth<DstType>-pixel run. Call ApplyPackedGather() when source reads are scattered
 * (gather/geometric kernels) or ApplyPackedTransform() when input and output share the same coordinates (elementwise
 * kernels). Launch with PackedGrid<DstType>() and PackedBlock().
 */

namespace Kernels::Device {

/** @brief Wide store type for a run of T pixels: `uint3` for 3-channel types, `uint4` otherwise. */
template <typename T>
using PackVec = std::conditional_t<roccv::detail::NumElements<T> == 3, uint3, uint4>;

/** @brief True when `PackVec<T>` holds a whole number of T pixels and is at least one pixel wide. */
template <typename T>
constexpr bool PackEnabled = (sizeof(PackVec<T>) >= sizeof(T)) && (sizeof(PackVec<T>) % sizeof(T) == 0);

/** @brief Number of T pixels handled per thread: `sizeof(PackVec<T>) / sizeof(T)` when packing applies, else 1. */
template <typename T>
constexpr int PackWidth = PackEnabled<T> ? static_cast<int>(sizeof(PackVec<T>) / sizeof(T)) : 1;

/** @brief Address mask for a `PackVec<T>` access: `alignof - 1`. Zero means the address meets the alignment requirement
 * (4-byte for `uint3`, 16-byte for `uint4`). */
template <typename T>
constexpr uintptr_t PackAlignMask = alignof(PackVec<T>) - 1;

/**
 * @brief Loads `PackWidth<T>` contiguous pixels as a single `PackVec<T>` transaction. Caller must ensure alignment.
 * @param row Pointer to the first pixel in the run. Must be `PackVec<T>`-aligned.
 */
template <typename T>
__device__ __forceinline__ PackVec<T> PackLoad(const T* row) {
    PackVec<T> v;
    __builtin_memcpy(&v, __builtin_assume_aligned(row, alignof(PackVec<T>)), sizeof(v));
    return v;
}

/**
 * @brief Stores `PackWidth<T>` pixels as a single `PackVec<T>` transaction. Caller must ensure alignment.
 * @param dst    Destination pointer. Must be `PackVec<T>`-aligned.
 * @param pixels Array of `PackWidth<T>` pixels to write.
 */
template <typename T>
__device__ __forceinline__ void PackStore(T* dst, const T (&pixels)[PackWidth<T>]) {
    __builtin_memcpy(__builtin_assume_aligned(dst, alignof(PackVec<T>)), &pixels[0], sizeof(pixels));
}

/**
 * @brief Extracts pixel @p i from a loaded `PackVec<T>` using a byte-offset copy.
 * @param v Packed vector loaded by `PackLoad`.
 * @param i Pixel index within the pack, in `[0, PackWidth<T>)`.
 */
template <typename T>
__device__ __forceinline__ T UnpackPixel(const PackVec<T>& v, int i) {
    T out;
    __builtin_memcpy(&out, reinterpret_cast<const unsigned char*>(&v) + i * sizeof(T), sizeof(T));
    return out;
}

/**
 * @brief Reads @p N contiguous pixels at (batch, y, x0) into @p out. The whole run must be in bounds.
 *
 * Issues a single `PackVec<Src>` load when the source pixels fill the run exactly and the row is aligned and
 * contiguous; otherwise reads per pixel through the wrapper.
 *
 * @tparam N          Number of pixels to read.
 * @tparam SrcWrapper Input wrapper (must expose `ValueType`, `at()`, `isContiguousX()`).
 * @param input  Input image wrapper.
 * @param batch  Batch index of the current row.
 * @param y      Row index of the current row.
 * @param x0     First pixel index in the run.
 * @param out    Destination array receiving the @p N pixels.
 */
template <int N, typename SrcWrapper>
__device__ __forceinline__ void PackLoadRow(SrcWrapper input, int batch, int y, int x0,
                                            typename SrcWrapper::ValueType (&out)[N]) {
    using Src = typename SrcWrapper::ValueType;
    if constexpr (PackEnabled<Src> && PackWidth<Src> == N) {
        Src* srcRow = &input.at(batch, y, x0, 0);
        if (input.isContiguousX() && (reinterpret_cast<uintptr_t>(srcRow) & PackAlignMask<Src>) == 0) {
            PackVec<Src> v = PackLoad<Src>(srcRow);
            for (int i = 0; i < N; i++) out[i] = UnpackPixel<Src>(v, i);
            return;
        }
    }
    for (int i = 0; i < N; i++) out[i] = input.at(batch, y, x0 + i, 0);
}

/**
 * @brief Writes @p N contiguous pixels from @p pack to (batch, y, x0). The whole run must be in bounds.
 *
 * Issues a single `PackVec<T>` store when packing is enabled and the row is aligned and contiguous; otherwise writes
 * per pixel through the wrapper.
 *
 * @tparam N          Number of pixels to write (must equal `PackWidth<T>` when packing is enabled).
 * @tparam DstWrapper Output wrapper (must expose `ValueType`, `at()`, `isContiguousX()`).
 * @param output Output image wrapper.
 * @param batch  Batch index of the current row.
 * @param y      Row index of the current row.
 * @param x0     First pixel index in the run.
 * @param pack   Array of @p N pixels to write.
 */
template <int N, typename DstWrapper>
__device__ __forceinline__ void PackStoreRow(DstWrapper output, int batch, int y, int x0,
                                             const typename DstWrapper::ValueType (&pack)[N]) {
    using T = typename DstWrapper::ValueType;
    if constexpr (PackEnabled<T>) {
        T* dstRow = &output.at(batch, y, x0, 0);
        if (output.isContiguousX() && (reinterpret_cast<uintptr_t>(dstRow) & PackAlignMask<T>) == 0) {
            PackStore<T>(dstRow, pack);
            return;
        }
    }
    for (int i = 0; i < N; i++) output.at(batch, y, x0 + i, 0) = pack[i];
}

/**
 * @brief Writes a `PackWidth<T>`-pixel run at (batch, y, x0) using scattered per-pixel reads and a packed store.
 *
 * @p pixelFn is called as `pixelFn(batch, y, x)` for each pixel in the run. On aligned, contiguous rows the run is
 * written as a single `PackVec<T>` store; otherwise falls back to per-pixel writes.
 *
 * @tparam DstWrapper Output wrapper (must expose `ValueType`, `width()`, `at()`, `isContiguousX()`).
 * @tparam PixelFn   `T(int batch, int y, int x)`
 * @param output   Output image wrapper.
 * @param batch    Batch index of the current row.
 * @param y        Row index of the current row.
 * @param pixelFn  Callable returning the output pixel for a given (batch, y, x).
 */
template <typename DstWrapper, typename PixelFn>
__device__ __forceinline__ void ApplyPackedGather(DstWrapper output, int batch, int y, PixelFn pixelFn) {
    using T = typename DstWrapper::ValueType;
    constexpr int NIX = PackWidth<T>;

    const int width = output.width();
    const int x0 = (blockIdx.x * blockDim.x + threadIdx.x) * NIX;
    if (x0 >= width) return;

    if (x0 + NIX - 1 < width) {
        // Fast path: the whole NIX-wide run is in bounds.
        T pack[NIX];
        for (int i = 0; i < NIX; i++) pack[i] = pixelFn(batch, y, x0 + i);
        PackStoreRow(output, batch, y, x0, pack);
    } else {
        // Tail: ragged right edge where fewer than NIX pixels remain.
        for (int i = 0; i < NIX; i++) {
            const int x = x0 + i;
            if (x >= width) break;
            output.at(batch, y, x, 0) = pixelFn(batch, y, x);
        }
    }
}

/**
 * @brief Reads a `PackWidth<Dst>`-pixel run from @p input and writes the transformed run to @p output at (batch, y).
 *
 * @p pixelFn is called as `pixelFn(srcPixel, batch, y, x)` for each pixel. Both the load and the store use packed
 * `PackVec` transactions when the row is aligned and contiguous; otherwise fall back to per-pixel accesses.
 *
 * @tparam SrcWrapper Input wrapper (must expose `ValueType`, `at()`, `isContiguousX()`).
 * @tparam DstWrapper Output wrapper (must expose `ValueType`, `width()`, `at()`, `isContiguousX()`).
 * @tparam PixelFn   `Dst(Src srcPixel, int batch, int y, int x)`
 * @param input    Input image wrapper.
 * @param output   Output image wrapper.
 * @param batch    Batch index of the current row.
 * @param y        Row index of the current row.
 * @param pixelFn  Callable returning the output pixel given the source pixel and its (batch, y, x) coordinates.
 */
template <typename SrcWrapper, typename DstWrapper, typename PixelFn>
__device__ __forceinline__ void ApplyPackedTransform(SrcWrapper input, DstWrapper output, int batch, int y,
                                                     PixelFn pixelFn) {
    using Src = typename SrcWrapper::ValueType;
    using Dst = typename DstWrapper::ValueType;
    constexpr int NIX = PackWidth<Dst>;

    const int width = output.width();
    const int x0 = (blockIdx.x * blockDim.x + threadIdx.x) * NIX;
    if (x0 >= width) return;

    if (x0 + NIX - 1 < width) {
        // Fast path: the whole NIX-wide run is in bounds.
        Src src[NIX];
        PackLoadRow(input, batch, y, x0, src);

        Dst pack[NIX];
        for (int i = 0; i < NIX; i++) pack[i] = pixelFn(src[i], batch, y, x0 + i);

        PackStoreRow(output, batch, y, x0, pack);
    } else {
        // Tail: ragged right edge where fewer than NIX pixels remain.
        for (int i = 0; i < NIX; i++) {
            const int x = x0 + i;
            if (x >= width) break;
            output.at(batch, y, x, 0) = pixelFn(input.at(batch, y, x, 0), batch, y, x);
        }
    }
}

/** @brief Default thread-block shape for packed kernels. */
__host__ inline dim3 PackedBlock() { return dim3(32, 4, 1); }

/**
 * @brief Grid dimensions for a packed kernel launch: x covers `ceil(width / (block.x * PackWidth<T>))`,
 * y covers rows, z covers batches.
 *
 * @tparam T       Output pixel type.
 * @param width   Image width in pixels.
 * @param height  Image height in pixels.
 * @param batches Batch size.
 * @param block   Thread-block dimensions, typically from `PackedBlock()`.
 */
template <typename T>
__host__ inline dim3 PackedGrid(int64_t width, int64_t height, int64_t batches, dim3 block) {
    const int64_t nix = PackWidth<T>;
    return dim3(static_cast<unsigned int>((width + block.x * nix - 1) / (block.x * nix)),
                static_cast<unsigned int>((height + block.y - 1) / block.y), static_cast<unsigned int>(batches));
}

}  // namespace Kernels::Device
