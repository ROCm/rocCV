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

#include <array>
#include <cstdint>
#include <tuple>
#include <type_traits>
#include <utility>

#include "core/detail/type_traits.hpp"

/**
 * @file packed_apply.hpp
 * @brief Reusable vector-pack helpers for per-output-pixel ("elementwise") device kernels.
 *
 * Many operators compute one output pixel independently from the input(s) at the same (or a per-pixel-derived) source
 * coordinate. When the output is an interleaved image, consecutive pixels along x are contiguous in memory, so several
 * of them can be written with a single wide, aligned store. This removes the sub-word store penalty paid by small pixel
 * types (notably 3-byte uchar3) and amortizes per-thread setup, which is a large win for these memory-bound kernels.
 *
 * Usage: a kernel computes its row/batch indices, bounds-checks them, then calls ApplyPackedGather() (scattered reads,
 * packed store) or ApplyPackedTransform() (packed load and store) with a callable that returns the output pixel. The
 * grid must be launched with PackedGrid<DstType>() so that each thread's x-index maps to a run of PackWidth<DstType>
 * pixels.
 */

namespace Kernels::Device {

/**
 * @brief Wide store type used to write a run of output pixels in one transaction. A 3-element pixel is written as a
 * uint3 (12 bytes); everything else uses uint4 (16 bytes).
 */
template <typename T>
using PackVec = std::conditional_t<roccv::detail::NumElements<T> == 3, uint3, uint4>;

/**
 * @brief Whether a run of T pixels can be written as a single PackVec<T> store: the pack must hold a whole number of
 * pixels and be at least one pixel wide. This excludes pixel types as large as or larger than the pack (e.g. double3 at
 * 24 bytes vs a 12-byte uint3), which simply fall back to per-pixel stores.
 */
template <typename T>
constexpr bool PackEnabled = (sizeof(PackVec<T>) >= sizeof(T)) && (sizeof(PackVec<T>) % sizeof(T) == 0);

/**
 * @brief Number of pixels of type T produced per thread. When packing applies this is how many pixels fit in one
 * PackVec<T> (uchar1->16, uchar3->4, uchar4->4, float1->4, float3->1, float4->1); otherwise it is 1.
 */
template <typename T>
constexpr int PackWidth = PackEnabled<T> ? static_cast<int>(sizeof(PackVec<T>) / sizeof(T)) : 1;

/**
 * @brief Byte alignment required to issue a PackVec<T> store. A uint3 is only 4-byte aligned (three uints); a uint4 is
 * 16-byte aligned.
 */
template <typename T>
constexpr uintptr_t PackAlignMask =
    (sizeof(PackVec<T>) == sizeof(uint3) ? sizeof(unsigned int) : sizeof(PackVec<T>)) - 1;

/**
 * @brief Explicit, type-punning-safe pack/unpack primitives.
 *
 * These exist so the wide-vector load/store is expressed as a defined byte copy rather than a glvalue access through an
 * unrelated type. `__builtin_memcpy` of a compile-time-constant size lowers to a single wide transaction while keeping
 * strict-aliasing intact; `__builtin_assume_aligned` is what lets the compiler widen the access (a bare uchar3* is only
 * 1-byte aligned and would otherwise emit byte-wise transfers). Callers must still gate these on PackAlignMask<T> — the
 * builtins fix the type-punning, not the alignment precondition.
 */

/**
 * @brief Loads a run of PackWidth<T> contiguous pixels at @p row as one wide PackVec<T> transaction.
 */
template <typename T>
__device__ __forceinline__ PackVec<T> PackLoad(const T* row) {
    PackVec<T> v;
    __builtin_memcpy(&v, __builtin_assume_aligned(row, alignof(PackVec<T>)), sizeof(v));
    return v;
}

/**
 * @brief Stores a run of PackWidth<T> pixels to @p dst as one wide PackVec<T> transaction.
 */
template <typename T>
__device__ __forceinline__ void PackStore(T* dst, const T (&pixels)[PackWidth<T>]) {
    __builtin_memcpy(__builtin_assume_aligned(dst, alignof(PackVec<T>)), &pixels[0], sizeof(pixels));
}

/**
 * @brief Extracts pixel @p i from a loaded PackVec<T>. Uses a byte-offset copy, so it is correct even when pixels do
 * not align to the 32-bit lanes of the pack (e.g. 3-byte uchar3 pixels straddling uint3 lane boundaries).
 */
template <typename T>
__device__ __forceinline__ T UnpackPixel(const PackVec<T>& v, int i) {
    T out;
    __builtin_memcpy(&out, reinterpret_cast<const unsigned char*>(&v) + i * sizeof(T), sizeof(T));
    return out;
}

/**
 * @brief Produces a contiguous run of PackWidth<T> output pixels at row (batch, y) and writes them. Use this for
 * "gather" kernels whose source reads are scattered by interpolation/border math and therefore cannot be widened — only
 * the store is packed.
 *
 * Each pixel is computed by @p pixelFn(batch, y, x). When the run is fully in bounds, the destination is contiguous in
 * x, and the row is suitably aligned, the run is written with a single PackVec<T> vector store; otherwise it falls back
 * to coordinate-correct per-pixel stores. The fallback paths use the wrapper's own indexing, so the result is correct
 * for any layout/stride — only the fast path requires contiguity, which it checks at runtime.
 *
 * @tparam DstWrapper Output wrapper type (must expose ValueType, width(), at(), and isContiguousX()).
 * @tparam PixelFn Callable with signature `T(int batch, int y, int x)` returning the output pixel.
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

        if constexpr (PackEnabled<T>) {
            T* dstRow = &output.at(batch, y, x0, 0);
            // Alignment/contiguity is identical for every thread writing a given row, so this branch is warp-uniform.
            if (output.isContiguousX() && (reinterpret_cast<uintptr_t>(dstRow) & PackAlignMask<T>) == 0) {
                PackStore<T>(dstRow, pack);
            } else {
                for (int i = 0; i < NIX; i++) output.at(batch, y, x0 + i, 0) = pack[i];
            }
        } else {
            for (int i = 0; i < NIX; i++) output.at(batch, y, x0 + i, 0) = pack[i];
        }
    } else {
        // Tail: ragged right edge where fewer than NIX pixels remain.

        for (int i = 0; i < NIX; i++) {
            const int x = x0 + i;
            if (x >= width) break;
            output.at(batch, y, x, 0) = pixelFn(batch, y, x);
        }
    }
}

namespace detail {

/**
 * @brief Reads a run of NIX consecutive pixels from one input wrapper at (batch, y, x0 .. x0+NIX-1). The load is
 * widened to a single PackVec<Src> transaction when the input's pack width matches the run and the row is contiguous
 * and aligned; otherwise each pixel is read individually. Inputs whose pack width does not match NIX (e.g. a
 * type-changing or differently-shaped input) always take the per-pixel path.
 */
template <int NIX, typename SrcWrapper>
__device__ __forceinline__ std::array<typename SrcWrapper::ValueType, NIX> ReadPackedRun(SrcWrapper input, int batch,
                                                                                         int y, int x0) {
    using Src = typename SrcWrapper::ValueType;
    std::array<Src, NIX> run;

    constexpr bool kPackedRead = PackEnabled<Src> && (PackWidth<Src> == NIX);
    if constexpr (kPackedRead) {
        Src* srcRow = &input.at(batch, y, x0, 0);
        // Warp-uniform: contiguity/alignment is identical for every thread reading a given row.
        if (input.isContiguousX() && (reinterpret_cast<uintptr_t>(srcRow) & PackAlignMask<Src>) == 0) {
            PackVec<Src> v = PackLoad<Src>(srcRow);
            for (int i = 0; i < NIX; i++) run[i] = UnpackPixel<Src>(v, i);
        } else {
            for (int i = 0; i < NIX; i++) run[i] = input.at(batch, y, x0 + i, 0);
        }
    } else {
        for (int i = 0; i < NIX; i++) run[i] = input.at(batch, y, x0 + i, 0);
    }
    return run;
}

/**
 * @brief Invokes pixelFn(batch, y, x, run0[i], run1[i], ...) by expanding the tuple of per-input runs at element i.
 */
template <typename PixelFn, typename Runs, std::size_t... Is>
__device__ __forceinline__ auto InvokeAtIndex(PixelFn& pixelFn, int batch, int y, int x, int i, Runs& runs,
                                              std::index_sequence<Is...>) {
    return pixelFn(batch, y, x, std::get<Is>(runs)[i]...);
}

}  // namespace detail

/**
 * @brief Transforms a contiguous run of PackWidth<Dst> pixels at row (batch, y), reading one or more inputs at the same
 * coordinates. Use this for "elementwise" kernels where the output pixel is a pure function of the input pixel(s) at
 * the matching coordinate — both the loads and the store can be packed.
 *
 * Each output pixel is computed by @p pixelFn(batch, y, x, p0, p1, ...), where pN is the pixel from the Nth input at
 * (batch, y, x). The callable receives the coordinates as well, so it can still fetch any secondary/broadcast inputs
 * (which are not contiguous along x) itself. Each input is independently widened to a single PackVec load only when its
 * pixels are contiguous, aligned, and its pack width matches the output run (the common same-pixel-size case);
 * otherwise that input is read per pixel. The store side follows the same packed/fallback logic as ApplyPackedGather.
 *
 * @tparam DstWrapper Output wrapper type (must expose ValueType, width(), at(), and isContiguousX()).
 * @tparam PixelFn Callable `Dst(int batch, int y, int x, Src0 p0, Src1 p1, ...)` returning the output pixel.
 * @tparam SrcWrappers Input wrapper types (each must expose ValueType, at(), and isContiguousX()).
 */
template <typename DstWrapper, typename PixelFn, typename... SrcWrappers>
__device__ __forceinline__ void ApplyPackedTransform(DstWrapper output, int batch, int y, PixelFn pixelFn,
                                                     SrcWrappers... inputs) {
    using Dst = typename DstWrapper::ValueType;
    constexpr int NIX = PackWidth<Dst>;

    const int width = output.width();
    const int x0 = (blockIdx.x * blockDim.x + threadIdx.x) * NIX;
    if (x0 >= width) return;

    if (x0 + NIX - 1 < width) {
        // Fast path: read each input's NIX-wide run (packed per-input where possible), compute, then packed-store.
        auto runs = std::make_tuple(detail::ReadPackedRun<NIX>(inputs, batch, y, x0)...);

        Dst pack[NIX];
        for (int i = 0; i < NIX; i++) {
            pack[i] =
                detail::InvokeAtIndex(pixelFn, batch, y, x0 + i, i, runs, std::index_sequence_for<SrcWrappers...>{});
        }

        if constexpr (PackEnabled<Dst>) {
            Dst* dstRow = &output.at(batch, y, x0, 0);
            if (output.isContiguousX() && (reinterpret_cast<uintptr_t>(dstRow) & PackAlignMask<Dst>) == 0) {
                PackStore<Dst>(dstRow, pack);
            } else {
                for (int i = 0; i < NIX; i++) output.at(batch, y, x0 + i, 0) = pack[i];
            }
        } else {
            for (int i = 0; i < NIX; i++) output.at(batch, y, x0 + i, 0) = pack[i];
        }
    } else {
        // Tail: ragged right edge where fewer than NIX pixels remain.

        for (int i = 0; i < NIX; i++) {
            const int x = x0 + i;
            if (x >= width) break;
            output.at(batch, y, x, 0) = pixelFn(batch, y, x, inputs.at(batch, y, x, 0)...);
        }
    }
}

/**
 * @brief Default thread-block shape for packed elementwise kernels.
 */
__host__ inline dim3 PackedBlock() { return dim3(32, 4, 1); }

/**
 * @brief Grid for a packed elementwise launch: the x dimension covers ceil(width / (block.x * PackWidth<T>)) so each
 * thread handles a PackWidth<T>-pixel run; y/z map to rows and batches as usual.
 *
 * @tparam T Output pixel type (determines the pack width).
 */
template <typename T>
__host__ inline dim3 PackedGrid(int64_t width, int64_t height, int64_t batches, dim3 block) {
    const int64_t nix = PackWidth<T>;
    return dim3(static_cast<unsigned int>((width + block.x * nix - 1) / (block.x * nix)),
                static_cast<unsigned int>((height + block.y - 1) / block.y), static_cast<unsigned int>(batches));
}

}  // namespace Kernels::Device
