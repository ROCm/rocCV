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

#include <functional>
#include <memory>

#include "core/hip_assert.h"

namespace roccv::detail {
static inline void StreamCallback(void* userData) {
    std::function<void()>* func = static_cast<std::function<void()>*>(userData);
    (*func)();
    delete func;
}

/**
 * @brief A non-blocking call which queues a host-side function call on the given stream. This call will execute only
 * after all previous work in the queue has completed, and will block any future events in the stream until it has
 * completed. Ensure no HIP calls are made in the callback function as doing so may cause deadlocks.
 *
 * @param[in] stream The stream to queue the function call on.
 * @param[in] cb A lambda containing the function definition. E.g. `[&]() {...}`
 */
template <typename Callable>
void LaunchHostFuncAsync(hipStream_t stream, Callable&& cb) {
    std::unique_ptr<std::function<void()>> data = std::make_unique<std::function<void()>>(std::forward<Callable>(cb));
    HIP_VALIDATE_NO_ERRORS(hipLaunchHostFunc(stream, StreamCallback, data.get()));
    data.release();  // Release ownership, StreamCallback is responsible for it now
}

/**
 * @brief Get the device's wavefront/warp size, cached per-thread.
 *
 * Re-queries the runtime if the active HIP device has changed since the
 * last call from the calling thread (so multi-device callers stay correct
 * without paying for a query on every launch).
 */
inline int CachedWarpSize() {
    static thread_local int cachedDeviceId = -1;
    static thread_local int cachedWarpSize = 0;
    int deviceId;
    HIP_VALIDATE_NO_ERRORS(hipGetDevice(&deviceId));
    if (deviceId != cachedDeviceId) {
        HIP_VALIDATE_NO_ERRORS(hipDeviceGetAttribute(&cachedWarpSize, hipDeviceAttributeWarpSize, deviceId));
        cachedDeviceId = deviceId;
    }
    return cachedWarpSize;
}

/**
 * @brief Cache hipOccupancyMaxPotentialBlockSize per (kernel, device).
 *
 * The driver picks a thread count that maximizes resident wavefronts per CU
 * for the given kernel on the current device, accounting for the kernel's
 * register and static-shared-memory usage. The result is bounded above by
 * Cap so the API's drive toward maximum occupancy can't override workload-
 * class judgment (memory-bound ops gain nothing past ~50% occupancy and
 * can lose throughput to cache pressure with overly large blocks).
 *
 * Each (Kernel, Cap) instantiation gets its own thread-local cache slot,
 * so the runtime query runs once per (kernel, device, cap) per thread.
 *
 * @tparam Kernel The __global__ function pointer (auto NTTP — each unique
 *                kernel address gets its own cached result).
 * @tparam Cap    Upper bound on the returned block size.
 */
template <auto Kernel, int Cap>
inline int CachedOccupancyBlockSize() {
    static thread_local int cachedDeviceId = -1;
    static thread_local int cachedBlockSize = 0;
    int deviceId;
    HIP_VALIDATE_NO_ERRORS(hipGetDevice(&deviceId));
    if (deviceId != cachedDeviceId) {
        int minGridSize;
        HIP_VALIDATE_NO_ERRORS(hipOccupancyMaxPotentialBlockSize(&minGridSize, &cachedBlockSize, Kernel, 0, Cap));
        cachedDeviceId = deviceId;
    }
    return cachedBlockSize;
}

/**
 * @brief Pick a 1D block size for a pointwise kernel via runtime occupancy
 *        query, capped at Cap, and cached per (kernel, device).
 *
 * Use for pointwise kernels (no neighborhood reads). The driver returns a
 * thread count tuned to this specific kernel's register pressure on the
 * current device — important on architectures with very different SIMD-per-CU
 * counts and register-file sizes (e.g. CDNA wants more wavefronts in flight
 * per CU than RDNA to hide HBM latency).
 *
 * Pair with GetGridSize1D — see its docs for the row-major launch shape.
 *
 * @tparam Kernel The __global__ function pointer.
 * @tparam Cap    Upper bound on threads per block.
 * @return dim3(blockSize, 1, 1).
 */
template <auto Kernel, int Cap = 512>
inline dim3 GetBlockSize1D() {
    return dim3(CachedOccupancyBlockSize<Kernel, Cap>(), 1, 1);
}

/**
 * @brief Pick a 2D block size for a stencil/transform kernel via runtime
 *        occupancy query, capped at Cap, and cached per (kernel, device).
 *
 * Use for kernels with 2D locality (stencils, interpolation neighborhoods,
 * affine warps). Reshapes the queried thread count as
 * (warpSize, blockSize / warpSize, 1) so threadIdx.x is wavefront-aligned
 * for coalescing while threadIdx.y stacks rows for tile-style cache reuse.
 *
 * @tparam Kernel The __global__ function pointer.
 * @tparam Cap    Upper bound on threads per block.
 * @return dim3(warpSize, blockSize / warpSize, 1).
 */
template <auto Kernel, int Cap = 1024>
inline dim3 GetBlockSize2D() {
    int blockSize = CachedOccupancyBlockSize<Kernel, Cap>();
    int warpSize = CachedWarpSize();
    return dim3(warpSize, blockSize / warpSize, 1);
}

/**
 * @brief Get the grid size for a 2D kernel.
 *
 * @param[in] width The width of the image.
 * @param[in] height The height of the image.
 * @param[in] batchSize The batch size of the image.
 * @param[in] blockSize The block size of the kernel.
 * @return The grid size.
 */
static inline dim3 GetGridSize2D(size_t width, size_t height, size_t batchSize, dim3 blockSize) {
    return dim3((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y, batchSize);
}

/**
 * @brief Get the grid size for a 1D-row-major launch.
 *
 * Lays one image row along gridDim.y and the batch along gridDim.z. The
 * block from GetBlockSize1D has blockDim.y == 1, so the kernel's standard
 * `y = blockDim.y * blockIdx.y + threadIdx.y` collapses to `y = blockIdx.y`
 * with no kernel changes required.
 *
 * Caller is responsible for ensuring height does not exceed the device's
 * gridDim.y limit (typically 65535) and batchSize does not exceed gridDim.z.
 *
 * @param[in] width      The width of the image.
 * @param[in] height     The height of the image (becomes gridDim.y).
 * @param[in] batchSize  The number of images (becomes gridDim.z).
 * @param[in] blockSize  Block size from GetBlockSize1D.
 * @return The grid size: dim3(ceil(width / blockSize.x), height, batchSize).
 */
static inline dim3 GetGridSize1D(size_t width, size_t height, size_t batchSize, dim3 blockSize) {
    return dim3((width + blockSize.x - 1) / blockSize.x, height, batchSize);
}

}  // namespace roccv::detail
