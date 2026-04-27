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
 * @brief Get the block size for a 2D kernel.
 *
 * @param[in] targetBlockSize The target block size.
 * @return The block size.
 */
static inline dim3 GetBlockSize2D(int targetBlockSize = 128) {
    int deviceId;
    int warpSize;

    HIP_VALIDATE_NO_ERRORS(hipGetDevice(&deviceId));
    HIP_VALIDATE_NO_ERRORS(hipDeviceGetAttribute(&warpSize, hipDeviceAttributeWarpSize, deviceId));

    return dim3(warpSize, targetBlockSize / warpSize, 1);
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
 * @brief Get the block size for a 1D kernel — all threads on the x axis.
 *
 * Use for pointwise kernels (no neighborhood reads). There is no locality
 * benefit to grouping threads from different rows in the same block when
 * each thread only touches its own pixel; a 1D block keeps every wavefront
 * on a single contiguous row, maximizing coalescing and eliminating the
 * y-axis index math and bottom-edge tail-wave waste of a 2D launch.
 *
 * Pair with GetGridSize1D, which lays the rows of the image out along
 * gridDim.y so existing kernels can derive y directly from blockIdx.y
 * without any indexing changes (since blockDim.y == 1 collapses the
 * standard `y = blockDim.y * blockIdx.y + threadIdx.y` to `y = blockIdx.y`).
 *
 * @param[in] targetBlockSize Total threads per block. Should be a multiple
 *                            of warpSize; otherwise it is silently floored
 *                            to the nearest multiple. Defaults to 256.
 * @return The block size: dim3(targetBlockSize, 1, 1), aligned to warpSize.
 */
static inline dim3 GetBlockSize1D(int targetBlockSize = 128) {
    int deviceId;
    int warpSize;

    HIP_VALIDATE_NO_ERRORS(hipGetDevice(&deviceId));
    HIP_VALIDATE_NO_ERRORS(hipDeviceGetAttribute(&warpSize, hipDeviceAttributeWarpSize, deviceId));

    return dim3((targetBlockSize / warpSize) * warpSize, 1, 1);
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