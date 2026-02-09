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
 * @brief Get the maximum potential block size for a 2D kernel.
 *
 * @param[in] kernel The kernel function to get the maximum potential block size for.
 * @param[in] sharedMemSizePerBlock The shared memory size per block.
 * @return The maximum potential block size.
 */
template <typename KernalFunc>
dim3 GetMaximumPotentialBlockSize2D(KernalFunc kernel, size_t sharedMemSizePerBlock) {
    int minimumGridSize;
    int blockSize;
    int deviceId;
    int warpSize;

    HIP_VALIDATE_NO_ERRORS(hipGetDevice(&deviceId));
    HIP_VALIDATE_NO_ERRORS(hipDeviceGetAttribute(&warpSize, hipDeviceAttributeWarpSize, deviceId));
    HIP_VALIDATE_NO_ERRORS(
        hipOccupancyMaxPotentialBlockSize(&minimumGridSize, &blockSize, kernel, sharedMemSizePerBlock, warpSize));

    return dim3(warpSize, blockSize / warpSize, 1);
}
}  // namespace roccv::detail