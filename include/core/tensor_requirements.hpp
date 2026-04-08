/**
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#pragma once

#include <array>

#include "core/tensor_layout.hpp"
#include "core/util_enums.h"

namespace roccv {

/**
 * @brief Specifies basic memory requirements for an allocation.
 *
 * This struct expresses the number of bytes required for a memory region that backs
 * a particular tensor or buffer allocation. It is typically used to indicate the raw
 * size required for device, host, or pinned memory allocations.
 */
struct MemRequirements {
    size_t bytes = 0;
};

/**
 * @brief Specifies resource requirements for tensor memory allocations.
 *
 * This struct aggregates requirements for different types of memory resources
 * that may be used for tensor allocation and operation:
 * - deviceMem: Memory required on the device (e.g., GPU).
 * - hostMem: Memory required on the host (CPU-accessible memory).
 * - pinnedMem: Memory required in pinned (page-locked) host memory, which may
 *              be used for efficient device-host transfers.
 */
struct ResourceRequirements {
    MemRequirements deviceMem;
    MemRequirements hostMem;
    MemRequirements pinnedMem;
};

/**
 * @brief Specifies the requirements for creating and allocating a tensor.
 *
 * This struct defines all the necessary properties for specifying the memory and layout
 * requirements of a tensor, including its datatype, shape, memory alignment, layout, strides,
 * memory resource requirements, and device placement.
 */
struct TensorRequirements {
    eDataType dtype;                                     // Data type
    eTensorLayout layout;                                // Tensor layout
    int32_t rank;                                        // Number of dimensions
    std::array<int64_t, ROCCV_TENSOR_MAX_RANK> shape;    // Shape in elements
    std::array<int64_t, ROCCV_TENSOR_MAX_RANK> strides;  // Strides in bytes
    int32_t alignBytes;                                  // Base address alignment in bytes
    ResourceRequirements res;                            // Resource requirements for memory allocation
    eDeviceType device;                                  // Device type
};
}  // namespace roccv