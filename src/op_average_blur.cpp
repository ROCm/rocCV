/**
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

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

#include "op_average_blur.hpp"

#include <hip/hip_runtime.h>

#include <cfenv>
#include <cmath>
#include <functional>

#include "common/validation_helpers.hpp"
#include "core/detail/casting.hpp"
#include "details/filter_2d.hpp"

namespace roccv {
AverageBlur::AverageBlur(int32_t maxKernelWidth, int32_t maxKernelHeight)
    : m_maxKernelWidth(maxKernelWidth), m_maxKernelHeight(maxKernelHeight) {
    if (maxKernelWidth <= 0) {
        throw roccv::Exception(
            "Invalid maxKernelWidth = " + std::to_string(maxKernelWidth) + ": Ensure that it is positive.",
            eStatusType::INVALID_VALUE);
    }
    if (maxKernelHeight <= 0) {
        throw roccv::Exception(
            "Invalid maxKernelHeight = " + std::to_string(maxKernelHeight) + ": Ensure that it is positive.",
            eStatusType::INVALID_VALUE);
    }

    size_t memSize = maxKernelHeight * maxKernelWidth * sizeof(float);
    size_t memSizeH = m_maxKernelWidth * sizeof(float);
    size_t memSizeV = m_maxKernelHeight * sizeof(float);

    if (maxKernelHeight <= 3 && maxKernelWidth <= 3) {
        m_use2D = true;
        m_hostKernelMem = static_cast<float*>(m_allocator.allocHostPinnedMem(memSize));
    } else {
        m_hostKernelMemH = static_cast<float*>(m_allocator.allocHostPinnedMem(memSizeH));
        m_hostKernelMemV = static_cast<float*>(m_allocator.allocHostPinnedMem(memSizeV));
    }

    int gpuCount = 0;
    hipError_t err = hipGetDeviceCount(&gpuCount);
    if (err == hipSuccess) {
        if (m_use2D) {
            m_deviceKernelMem = static_cast<float*>(m_allocator.allocHipMem(memSize));
        } else {
            m_deviceKernelMemH = static_cast<float*>(m_allocator.allocHipMem(memSizeH));
            m_deviceKernelMemV = static_cast<float*>(m_allocator.allocHipMem(memSizeV));
        }
        HIP_VALIDATE_NO_ERRORS(hipEventCreateWithFlags(&m_completionEvent, hipEventDisableTiming));
    }
}

AverageBlur::~AverageBlur() {
    if (m_use2D) {
        m_allocator.freeHostPinnedMem(m_hostKernelMem);
    } else {
        m_allocator.freeHostPinnedMem(m_hostKernelMemH);
        m_allocator.freeHostPinnedMem(m_hostKernelMemV);
    }
    if (m_deviceKernelMem != nullptr) {
        m_allocator.freeHipMem(m_deviceKernelMem);
    }
    if (m_deviceKernelMemH != nullptr) {
        m_allocator.freeHipMem(m_deviceKernelMemH);
    }
    if (m_deviceKernelMemV != nullptr) {
        m_allocator.freeHipMem(m_deviceKernelMemV);
    }
    if (m_completionEvent != nullptr) {
        (void)hipEventDestroy(m_completionEvent);
    }
}

void AverageBlur::operator()(hipStream_t stream, const Tensor& input, Tensor& output, int kernelWidth, int kernelHeight,
                             int anchorX, int anchorY, eBorderType borderMode, eDeviceType device) {
    // Validate input tensor
    CHECK_TENSOR_DEVICE(input, device);
    CHECK_TENSOR_DATATYPES(input, DATA_TYPE_U8, DATA_TYPE_U16, DATA_TYPE_S16, DATA_TYPE_S32, DATA_TYPE_F32);
    CHECK_TENSOR_LAYOUT(input, TENSOR_LAYOUT_HWC, TENSOR_LAYOUT_NHWC);
    CHECK_TENSOR_CHANNELS(input, 1, 3, 4);

    // Validate output tensor
    CHECK_TENSOR_COMPARISON(input.dtype() == output.dtype());
    CHECK_TENSOR_COMPARISON(input.device() == output.device());
    CHECK_TENSOR_COMPARISON(input.shape() == output.shape());

    // Validate kernel size
    if (!(kernelWidth > 0 && kernelWidth % 2 == 1 && kernelWidth <= m_maxKernelWidth && kernelHeight > 0 &&
          kernelHeight % 2 == 1 && kernelHeight <= m_maxKernelHeight)) {
        throw roccv::Exception("Invalid kernel size = " + std::to_string(kernelWidth) + ", " +
                                   std::to_string(kernelHeight) +
                                   ": Ensure that the kernel size is odd, positive, and less than the max:" +
                                   std::to_string(m_maxKernelWidth) + ", " + std::to_string(m_maxKernelHeight),
                               eStatusType::INVALID_VALUE);
    }

    // Validate anchor
    if (!((anchorX == -1 || (anchorX >= 0 && anchorX < kernelWidth)) &&
          (anchorY == -1 || (anchorY >= 0 && anchorY < kernelHeight)))) {
        throw roccv::Exception("Invalid anchor = " + std::to_string(anchorX) + ", " + std::to_string(anchorY) +
                                   ": Ensure that the anchorX and anchorY are -1, or nonnegative and less than "
                                   "kernelWidth and kernelHeight, respectively:" +
                                   std::to_string(kernelWidth) + ", " + std::to_string(kernelHeight),
                               eStatusType::INVALID_VALUE);
    }
    processAnchor(anchorX, anchorY, kernelWidth, kernelHeight);

    std::lock_guard<std::mutex> lock(m_bufferMutex);
    if (device == eDeviceType::GPU) {
        HIP_VALIDATE_NO_ERRORS(hipEventSynchronize(m_completionEvent));
    }

    // Compute the kernel
    if (m_use2D) {
        float val = 1.0 / (kernelWidth * kernelHeight);
        for (int y = 0; y < kernelHeight; ++y) {
            for (int x = 0; x < kernelWidth; ++x) {
                m_hostKernelMem[y * kernelWidth + x] = val;
            }
        }
    } else {
        float valX = 1.0 / kernelWidth;
        for (int x = 0; x < kernelWidth; ++x) {
            m_hostKernelMemH[x] = valX;
        }
        float valY = 1.0 / kernelHeight;
        for (int y = 0; y < kernelHeight; ++y) {
            m_hostKernelMemV[y] = valY;
        }
    }

    if (device == eDeviceType::GPU) {
        if (m_use2D) {
            if (m_deviceKernelMem == nullptr) {
                throw roccv::Exception("Device memory not allocated for AverageBlur kernel, GPU may not be available.",
                                       eStatusType::INVALID_OPERATION);
            }
            HIP_VALIDATE_NO_ERRORS(hipMemcpyAsync(m_deviceKernelMem, m_hostKernelMem,
                                                  kernelWidth * kernelHeight * sizeof(float), hipMemcpyHostToDevice,
                                                  stream));
        } else {
            if (m_deviceKernelMemH == nullptr || m_deviceKernelMemV == nullptr) {
                throw roccv::Exception("Device memory not allocated for AverageBlur kernel, GPU may not be available.",
                                       eStatusType::INVALID_OPERATION);
            }
            HIP_VALIDATE_NO_ERRORS(hipMemcpyAsync(m_deviceKernelMemH, m_hostKernelMemH, kernelWidth * sizeof(float),
                                                  hipMemcpyHostToDevice, stream));
            HIP_VALIDATE_NO_ERRORS(hipMemcpyAsync(m_deviceKernelMemV, m_hostKernelMemV, kernelHeight * sizeof(float),
                                                  hipMemcpyHostToDevice, stream));
        }
    }

    if (m_use2D) {
        // clang-format off
        static const std::unordered_map<
        eDataType, std::array<std::function<void(hipStream_t, const Tensor&, const Tensor&, float*, int, int, int, int, eBorderType, eDeviceType)>, 4>>
            funcs =
            {
                {eDataType::DATA_TYPE_U8, {dispatch_filter_2d_dtype<uchar1, float*>, 0, dispatch_filter_2d_dtype<uchar3, float*>, dispatch_filter_2d_dtype<uchar4, float*>}},
                {eDataType::DATA_TYPE_U16, {dispatch_filter_2d_dtype<ushort1, float*>, 0, dispatch_filter_2d_dtype<ushort3, float*>, dispatch_filter_2d_dtype<ushort4, float*>}},
                {eDataType::DATA_TYPE_S16, {dispatch_filter_2d_dtype<short1, float*>, 0, dispatch_filter_2d_dtype<short3, float*>, dispatch_filter_2d_dtype<short4, float*>}},
                {eDataType::DATA_TYPE_S32, {dispatch_filter_2d_dtype<int1, float*>, 0, dispatch_filter_2d_dtype<int3, float*>, dispatch_filter_2d_dtype<int4, float*>}},
                {eDataType::DATA_TYPE_F32, {dispatch_filter_2d_dtype<float1, float*>, 0, dispatch_filter_2d_dtype<float3, float*>, dispatch_filter_2d_dtype<float4, float*>}},
            };
        // clang-format on
        auto func = funcs.at(input.dtype().etype())[input.shape(input.layout().channels_index()) - 1];
        if (func == 0) throw Exception("Not mapped to a defined function.", eStatusType::INVALID_OPERATION);

        if (device == eDeviceType::GPU) {
            func(stream, input, output, m_deviceKernelMem, kernelWidth, kernelHeight, anchorX, anchorY, borderMode,
                 device);
            HIP_VALIDATE_NO_ERRORS(hipEventRecord(m_completionEvent, stream));
        } else if (device == eDeviceType::CPU) {
            func(stream, input, output, m_hostKernelMem, kernelWidth, kernelHeight, anchorX, anchorY, borderMode,
                 device);
        }
    } else {
        // clang-format off
        static const std::unordered_map<
        eDataType, std::array<std::function<void(hipStream_t, const Tensor&, const Tensor&, float*, float*, int, int, int, int, eBorderType, eDeviceType)>, 4>>
            funcs =
            {
                {eDataType::DATA_TYPE_U8, {dispatch_filter_2d_dtype_separable<uchar1, float*>, 0, dispatch_filter_2d_dtype_separable<uchar3, float*>, dispatch_filter_2d_dtype_separable<uchar4, float*>}},
                {eDataType::DATA_TYPE_U16, {dispatch_filter_2d_dtype_separable<ushort1, float*>, 0, dispatch_filter_2d_dtype_separable<ushort3, float*>, dispatch_filter_2d_dtype_separable<ushort4, float*>}},
                {eDataType::DATA_TYPE_S16, {dispatch_filter_2d_dtype_separable<short1, float*>, 0, dispatch_filter_2d_dtype_separable<short3, float*>, dispatch_filter_2d_dtype_separable<short4, float*>}},
                {eDataType::DATA_TYPE_S32, {dispatch_filter_2d_dtype_separable<int1, float*>, 0, dispatch_filter_2d_dtype_separable<int3, float*>, dispatch_filter_2d_dtype_separable<int4, float*>}},
                {eDataType::DATA_TYPE_F32, {dispatch_filter_2d_dtype_separable<float1, float*>, 0, dispatch_filter_2d_dtype_separable<float3, float*>, dispatch_filter_2d_dtype_separable<float4, float*>}},
            };
        // clang-format on
        auto func = funcs.at(input.dtype().etype())[input.shape(input.layout().channels_index()) - 1];
        if (func == 0) throw Exception("Not mapped to a defined function.", eStatusType::INVALID_OPERATION);

        if (device == eDeviceType::GPU) {
            func(stream, input, output, m_deviceKernelMemH, m_deviceKernelMemV, kernelWidth, kernelHeight, anchorX,
                 anchorY, borderMode, device);
            HIP_VALIDATE_NO_ERRORS(hipEventRecord(m_completionEvent, stream));
        } else if (device == eDeviceType::CPU) {
            func(stream, input, output, m_hostKernelMemH, m_hostKernelMemV, kernelWidth, kernelHeight, anchorX, anchorY,
                 borderMode, device);
        }
    }
}
}  // namespace roccv