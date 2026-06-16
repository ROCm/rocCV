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
#include "op_gaussian.hpp"

#include <hip/hip_runtime.h>

#include <cfenv>
#include <cmath>
#include <functional>

#include "common/validation_helpers.hpp"
#include "core/detail/casting.hpp"
#include "filter2D.hpp"

namespace roccv {
Gaussian::Gaussian(int32_t maxKernelWidth, int32_t maxKernelHeight)
    : m_maxKernelWidth(maxKernelWidth), m_maxKernelHeight(maxKernelHeight) {
    size_t memSize = m_maxKernelWidth * m_maxKernelHeight * sizeof(float);
    m_hostKernelMem = static_cast<float*>(m_allocator.allocHostPinnedMem(memSize));

    int gpuCount = 0;
    hipError_t err = hipGetDeviceCount(&gpuCount);
    if (err == hipSuccess) {
        m_deviceKernelMem = static_cast<float*>(m_allocator.allocHipMem(memSize));
    }
}

Gaussian::~Gaussian() {
    m_allocator.freeHostPinnedMem(m_hostKernelMem);

    if (m_deviceKernelMem != nullptr) {
        m_allocator.freeHipMem(m_deviceKernelMem);
    }
}

void Gaussian::operator()(hipStream_t stream, const Tensor& input, Tensor& output, int kernelWidth, int kernelHeight,
                          double sigmaX, double sigmaY, eBorderType borderMode, eDeviceType device) {
    // Validate input tensor
    CHECK_TENSOR_DEVICE(input, device);
    CHECK_TENSOR_DATATYPES(input, DATA_TYPE_U8, DATA_TYPE_U16, DATA_TYPE_S16, DATA_TYPE_S32, DATA_TYPE_F32);
    CHECK_TENSOR_LAYOUT(input, TENSOR_LAYOUT_HWC, TENSOR_LAYOUT_NHWC);
    CHECK_TENSOR_CHANNELS(input, 1, 3, 4);

    // Validate output tensor
    CHECK_TENSOR_COMPARISON(input.dtype() == output.dtype());
    CHECK_TENSOR_COMPARISON(input.device() == output.device());
    CHECK_TENSOR_COMPARISON(input.shape() == output.shape());

    // Validate sigma
    if (sigmaX <= 0) {
        throw roccv::Exception("Invalid sigmaX = " + std::to_string(sigmaX) + ": Ensure that sigmaX is positive.",
                               eStatusType::INVALID_VALUE);
    }
    if (sigmaY <= 0) {
        sigmaY = sigmaX;
    }

    // Infer kernel size
    eDataType input_dtype = input.dtype().etype();
    const int prev_round = fegetround();
    fesetround(FE_TONEAREST);
    if (kernelWidth <= 0 && sigmaX > 0) {
        kernelWidth = static_cast<int>(std::rint(sigmaX * (input_dtype == DATA_TYPE_U8 ? 3 : 4) * 2 + 1)) | 1;
    }
    if (kernelHeight <= 0 && sigmaY > 0) {
        kernelHeight = static_cast<int>(std::rint(sigmaY * (input_dtype == DATA_TYPE_U8 ? 3 : 4) * 2 + 1)) | 1;
    }
    fesetround(prev_round);
    // Validate kernel size
    if (!(kernelWidth > 0 && kernelWidth % 2 == 1 && kernelWidth <= m_maxKernelWidth && kernelHeight > 0 &&
          kernelHeight % 2 == 1 && kernelHeight <= m_maxKernelHeight)) {
        throw roccv::Exception(
            "Invalid kernel size = " + std::to_string(kernelWidth) + ", " + std::to_string(kernelHeight) +
                ": Ensure that the kernel size is odd and less than the max:" + std::to_string(m_maxKernelWidth) +
                ", " + std::to_string(m_maxKernelHeight),
            eStatusType::INVALID_VALUE);
    }

    // compute the kernel
    int halfW = kernelWidth / 2;
    int halfH = kernelHeight / 2;
    float sqSigX = 2.0f * sigmaX * sigmaX;
    float sqSigY = 2.0f * sigmaY * sigmaY;
    float sum = 0.0f;
    for (int y = -halfH; y <= halfH; ++y) {
        for (int x = -halfW; x <= halfW; ++x) {
            float value = exp(-((x * x) / sqSigX + (y * y) / sqSigY));
            m_hostKernelMem[x + halfW + (y + halfH) * kernelWidth] = value;
            sum += value;
        }
    }
    for (int i = 0; i < kernelWidth * kernelHeight; i++) {
        m_hostKernelMem[i] /= sum;
    }
    if (device == eDeviceType::GPU) {
        HIP_VALIDATE_NO_ERRORS(hipMemcpyAsync(m_deviceKernelMem, m_hostKernelMem,
                                              kernelWidth * kernelHeight * sizeof(float), hipMemcpyHostToDevice,
                                              stream));
    }

    // compute the anchor to be center of kernel
    int anchorX = -1;
    int anchorY = -1;
    processAnchor(anchorX, anchorY, kernelWidth, kernelHeight);

    // clang-format off
    static const std::unordered_map<
    eDataType, std::array<std::function<void(hipStream_t, const Tensor&, const Tensor&, float*, int, int, int, int, eBorderType, eDeviceType)>, 4>>
        funcs = 
        {
            {eDataType::DATA_TYPE_U8, {dispatch_filter2D_dtype<uchar1, float*>, 0, dispatch_filter2D_dtype<uchar3, float*>, dispatch_filter2D_dtype<uchar4, float*>}},
            {eDataType::DATA_TYPE_U16, {dispatch_filter2D_dtype<ushort1, float*>, 0, dispatch_filter2D_dtype<ushort3, float*>, dispatch_filter2D_dtype<ushort4, float*>}},
            {eDataType::DATA_TYPE_S16, {dispatch_filter2D_dtype<short1, float*>, 0, dispatch_filter2D_dtype<short3, float*>, dispatch_filter2D_dtype<short4, float*>}},
            {eDataType::DATA_TYPE_S32, {dispatch_filter2D_dtype<int1, float*>, 0, dispatch_filter2D_dtype<int3, float*>, dispatch_filter2D_dtype<int4, float*>}},
            {eDataType::DATA_TYPE_F32, {dispatch_filter2D_dtype<float1, float*>, 0, dispatch_filter2D_dtype<float3, float*>, dispatch_filter2D_dtype<float4, float*>}},
        };
    // clang-format on
    auto func = funcs.at(input.dtype().etype())[input.shape(input.layout().channels_index()) - 1];
    if (func == 0) throw Exception("Not mapped to a defined function.", eStatusType::INVALID_OPERATION);

    if (device == eDeviceType::GPU) {
        func(stream, input, output, m_deviceKernelMem, kernelWidth, kernelHeight, anchorX, anchorY, borderMode, device);
    } else if (device == eDeviceType::CPU) {
        func(stream, input, output, m_hostKernelMem, kernelWidth, kernelHeight, anchorX, anchorY, borderMode, device);
    }
}
}  // namespace roccv