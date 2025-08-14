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

#include <hip/hip_runtime.h>
#include "operator_types.h"

namespace Kernels::Device {

template <typename T, typename SrcWrapper, typename DstWrapper>
__global__ void rgb_or_bgr_to_yuv(SrcWrapper input, DstWrapper output, int orderIdx, float delta) {
    const auto x_idx = threadIdx.x + blockIdx.x * blockDim.x;
    const auto y_idx = threadIdx.y + blockIdx.y * blockDim.y;
    const auto z_idx = threadIdx.z + blockIdx.z * blockDim.z;
    if (x_idx < output.width() && y_idx < output.height() && z_idx < output.batches()) {
        float R = static_cast<float>(input.at(z_idx, y_idx, x_idx, orderIdx));
        float G = static_cast<float>(input.at(z_idx, y_idx, x_idx, 1));
        float B = static_cast<float>(input.at(z_idx, y_idx, x_idx, orderIdx ^ 2));

        float Y = R * 0.299f + G * 0.587f + B * 0.114f;
        float Cr = (R - Y) * 0.877f + delta;
        float Cb = (B - Y) * 0.492f + delta;

        output.at(z_idx, y_idx, x_idx, 0) = RoundImplementationsToYUV<float>(Y);
        output.at(z_idx, y_idx, x_idx, 1) = RoundImplementationsToYUV<float>(Cb);
        output.at(z_idx, y_idx, x_idx, 2) = RoundImplementationsToYUV<float>(Cr);
    }
}

template <typename T, typename SrcWrapper, typename DstWrapper>
__global__ void yuv_to_rgb_or_bgr(SrcWrapper input, DstWrapper output, int orderIdx, float delta) {
    const int x_idx = threadIdx.x + blockDim.x * blockIdx.x;
    const int y_idx = threadIdx.y + blockDim.y * blockIdx.y;
    const int z_idx = threadIdx.z + blockDim.z * blockIdx.z;

    T max_type_value = std::numeric_limits<T>::max();
    T min_type_value = std::numeric_limits<T>::min();

    if (x_idx < output.width() && y_idx < output.height() && z_idx < output.batches()) {
        float Y  = static_cast<float>(input.at(z_idx, y_idx, x_idx, 0));
        float Cb = static_cast<float>(input.at(z_idx, y_idx, x_idx, 1));
        float Cr = static_cast<float>(input.at(z_idx, y_idx, x_idx, 2));

        float B = Y + (Cb - delta) * 2.032f;
        float G = Y + (Cb - delta) * -0.395f + (Cr - delta) * -0.581f;
        float R = Y + (Cr - delta) * 1.140f;

        output.at(z_idx, y_idx, x_idx, orderIdx) = Clamp<T, float>(RoundImplementationsFromYUV<float>(R), min_type_value, max_type_value);
        output.at(z_idx, y_idx, x_idx, 1) = Clamp<T, float>(RoundImplementationsFromYUV<float>(G), min_type_value, max_type_value);
        output.at(z_idx, y_idx, x_idx, orderIdx ^ 2) = Clamp<T, float>(RoundImplementationsFromYUV<float>(B), min_type_value, max_type_value);
    }
}

template <typename T, typename SrcWrapper, typename DstWrapper>
__global__ void rgb_or_bgr_to_bgr_or_rgb(SrcWrapper input, DstWrapper output, int orderIdxInput, int orderIdxOutput) {
    const int x_idx = threadIdx.x + blockDim.x * blockIdx.x;
    const int y_idx = threadIdx.y + blockDim.y * blockIdx.y;
    const int z_idx = threadIdx.z + blockDim.z * blockIdx.z;

    if (x_idx < output.width() && y_idx < output.height() && z_idx < output.batches()) {
        output.at(z_idx, y_idx, x_idx, orderIdxOutput) = input.at(z_idx, y_idx, x_idx, orderIdxInput);
        output.at(z_idx, y_idx, x_idx, 1) = input.at(z_idx, y_idx, x_idx, 1);
        output.at(z_idx, y_idx, x_idx, orderIdxOutput ^ 2) = input.at(z_idx, y_idx, x_idx, orderIdxInput ^ 2);
    }
}

template <typename T, typename SrcWrapper, typename DstWrapper>
__global__ void rgb_or_bgr_to_grayscale(SrcWrapper input, DstWrapper output, int orderIdxInput) {
    const int x_idx = threadIdx.x + blockDim.x * blockIdx.x;
    const int y_idx = threadIdx.y + blockDim.y * blockIdx.y;
    const int z_idx = threadIdx.z + blockDim.z * blockIdx.z;

    if (x_idx < output.width() && y_idx < output.height() && z_idx < output.batches()) {
        float grayValue = 0.0f;
        grayValue += static_cast<float>(input.at(z_idx, y_idx, x_idx, orderIdxInput)) * 0.299;
        grayValue += static_cast<float>(input.at(z_idx, y_idx, x_idx, 1)) * 0.587;
        grayValue += static_cast<float>(input.at(z_idx, y_idx, x_idx, orderIdxInput ^ 2)) * 0.114;
        output.at(z_idx, y_idx, x_idx, 0) = RoundImplementationsToYUV<float>(grayValue);
    }
}

}  // namespace Kernels::Device