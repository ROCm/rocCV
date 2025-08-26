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
        // one read
        T pixel = input.at(z_idx, y_idx, x_idx, 0);
        float RGB[3] = {static_cast<float>(pixel.x), static_cast<float>(pixel.y), static_cast<float>(pixel.z)};
        // order
        float R = RGB[orderIdx];
        float G = RGB[1];
        float B = RGB[orderIdx ^ 2];
        // convert
        float Y  = R * 0.299f + G * 0.587f + B * 0.114f;
        float Cr = (R - Y) * 0.877f + delta;
        float Cb = (B - Y) * 0.492f + delta;
        // round
        T YCbCr = {
            RoundImplementationsToYUV<float>(Y),
            RoundImplementationsToYUV<float>(Cb),
            RoundImplementationsToYUV<float>(Cr)
        };
        // output
        output.at(z_idx, y_idx, x_idx, 0) = YCbCr;
    }
}

template <typename T, typename SrcWrapper, typename DstWrapper>
__global__ void yuv_to_rgb_or_bgr(SrcWrapper input, DstWrapper output, int orderIdx, float delta) {
    const int x_idx = threadIdx.x + blockDim.x * blockIdx.x;
    const int y_idx = threadIdx.y + blockDim.y * blockIdx.y;
    const int z_idx = threadIdx.z + blockDim.z * blockIdx.z;
    using base_type = roccv::detail::BaseType<T>;
    base_type mn = std::numeric_limits<base_type>::max();
    base_type mx = std::numeric_limits<base_type>::min();

    if (x_idx < output.width() && y_idx < output.height() && z_idx < output.batches()) {
        // one read
        T pixel = input.at(z_idx, y_idx, x_idx, 0);
        float YCbCr[3] = {static_cast<float>(pixel.x), static_cast<float>(pixel.y), static_cast<float>(pixel.z)};
        // split
        auto [Y, Cb, Cr] = YCbCr;
        // convert
        float B = Y + (Cb - delta) * 2.032f;
        float G = Y + (Cb - delta) * -0.395f + (Cr - delta) * -0.581f;
        float R = Y + (Cr - delta) * 1.140f;
        // Round
        T RGB = {
            RoundImplementationsFromYUV<float>(R),
            RoundImplementationsFromYUV<float>(G),
            RoundImplementationsFromYUV<float>(B)
        };
        // Clamp
        RGB.x = Clamp<base_type, float>(RGB.x, mn, mx);
        RGB.y = Clamp<base_type, float>(RGB.y, mn, mx);
        RGB.z = Clamp<base_type, float>(RGB.z, mn, mx);
        // out order
        T pixOut;
        pixOut.x = RGB[orderIdx];
        pixOut.y = RGB[1];
        pixOut.z = RGB[orderIdx ^ 2];
        // output
        output.at(z_idx, y_idx, x_idx, 0) = pixOut;
    }
}

template <typename T, typename SrcWrapper, typename DstWrapper>
__global__ void rgb_or_bgr_to_bgr_or_rgb(SrcWrapper input, DstWrapper output, int orderIdxInput, int orderIdxOutput) {
    using base_type = roccv::detail::BaseType<T>;
    const int x_idx = threadIdx.x + blockDim.x * blockIdx.x;
    const int y_idx = threadIdx.y + blockDim.y * blockIdx.y;
    const int z_idx = threadIdx.z + blockDim.z * blockIdx.z;
    if (x_idx < output.width() && y_idx < output.height() && z_idx < output.batches()) {
        T pixel = input.at(z_idx, y_idx, x_idx, 0);
        base_type in[3] = {pixel.x, pixel.y, pixel.z};
        T ot = {in[orderIdxInput], in[1], in[orderIdxInput ^ 2]};
        T pixOut = {ot[orderIdxOutput], ot[1], ot[orderIdxOutput ^ 2]};
        output.at(z_idx, y_idx, x_idx, 0) = pixOut;
    }
}

template <typename T, typename SrcWrapper, typename DstWrapper>
__global__ void rgb_or_bgr_to_grayscale(SrcWrapper input, DstWrapper output, int orderIdxInput) {
    const int x_idx = threadIdx.x + blockDim.x * blockIdx.x;
    const int y_idx = threadIdx.y + blockDim.y * blockIdx.y;
    const int z_idx = threadIdx.z + blockDim.z * blockIdx.z;
    if (x_idx < output.width() && y_idx < output.height() && z_idx < output.batches()) {
        T pixel = input.at(z_idx, y_idx, x_idx, 0);
        float RGB[3] = {static_cast<float>(pixel.x), static_cast<float>(pixel.y), static_cast<float>(pixel.z)};
        float R = RGB[orderIdxInput];
        float G = RGB[1];
        float B = RGB[orderIdxInput ^ 2];
        float Y  = R * 0.299f + G * 0.587f + B * 0.114f;
        output.at(z_idx, y_idx, x_idx, 0).x = RoundImplementationsToYUV<float>(Y);
    }
}

}  // namespace Kernels::Device