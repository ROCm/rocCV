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

namespace Kernels::Host {

template <typename T, typename SrcWrapper, typename DstWrapper>
void rgb_or_bgr_to_yuv(SrcWrapper input, DstWrapper output, int orderIdx, float delta) {
    using namespace roccv;
    using namespace roccv::detail;

    // Working type will always be a 3-channel floating point since input/output is always RGB/BGR
    using work_type_t = MakeType<float, 3>;

#pragma omp parallel for
    for (int z_idx = 0; z_idx < output.batches(); z_idx++) {
        for (int y_idx = 0; y_idx < output.height(); y_idx++) {
            for (int x_idx = 0; x_idx < output.width(); x_idx++) {
                T val = input.at(z_idx, y_idx, x_idx, 0);
                work_type_t valF = StaticCast<work_type_t>(val);

                float r = valF[orderIdx];
                float g = valF[1];
                float b = valF[orderIdx ^ 2];

                float y = r * 0.299f + g * 0.587f + b * 0.114f;
                float cr = (r - y) * 0.877f + delta;
                float cb = (b - y) * 0.492f + delta;

                float3 out = make_float3(y, cb, cr);

                output.at(z_idx, y_idx, x_idx, 0) = detail::SaturateCast<T>(out);
            }
        }
    }
}

template <typename T, typename SrcWrapper, typename DstWrapper>
void yuv_to_rgb_or_bgr(SrcWrapper input, DstWrapper output, int orderIdx, float delta) {
    using namespace roccv;
    using namespace roccv::detail;
    using work_type_t = MakeType<float, 3>;

#pragma omp parallel for
    for (int z_idx = 0; z_idx < output.batches(); z_idx++) {
        for (int y_idx = 0; y_idx < output.height(); y_idx++) {
            for (int x_idx = 0; x_idx < output.width(); x_idx++) {
                T val = input.at(z_idx, y_idx, x_idx, 0);
                work_type_t valF = StaticCast<work_type_t>(val);

                // Y = valF.x
                // Cr = valF.y
                // Cb = valF.z

                // Convert from YUV to RGB
                work_type_t rgb = make_float3(valF.x + (valF.z - delta) * 1.140f,                                // R
                                              valF.x + (valF.y - delta) * -0.395f + (valF.z - delta) * -0.581f,  // G
                                              valF.x + (valF.y - delta) * 2.032f);                               // B

                // Reorder to proper layout (RGB or BGR, depending on orderIdx)
                work_type_t out;
                out.x = rgb[orderIdx];
                out.y = rgb[1];
                out.z = rgb[orderIdx ^ 2];

                // Saturate cast to type T (this clamps to proper ranges)
                output.at(z_idx, y_idx, x_idx, 0) = SaturateCast<T>(out);
            }
        }
    }
}

template <typename T, typename SrcWrapper, typename DstWrapper>
void rgb_or_bgr_to_bgr_or_rgb(SrcWrapper input, DstWrapper output, int orderIdxInput, int orderIdxOutput) {
    using namespace roccv::detail;

#pragma omp parallel for
    for (int z_idx = 0; z_idx < output.batches(); z_idx++) {
        for (int y_idx = 0; y_idx < output.height(); y_idx++) {
            for (int x_idx = 0; x_idx < output.width(); x_idx++) {
                T inVal = input.at(z_idx, y_idx, x_idx, 0);

                // Convert inVal into RGB format, depends on orderIdxInput
                T inValRGB{inVal[orderIdxInput], inVal[1], inVal[orderIdxInput ^ 2]};

                // Convert from inValRGB to requested final format, depends on orderIdxOutput
                T outVal{inValRGB[orderIdxOutput], inValRGB[1], inValRGB[orderIdxOutput ^ 2]};

                output.at(z_idx, y_idx, x_idx, 0) = outVal;
            }
        }
    }
}

template <typename T, typename SrcWrapper, typename DstWrapper>
void rgb_or_bgr_to_grayscale(SrcWrapper input, DstWrapper output, int orderIdxInput) {
    using namespace roccv::detail;
    using work_type_t = MakeType<float, 3>;

#pragma omp parallel for
    for (int z_idx = 0; z_idx < output.batches(); z_idx++) {
        for (int y_idx = 0; y_idx < output.height(); y_idx++) {
            for (int x_idx = 0; x_idx < output.width(); x_idx++) {
                T inVal = input.at(z_idx, y_idx, x_idx, 0);
                work_type_t inValF = StaticCast<work_type_t>(inVal);

                // Get RGB elements (input order defined by orderIdxInput)
                float r = inValF[orderIdxInput];
                float g = inValF[1];
                float b = inValF[orderIdxInput ^ 2];

                // Calculate luminance
                float y = r * 0.299f + g * 0.587f + b * 0.114f;

                output.at(z_idx, y_idx, x_idx, 0) = SaturateCast<uchar1>(y);
            }
        }
    }
}
}  // namespace Kernels::Host