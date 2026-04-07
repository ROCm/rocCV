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

#pragma once

#include <hip/hip_runtime.h>

#include <core/detail/casting.hpp>
#include <core/detail/swizzling.hpp>
#include <core/detail/type_traits.hpp>

#include "kernels/common/adv_cvt_color_coefficients.hpp"

namespace Kernels::Host {

template <typename T, roccv::eSwizzle S, typename SrcWrapper, typename DstWrapper>
void rgb_or_bgr_to_yuv_adv(SrcWrapper input, DstWrapper output, AdvCvtColorCoefficients coeff, float delta) {
    using namespace roccv::detail;
    using work_type_t = MakeType<float, NumElements<T>>;

#pragma omp parallel for
    for (int z_idx = 0; z_idx < output.batches(); z_idx++) {
        for (int y_idx = 0; y_idx < output.height(); y_idx++) {
            for (int x_idx = 0; x_idx < output.width(); x_idx++) {
                T inVal = Swizzle<S>(input.at(z_idx, y_idx, x_idx, 0));
                work_type_t inValF = StaticCast<work_type_t>(inVal);

                float y = inValF.x * coeff.r2y + inValF.y * coeff.g2y + inValF.z * coeff.b2y;
                float u = (inValF.z - y) * coeff.b2u + delta;
                float v = (inValF.x - y) * coeff.r2v + delta;

                output.at(z_idx, y_idx, x_idx, 0) = SaturateCast<T>(make_float3(y, u, v));
            }
        }
    }
}

template <typename T, roccv::eSwizzle S, typename SrcWrapper, typename DstWrapper>
void yuv_to_rgb_or_bgr_adv(SrcWrapper input, DstWrapper output, AdvCvtColorCoefficients coeff, float delta) {
    using namespace roccv::detail;
    using work_type_t = MakeType<float, NumElements<T>>;

#pragma omp parallel for
    for (int z_idx = 0; z_idx < output.batches(); z_idx++) {
        for (int y_idx = 0; y_idx < output.height(); y_idx++) {
            for (int x_idx = 0; x_idx < output.width(); x_idx++) {
                T inVal = input.at(z_idx, y_idx, x_idx, 0);
                work_type_t inValF = StaticCast<work_type_t>(inVal);

                float r = inValF.x + (inValF.z - delta) * coeff.v2r;
                float g = inValF.x + (inValF.y - delta) * coeff.u2g + (inValF.z - delta) * coeff.v2g;
                float b = inValF.x + (inValF.y - delta) * coeff.u2b;

                output.at(z_idx, y_idx, x_idx, 0) = Swizzle<S>(SaturateCast<T>(make_float3(r, g, b)));
            }
        }
    }
}

template <roccv::eSwizzle S, typename SrcWrapper, typename DstWrapper, typename DstT>
void nv12_or_nv21_to_rgb_or_bgr_adv(SrcWrapper input, DstWrapper output, AdvCvtColorCoefficients coeff, float delta,
                                    int uidx) {
    using namespace roccv::detail;

#pragma omp parallel for
    for (int z_idx = 0; z_idx < output.batches(); z_idx++) {
        for (int y_idx = 0; y_idx < output.height(); y_idx++) {
            for (int x_idx = 0; x_idx < output.width(); x_idx++) {
                int rgbHeight = output.height();
                int uvRow = rgbHeight + y_idx / 2;
                int uvCol = x_idx & ~1;

                float y = static_cast<float>(input.at(z_idx, y_idx, x_idx, 0).x);
                float u = static_cast<float>(input.at(z_idx, uvRow, uvCol + uidx, 0).x);
                float v = static_cast<float>(input.at(z_idx, uvRow, uvCol + (1 - uidx), 0).x);

                float r = y + (v - delta) * coeff.v2r;
                float g = y + (u - delta) * coeff.u2g + (v - delta) * coeff.v2g;
                float b = y + (u - delta) * coeff.u2b;

                if constexpr (NumElements<DstT> == 4) {
                    output.at(z_idx, y_idx, x_idx, 0) =
                        Swizzle<S>(SaturateCast<DstT>(make_float4(r, g, b, 255.0f)));
                } else {
                    output.at(z_idx, y_idx, x_idx, 0) = Swizzle<S>(SaturateCast<DstT>(make_float3(r, g, b)));
                }
            }
        }
    }
}

template <typename SrcT, roccv::eSwizzle S, typename SrcWrapper, typename DstWrapper>
void rgb_or_bgr_to_nv12_or_nv21_adv(SrcWrapper input, DstWrapper output, AdvCvtColorCoefficients coeff, float delta,
                                    int uidx) {
    using namespace roccv::detail;
    using work_type_t = MakeType<float, NumElements<SrcT>>;

#pragma omp parallel for
    for (int z_idx = 0; z_idx < output.batches(); z_idx++) {
        for (int y_idx = 0; y_idx < input.height(); y_idx++) {
            for (int x_idx = 0; x_idx < input.width(); x_idx++) {
                SrcT p0 = Swizzle<S>(input.at(z_idx, y_idx, x_idx, 0));
                work_type_t rgb0 = StaticCast<work_type_t>(p0);
                float y0 = rgb0.x * coeff.r2y + rgb0.y * coeff.g2y + rgb0.z * coeff.b2y;
                float u0 = (rgb0.z - y0) * coeff.b2u;
                float v0 = (rgb0.x - y0) * coeff.r2v;

                output.at(z_idx, y_idx, x_idx, 0) = SaturateCast<uchar1>(y0);

                if ((x_idx & 1) != 0 || (y_idx & 1) != 0) continue;
                if (x_idx + 1 >= input.width() || y_idx + 1 >= input.height()) continue;

                SrcT p1 = Swizzle<S>(input.at(z_idx, y_idx, x_idx + 1, 0));
                SrcT p2 = Swizzle<S>(input.at(z_idx, y_idx + 1, x_idx, 0));
                SrcT p3 = Swizzle<S>(input.at(z_idx, y_idx + 1, x_idx + 1, 0));
                work_type_t rgb1 = StaticCast<work_type_t>(p1);
                work_type_t rgb2 = StaticCast<work_type_t>(p2);
                work_type_t rgb3 = StaticCast<work_type_t>(p3);

                float y1 = rgb1.x * coeff.r2y + rgb1.y * coeff.g2y + rgb1.z * coeff.b2y;
                float y2 = rgb2.x * coeff.r2y + rgb2.y * coeff.g2y + rgb2.z * coeff.b2y;
                float y3 = rgb3.x * coeff.r2y + rgb3.y * coeff.g2y + rgb3.z * coeff.b2y;

                float u1 = (rgb1.z - y1) * coeff.b2u;
                float u2 = (rgb2.z - y2) * coeff.b2u;
                float u3 = (rgb3.z - y3) * coeff.b2u;

                float v1 = (rgb1.x - y1) * coeff.r2v;
                float v2 = (rgb2.x - y2) * coeff.r2v;
                float v3 = (rgb3.x - y3) * coeff.r2v;

                int uvRow = input.height() + y_idx / 2;
                int uvCol = x_idx;

                float uAvg = (u0 + u1 + u2 + u3) * 0.25f + delta;
                float vAvg = (v0 + v1 + v2 + v3) * 0.25f + delta;

                output.at(z_idx, uvRow, uvCol + uidx, 0) = SaturateCast<uchar1>(uAvg);
                output.at(z_idx, uvRow, uvCol + (1 - uidx), 0) = SaturateCast<uchar1>(vAvg);
            }
        }
    }
}

}  // namespace Kernels::Host
