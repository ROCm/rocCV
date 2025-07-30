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
#include <core/detail/casting.hpp>
#include "core/detail/type_traits.hpp"
#include "operator_types.h"

using namespace roccv;

namespace Kernels {

static __host__ __device__ __forceinline__ bool pixel_in_box(float ix, float iy, float left, float right, float top, float bottom) {
    return (ix > left) && (ix < right) && (iy > top) && (iy < bottom);
}

template <typename T>
static __host__ __device__ void blend_single_color(T &color, const T &color_in) {
    // Working type for internal pixel format, which is float and has 4 channels.
    using WorkType = float4; //detail::MakeType<float, 4>;
    WorkType fgColor = detail::RangeCast<WorkType>(color_in);
    WorkType bgColor = detail::RangeCast<WorkType>(color);
    float fgAlpha = fgColor.w;
    float bgAlpha = bgColor.w;
    float blendAlpha = fgAlpha + bgAlpha * (1 - fgAlpha);

    WorkType blendColor = (fgColor * fgAlpha + bgColor * bgAlpha * (1 - fgAlpha)) / blendAlpha;
    blendColor.w = blendAlpha;
    color = detail::RangeCast<T>(blendColor);
}

template <typename T>
inline static __host__ __device__ void shade_rectangle(const Rect_t &rect, int ix, int iy, T *out_color) {
    if (rect.bordered) {
        if (!pixel_in_box(ix, iy, rect.i_left, rect.i_right, rect.i_top, rect.i_bottom) &&
            pixel_in_box(ix, iy, rect.o_left, rect.o_right, rect.o_top, rect.o_bottom)) {
            blend_single_color<T>(out_color[0], rect.color);
        }
    } else {
        if (pixel_in_box(ix, iy, rect.o_left, rect.o_right, rect.o_top, rect.o_bottom)) {
            blend_single_color<T>(out_color[0], rect.color);
        }
    }
}
}  // namespace Kernels

// BT.601 Spec conversion factors
const float BT601A = 0.299F;
const float BT601B = 0.587F;
const float BT601C = 0.114F;
const float BT601D = 1.772F;
const float BT601E = 1.402F;

// BT.709 Spec conversion factors
const float BT709A = 0.2126F;
const float BT709B = 0.7152F;
const float BT709C = 0.0722F;
const float BT709D = 1.8556F;
const float BT709E = 1.5748F;

// BT.2020 Spec conversion factors
const float BT2020A = 0.2627F;
const float BT2020B = 0.6780F;
const float BT2020C = 0.0593F;
const float BT2020D = 1.8814F;
const float BT2020E = 1.4746F;

struct ConversionFactors {
    float a;
    float b;
    float c;
    float d;
    float e;
} typedef ConversionFactors;

namespace {
template <typename T>
inline __device__ __host__ float norm1(const T &a) {
    using namespace roccv::detail;
    if constexpr (NumElements<T> == 1)
        return abs(a.x);
    else if constexpr (NumElements<T> == 2)
        return abs(a.x) + abs(a.y);
    else if constexpr (NumElements<T> == 3)
        return abs(a.x) + abs(a.y) + abs(a.z);
    else if constexpr (NumElements<T> == 4)
        return abs(a.x) + abs(a.y) + abs(a.z) + abs(a.w);
}

// Advanced Color Conversion
inline ConversionFactors GetConversionFactorsForSpec(eColorSpec spec) {
    ConversionFactors factors;
    switch (spec) {
        case BT601:
            factors.a = BT601A;
            factors.b = BT601B;
            factors.c = BT601C;
            factors.d = BT601D;
            factors.e = BT601E;
            break;
        case BT709:
            factors.a = BT709A;
            factors.b = BT709B;
            factors.c = BT709C;
            factors.d = BT709D;
            factors.e = BT709E;
            break;
        case BT2020:
            factors.a = BT2020A;
            factors.b = BT2020B;
            factors.c = BT2020C;
            factors.d = BT2020D;
            factors.e = BT2020E;
            break;
        default:
            throw roccv::Exception("Unexpected Color Specification", roccv::eStatusType::NOT_IMPLEMENTED);
    }

    return factors;
}

template <typename T>
inline __device__ __host__ void bgr2yuv_kernel(const T &r, const T &g, const T &b, float &y, float &u, float &v,
                                               const ConversionFactors factors) {
    y = r * factors.a + g * factors.b + b * factors.c;
    v = (r - y) / factors.e;
    u = (b - y) / factors.d;
}

template <typename T>
inline __device__ __host__ void yuv2rgb_kernel(const T &y, const T &u, const T &v, float &r, float &g, float &b,
                                               const ConversionFactors factors, float delta) {
    r = y + factors.e * (v - delta);
    g = y - (factors.a * factors.e / factors.b) * (v - delta) - (factors.c * factors.b / factors.b) * (u - delta);
    b = y + factors.d * (u - delta);
}
}  // namespace