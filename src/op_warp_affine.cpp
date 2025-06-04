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
#include "op_warp_affine.hpp"

#include <hip/hip_runtime.h>

namespace roccv {

void WarpAffine::operator()(hipStream_t stream, const Tensor& input, const Tensor& output, const AffineTransform xform,
                            const bool isInverted, const eInterpolationType interp, const eBorderType borderMode,
                            const float4 borderValue, const eDeviceType device) const {
    // An affine transformation is a subset of a perspective transform, so use the existing operator. All tensor
    // validation is performed in the WarpPerspective operator as well.

    PerspectiveTransform transform;
#pragma unroll
    for (int i = 0; i < 6; i++) {
        transform[i] = xform[i];
    }
    transform[6] = 0.0f;
    transform[7] = 0.0f;
    transform[8] = 1.0f;

    m_op(stream, input, output, transform, isInverted, interp, borderMode, borderValue, device);
}
}  // namespace roccv