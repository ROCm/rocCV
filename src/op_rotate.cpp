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
#include "op_rotate.hpp"

#include "op_warp_affine.hpp"
#include "operator_types.h"

namespace roccv {
void Rotate::operator()(hipStream_t stream, const Tensor &input, const Tensor &output, const double angle_deg,
                        const double2 shift, const eInterpolationType interpolation, const eDeviceType device) const {
    // Calculate an inverted affine matrix representing a clockwise rotation and a center shift.
    double angle_rad = angle_deg * (M_PI / 180.0f);
    AffineTransform affineMatrix = {static_cast<float>(cos(angle_rad)), static_cast<float>(sin(angle_rad)),
                                    static_cast<float>(shift.x),        static_cast<float>(-sin(angle_rad)),
                                    static_cast<float>(cos(angle_rad)), static_cast<float>(shift.y)};

    m_op(stream, input, output, affineMatrix, true, interpolation, eBorderType::BORDER_TYPE_CONSTANT,
         make_float4(0, 0, 0, 0), device);
};
}  // namespace roccv