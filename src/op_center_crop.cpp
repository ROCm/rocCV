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
#include "op_center_crop.hpp"

#include "op_custom_crop.hpp"
#include "operator_types.h"

namespace roccv {
void CenterCrop::operator()(hipStream_t stream, const Tensor& input, const Tensor& output, const int32_t cropWidth, const int32_t cropHeight,
                            const eDeviceType device) const {

    auto i_height = input.shape()[input.shape().layout().height_index()];
    auto i_width = input.shape()[input.shape().layout().width_index()];

    auto half_height = i_height >> 1;
    auto half_width = i_width >> 1;

    int32_t half_cropWidth = cropWidth >> 1;
    int32_t half_cropHeight = cropHeight >> 1;

    int64_t upper_left_corner_x = half_width - half_cropWidth;
    int64_t upper_left_corner_y = half_height - half_cropHeight;

    Box_t cropRect{upper_left_corner_x, upper_left_corner_y, cropWidth, cropHeight};

    m_op(stream, input, output, cropRect, device);
}
}  // namespace roccv