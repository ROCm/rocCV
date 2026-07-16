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
#include "op_laplacian.hpp"

#include <hip/hip_runtime.h>

#include <functional>

#include "common/validation_helpers.hpp"
#include "core/detail/casting.hpp"
#include "details/filter_2d.hpp"
#include "kernels/common/laplacian_kernels.hpp"

namespace roccv {
void Laplacian::operator()(hipStream_t stream, const roccv::Tensor &input, const roccv::Tensor &output, int32_t ksize,
                           float scale, eBorderType borderMode, eDeviceType device) const {
    // Validate input tensor
    CHECK_TENSOR_DEVICE(input, device);
    CHECK_TENSOR_DATATYPES(input, DATA_TYPE_U8, DATA_TYPE_U16, DATA_TYPE_F32);
    CHECK_TENSOR_LAYOUT(input, TENSOR_LAYOUT_HWC, TENSOR_LAYOUT_NHWC);
    CHECK_TENSOR_CHANNELS(input, 1, 3, 4);

    // Validate output tensor
    CHECK_TENSOR_COMPARISON(input.dtype() == output.dtype());
    CHECK_TENSOR_COMPARISON(input.device() == output.device());
    CHECK_TENSOR_COMPARISON(input.shape() == output.shape());

    // Validate ksize
    if (!(ksize == 1 || ksize == 3)) {
        throw roccv::Exception("Invalid ksize = " + std::to_string(ksize) + ": Must be 1 or 3.",
                               eStatusType::INVALID_VALUE);
    }

    using namespace Kernels;
    LaplacianKernel kernel;

    if (ksize == 1) {
        kernel = LK1;
    } else if (ksize == 3) {
        kernel = LK3;
    }

    if (scale != 1) {
        kernel *= scale;
    }

    // compute the anchor to be center of kernel
    int anchorX = -1;
    int anchorY = -1;
    processAnchor(anchorX, anchorY, LaplaceKWidth, LaplaceKHeight);

    // clang-format off
    static const std::unordered_map<
    eDataType, std::array<std::function<void(hipStream_t, const Tensor&, const Tensor&, LaplacianKernel, int, int, int, int, eBorderType, eDeviceType)>, 4>>
        funcs =
        {
            {eDataType::DATA_TYPE_U8, {dispatch_filter_2d_dtype<uchar1, LaplacianKernel>, 0, dispatch_filter_2d_dtype<uchar3, LaplacianKernel>, dispatch_filter_2d_dtype<uchar4, LaplacianKernel>}},
            {eDataType::DATA_TYPE_U16, {dispatch_filter_2d_dtype<ushort1, LaplacianKernel>, 0, dispatch_filter_2d_dtype<ushort3, LaplacianKernel>, dispatch_filter_2d_dtype<ushort4, LaplacianKernel>}},
            {eDataType::DATA_TYPE_F32, {dispatch_filter_2d_dtype<float1, LaplacianKernel>, 0, dispatch_filter_2d_dtype<float3, LaplacianKernel>, dispatch_filter_2d_dtype<float4, LaplacianKernel>}},
        };
    // clang-format on
    auto func = funcs.at(input.dtype().etype())[input.shape(input.layout().channels_index()) - 1];
    if (func == 0) throw Exception("Not mapped to a defined function.", eStatusType::INVALID_OPERATION);

    func(stream, input, output, kernel, LaplaceKWidth, LaplaceKHeight, anchorX, anchorY, borderMode, device);
}
}  // namespace roccv