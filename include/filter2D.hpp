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

#include <functional>
#include <unordered_map>

#include "core/exception.hpp"
#include "core/detail/vector_utils.hpp"
#include "core/tensor.hpp"
#include "core/wrappers/border_wrapper.hpp"
#include "core/wrappers/image_wrapper.hpp"
#include "operator_types.h"
#include "kernels/device/filter2D_device.hpp"
#include "kernels/host/filter2D_host.hpp"

namespace roccv {

inline void processAnchor(int& anchorX, int& anchorY, int kernelWidth, int kernelHeight) {
    if (anchorX < 0) {
        anchorX = kernelWidth >> 1;
    }
    if (anchorY < 0) {
        anchorY = kernelHeight >> 1;
    }
}

template <typename DT, typename KT, eBorderType BT>
void dispatch_filter2D_bordertype(hipStream_t stream, const Tensor& input, const Tensor& output, KT kernel,
                                  int kernelWidth, int kernelHeight, int anchorX, int anchorY, eDeviceType device) {
    BorderWrapper<DT, BT> inputWrapper(input, roccv::detail::SetAll<DT>(0));
    ImageWrapper<DT> outputWrapper(output);

    switch (device) {
        case eDeviceType::GPU: {
            dim3 block(16, 16);
            dim3 grid((outputWrapper.width() + block.x - 1) / block.x, (outputWrapper.height() + block.y - 1) / block.y,
                      outputWrapper.batches());
            Kernels::Device::filter2D<<<grid, block, 0, stream>>>(inputWrapper, outputWrapper, kernel, kernelWidth,
                                                                  kernelHeight, anchorX, anchorY);
            break;
        }
        case eDeviceType::CPU: {
            Kernels::Host::filter2D(inputWrapper, outputWrapper, kernel, kernelWidth, kernelHeight, anchorX, anchorY);
            break;
        }
    }
}

template <typename DT, typename KT>
void dispatch_filter2D_dtype(hipStream_t stream, const Tensor& input, const Tensor& output, KT kernel, int kernelWidth,
                             int kernelHeight, int anchorX, int anchorY, eBorderType borderMode, eDeviceType device) {
    // clang-format off
    static const std::unordered_map<eBorderType, std::function<void(hipStream_t, const Tensor&, const Tensor&, KT, int, int, int, int, eDeviceType)>>
        funcs = {
            {eBorderType::BORDER_TYPE_REPLICATE,   dispatch_filter2D_bordertype<DT, KT, eBorderType::BORDER_TYPE_REPLICATE>},
            {eBorderType::BORDER_TYPE_CONSTANT,    dispatch_filter2D_bordertype<DT, KT, eBorderType::BORDER_TYPE_CONSTANT>},
            {eBorderType::BORDER_TYPE_REFLECT,     dispatch_filter2D_bordertype<DT, KT, eBorderType::BORDER_TYPE_REFLECT>},
            {eBorderType::BORDER_TYPE_REFLECT101,  dispatch_filter2D_bordertype<DT, KT, eBorderType::BORDER_TYPE_REFLECT101>},
            {eBorderType::BORDER_TYPE_WRAP,        dispatch_filter2D_bordertype<DT, KT, eBorderType::BORDER_TYPE_WRAP>}
        };
    // clang-format on
    if (!funcs.contains(borderMode)) {
        throw Exception("The given border mode is not supported.", eStatusType::NOT_IMPLEMENTED);
    }

    auto func = funcs.at(borderMode);
    func(stream, input, output, kernel, kernelWidth, kernelHeight, anchorX, anchorY, device);
}
}  // namespace roccv