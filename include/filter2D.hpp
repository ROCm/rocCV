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

#include "core/detail/vector_utils.hpp"
#include "core/exception.hpp"
#include "core/tensor.hpp"
#include "core/wrappers/border_wrapper.hpp"
#include "core/wrappers/image_wrapper.hpp"
#include "kernels/device/filter2D_device.hpp"
#include "kernels/host/filter2D_host.hpp"
#include "operator_types.h"

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
template <typename DT, typename KT, eBorderType BT>
void dispatch_filter2D_bordertype_separable(hipStream_t stream, const Tensor& input, const Tensor& output, KT kernelH,
                                            KT kernelV, int kernelWidth, int kernelHeight, int anchorX, int anchorY,
                                            eDeviceType device) {
    if (device == eDeviceType::CPU) {
        // CPU fallback: compute 2D kernel from separable kernels
        std::vector<float> kernel2D(kernelWidth * kernelHeight);
        for (int y = 0; y < kernelHeight; ++y) {
            for (int x = 0; x < kernelWidth; ++x) {
                kernel2D[y * kernelWidth + x] = kernelH[x] * kernelV[y];
            }
        }
        dispatch_filter2D_bordertype<DT, float*, BT>(stream, input, output, kernel2D.data(), kernelWidth, kernelHeight, anchorX, anchorY, device);
        return;
    }

    BorderWrapper<DT, BT> inputWrapper(input, roccv::detail::SetAll<DT>(0));
    ImageWrapper<DT> outputWrapper(output);

    if (device == eDeviceType::GPU) {
        Tensor interm(output.shape(), output.dtype(), device);
        ImageWrapper<DT> intermWrapper(interm);

        // constants copied from Pavel
        constexpr int BLOCK_WIDTH = 128;
        constexpr int BLOCK_HEIGHT = 128;

        // horizontal pass
        {
            dim3 block(BLOCK_WIDTH, 1);
            dim3 grid((outputWrapper.width() + BLOCK_WIDTH - 1) / BLOCK_WIDTH, outputWrapper.height(),
                      outputWrapper.batches());

            int halo = kernelWidth - 1;
            int tileWidth = BLOCK_WIDTH + halo;
            size_t smemSize = tileWidth * sizeof(DT);

            Kernels::Device::filter2D_horizontal<DT, BLOCK_WIDTH, BorderWrapper<DT, BT>, ImageWrapper<DT>, KT>
                <<<grid, block, smemSize, stream>>>(inputWrapper, intermWrapper, kernelH, kernelWidth, anchorX);

            // not sure if error checking necessary since none of the other dispatches do it?
            hipError_t err = hipGetLastError();
            if (err != hipSuccess) {
                throw Exception("Horizontal filter2D kernel launch failed: " + std::string(hipGetErrorString(err)),
                                eStatusType::INVALID_OPERATION);
            }
        }

        // Vertical pass
        BorderWrapper<DT, BT> intermWrapperWithBorder(interm, roccv::detail::SetAll<DT>(0));
        {
            dim3 block(1, BLOCK_HEIGHT);
            dim3 grid(outputWrapper.width(), (outputWrapper.height() + BLOCK_HEIGHT - 1) / BLOCK_HEIGHT,
                      outputWrapper.batches());

            int halo = kernelHeight - 1;
            int tileHeight = BLOCK_HEIGHT + halo;
            size_t smemSize = tileHeight * sizeof(DT);

            Kernels::Device::filter2D_vertical<DT, BLOCK_HEIGHT, BorderWrapper<DT, BT>, ImageWrapper<DT>, KT>
                <<<grid, block, smemSize, stream>>>(intermWrapperWithBorder, outputWrapper, kernelV, kernelHeight,
                                                    anchorY);

            // not sure if error checking necessary since none of the other dispatches do it?
            hipError_t err = hipGetLastError();
            if (err != hipSuccess) {
                throw Exception("Vertical filter2D kernel launch failed: " + std::string(hipGetErrorString(err)),
                                eStatusType::INVALID_OPERATION);
            }
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

template <typename DT, typename KT>
void dispatch_filter2D_dtype_separable(hipStream_t stream, const Tensor& input, const Tensor& output, KT kernelH,
                                       KT kernelV, int kernelWidth, int kernelHeight, int anchorX, int anchorY,
                                       eBorderType borderMode, eDeviceType device) {
    // clang-format off
    static const std::unordered_map<eBorderType, std::function<void(hipStream_t, const Tensor&, const Tensor&, KT, KT, int, int, int, int, eDeviceType)>>
        funcs = {
            {eBorderType::BORDER_TYPE_REPLICATE,   dispatch_filter2D_bordertype_separable<DT, KT, eBorderType::BORDER_TYPE_REPLICATE>},
            {eBorderType::BORDER_TYPE_CONSTANT,    dispatch_filter2D_bordertype_separable<DT, KT, eBorderType::BORDER_TYPE_CONSTANT>},
            {eBorderType::BORDER_TYPE_REFLECT,     dispatch_filter2D_bordertype_separable<DT, KT, eBorderType::BORDER_TYPE_REFLECT>},
            {eBorderType::BORDER_TYPE_REFLECT101,  dispatch_filter2D_bordertype_separable<DT, KT, eBorderType::BORDER_TYPE_REFLECT101>},
            {eBorderType::BORDER_TYPE_WRAP,        dispatch_filter2D_bordertype_separable<DT, KT, eBorderType::BORDER_TYPE_WRAP>}
        };
    // clang-format on
    if (!funcs.contains(borderMode)) {
        throw Exception("The given border mode is not supported.", eStatusType::NOT_IMPLEMENTED);
    }

    auto func = funcs.at(borderMode);
    func(stream, input, output, kernelH, kernelV, kernelWidth, kernelHeight, anchorX, anchorY, device);
}
}  // namespace roccv