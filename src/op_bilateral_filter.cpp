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
#include "op_bilateral_filter.hpp"

#include <hip/hip_runtime.h>

#include <functional>
#include <iostream>

#include "common/array_wrapper.hpp"
#include "common/validation_helpers.hpp"
#include "core/detail/casting.hpp"
#include "core/detail/math/math.hpp"
#include "core/detail/type_traits.hpp"
#include "kernels/device/bilateral_filter_device.hpp"
#include "kernels/host/bilateral_filter_host.hpp"

namespace roccv {
BilateralFilter::BilateralFilter() {}

BilateralFilter::~BilateralFilter() {}

template <typename T, eBorderType B>
void dispatch_bilateral_filter_border_mode(hipStream_t stream, const Tensor &input, const Tensor &output, int diameter,
                                           float sigmaColor, float sigmaSpace, const T borderValue,
                                           const eDeviceType device) {
    BorderWrapper<T, B> inputWrapper(input, borderValue);
    ImageWrapper<T> outputWrapper(output);

    int radius = diameter >> 1;

    if (outputWrapper.channels() > 4 || outputWrapper.channels() < 1) {
        throw Exception("Invalid channel size: cannot be greater than 4 or less than 1.", eStatusType::OUT_OF_BOUNDS);
    }

    int numThreads = 32;
    std::vector<std::thread> threads;

    // sigma values should be non-negative
    if (sigmaColor <= 0) {
        sigmaColor = 1.0f;
    }
    if (sigmaSpace <= 0) {
        sigmaSpace = 1.0f;
    }

    if (radius <= 0) {
        radius = std::round(sigmaSpace * 1.5f);
    }

    float spaceCoeff = -1 / (2 * sigmaSpace * sigmaSpace);
    float colorCoeff = -1 / (2 * sigmaColor * sigmaColor);

    if (device == eDeviceType::GPU) {
        dim3 block(8, 8);
        uint32_t xGridSize = (outputWrapper.width() + (block.x * 2) - 1) / (block.x * 2);
        uint32_t yGridSize = (outputWrapper.height() + (block.y * 2) - 1) / (block.y * 2);
        uint32_t zGridSize = outputWrapper.batches();
        dim3 grid(xGridSize, yGridSize, zGridSize);

        Kernels::Device::bilateral_filter<T><<<grid, block, 0, stream>>>(
            inputWrapper, outputWrapper, radius, sigmaColor, sigmaSpace, spaceCoeff, colorCoeff);
    } else if (device == eDeviceType::CPU) {
        int divisor = 4;
        int dividend = numThreads / divisor;

        int factorW = outputWrapper.width() / dividend;
        int factorH = outputWrapper.height() / divisor;
        int rollingWidth = factorW;
        int rollingHeight = factorH;
        int prevWidth = 0;
        int prevHeight = 0;

        for (int j = 0; j < divisor; j++) {
            for (int i = 0; i < dividend; i++) {
                threads.push_back(std::thread(Kernels::Host::bilateral_filter<T, BorderWrapper<T, B>, ImageWrapper<T>>,
                                              inputWrapper, outputWrapper, radius, sigmaColor, sigmaSpace,
                                              rollingHeight, rollingWidth, prevHeight, prevWidth, spaceCoeff,
                                              colorCoeff));
                prevWidth = rollingWidth;
                rollingWidth += factorW;
            }
            prevWidth = 0;
            rollingWidth = factorW;
            prevHeight = rollingHeight;
            rollingHeight += factorH;
        }
        for (int i = 0; i < numThreads; i++) {
            threads[i].join();
        }
    }
}

template <typename T>
void dispatch_bilateral_filter_dtype(hipStream_t stream, const Tensor &input, const Tensor &output, int diameter,
                                     float sigmaColor, float sigmaSpace, eBorderType borderMode,
                                     const float4 borderValue, const eDeviceType device) {
    // Select kernel dispatcher based on requested border mode.
    // clang-format off
    const std::function<void(hipStream_t, const Tensor&, const Tensor&, int, float, float, T, const eDeviceType)>
        funcs[4] = {
            dispatch_bilateral_filter_border_mode<T, eBorderType::BORDER_TYPE_CONSTANT>,
            dispatch_bilateral_filter_border_mode<T, eBorderType::BORDER_TYPE_REPLICATE>,
            dispatch_bilateral_filter_border_mode<T, eBorderType::BORDER_TYPE_REFLECT>,
            dispatch_bilateral_filter_border_mode<T, eBorderType::BORDER_TYPE_WRAP>
        };
    // clang-format on

    auto func = funcs[borderMode];
    if (func == 0)
        throw Exception("Not mapped to a defined function.", eStatusType::INVALID_OPERATION);
    func(stream, input, output, diameter, sigmaColor, sigmaSpace, detail::RangeCast<T>(borderValue), device);
}

void BilateralFilter::operator()(hipStream_t stream, const roccv::Tensor &input, const roccv::Tensor &output,
                                 int diameter, float sigmaColor, float sigmaSpace, const eBorderType borderMode,
                                 const float4 borderValue, const eDeviceType device) {
    // Verify that the tensors are located on the right device (CPU or GPU).
    CHECK_TENSOR_DEVICE(input, device);
    CHECK_TENSOR_DEVICE(output, device);

    // Ensure all tensors are using supported datatypes
    CHECK_TENSOR_DATATYPES(input, eDataType::DATA_TYPE_U8);
    CHECK_TENSOR_DATATYPES(output, eDataType::DATA_TYPE_U8);

    // Ensure all tensors are using supported layouts.
    CHECK_TENSOR_LAYOUT(input, eTensorLayout::TENSOR_LAYOUT_NHWC, eTensorLayout::TENSOR_LAYOUT_HWC);
    CHECK_TENSOR_LAYOUT(output, eTensorLayout::TENSOR_LAYOUT_NHWC, eTensorLayout::TENSOR_LAYOUT_HWC);

    CHECK_TENSOR_CHANNELS(input, 1, 3, 4);

    eDataType dtype = input.dtype().etype();
    int64_t channels = input.shape(input.layout().channels_index());

    // Ensure the layout and shapes for the input/output tensor match
    CHECK_TENSOR_COMPARISON(input.layout() == output.layout());
    CHECK_TENSOR_COMPARISON(input.shape() == output.shape());

    // Select kernel dispatcher based on number of channels and a base datatype.
    // clang-format off
    const std::function<void(hipStream_t, const Tensor &, const Tensor &, int, float, float, eBorderType, const float4, const eDeviceType)>
        funcs[1][4] = {
            {dispatch_bilateral_filter_dtype<uchar1>, 0, dispatch_bilateral_filter_dtype<uchar3>, dispatch_bilateral_filter_dtype<uchar4>},
        };
    // clang-format on

    auto func = funcs[dtype][channels - 1];
    if (func == 0)
        throw Exception("Not mapped to a defined function.", eStatusType::INVALID_OPERATION);
    func(stream, input, output, diameter, sigmaColor, sigmaSpace, borderMode, borderValue, device);

    //////////////////////////////////////////////////////////////////////////////////////////////////////////
}
}  // namespace roccv