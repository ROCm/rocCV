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
#include "op_cvt_color.hpp"

#include <hip/hip_runtime.h>

#include "common/validation_helpers.hpp"
#include "core/detail/hip_utils.hpp"
#include "core/tensor.hpp"
#include "core/wrappers/image_wrapper.hpp"
#include "kernels/device/cvt_color_device.hpp"
#include "kernels/host/cvt_color_host.hpp"

namespace roccv {
CvtColor::CvtColor() {}

CvtColor::~CvtColor() {}

void CvtColor::operator()(hipStream_t stream, const Tensor &input, Tensor &output, eColorConversionCode conversionCode,
                          eDeviceType device) {
    // Verify that the tensors are located on the right device (CPU or GPU).
    CHECK_TENSOR_DEVICE(input, device);
    CHECK_TENSOR_DEVICE(output, device);

    // Ensure all tensors are using supported layouts.
    CHECK_TENSOR_LAYOUT(input, eTensorLayout::TENSOR_LAYOUT_NHWC, eTensorLayout::TENSOR_LAYOUT_HWC);
    CHECK_TENSOR_LAYOUT(output, eTensorLayout::TENSOR_LAYOUT_NHWC, eTensorLayout::TENSOR_LAYOUT_HWC);

    CHECK_TENSOR_DATATYPES(input, eDataType::DATA_TYPE_U8);
    CHECK_TENSOR_DATATYPES(output, eDataType::DATA_TYPE_U8);

    CHECK_TENSOR_COMPARISON(input.layout() == output.layout());

    CHECK_TENSOR_CHANNELS(input, 3);

    bool grayscaleConversion = conversionCode == eColorConversionCode::COLOR_BGR2GRAY ||
                               conversionCode == eColorConversionCode::COLOR_RGB2GRAY;
    if (grayscaleConversion) {
        // Verification for grayscale conversions must be done differently
        CHECK_TENSOR_COMPARISON(input.shape(input.layout().width_index()) ==
                                output.shape(output.layout().width_index()));
        CHECK_TENSOR_COMPARISON(input.shape(input.layout().height_index()) ==
                                output.shape(output.layout().height_index()));

        if (input.layout().batch_index() >= 0) {
            CHECK_TENSOR_COMPARISON(input.shape(input.layout().batch_index()) ==
                                    output.shape(output.layout().batch_index()));
        }

        CHECK_TENSOR_CHANNELS(output, 1);
    } else {
        CHECK_TENSOR_CHANNELS(output, 3);
    }

    // Launch kernel

    int64_t width = input.shape(input.layout().width_index());
    int64_t height = input.shape(input.layout().height_index());
    int64_t samples = input.shape(input.layout().batch_index());

    if (device == eDeviceType::GPU) {
        // Dispatch appropriate device kernel based on given conversion code

        switch (conversionCode) {
            case eColorConversionCode::COLOR_BGR2GRAY: {
                constexpr auto kernel = Kernels::Device::rgb_or_bgr_to_grayscale<
                    uchar3, eSwizzle::ZYXW, ImageWrapper<uchar3>, ImageWrapper<uchar1>>;
                dim3 block = detail::GetBlockSize1D<kernel>();
                dim3 grid = detail::GetGridSize1D(width, height, samples, block);
                kernel<<<grid, block, 0, stream>>>(ImageWrapper<uchar3>(input), ImageWrapper<uchar1>(output));
                break;
            }

            case eColorConversionCode::COLOR_RGB2GRAY: {
                constexpr auto kernel = Kernels::Device::rgb_or_bgr_to_grayscale<
                    uchar3, eSwizzle::XYZW, ImageWrapper<uchar3>, ImageWrapper<uchar1>>;
                dim3 block = detail::GetBlockSize1D<kernel>();
                dim3 grid = detail::GetGridSize1D(width, height, samples, block);
                kernel<<<grid, block, 0, stream>>>(ImageWrapper<uchar3>(input), ImageWrapper<uchar1>(output));
                break;
            }

            case eColorConversionCode::COLOR_BGR2RGB:
            case eColorConversionCode::COLOR_RGB2BGR: {
                constexpr auto kernel = Kernels::Device::reorder<
                    uchar3, eSwizzle::ZYXW, ImageWrapper<uchar3>, ImageWrapper<uchar3>>;
                dim3 block = detail::GetBlockSize1D<kernel>();
                dim3 grid = detail::GetGridSize1D(width, height, samples, block);
                kernel<<<grid, block, 0, stream>>>(ImageWrapper<uchar3>(input), ImageWrapper<uchar3>(output));
                break;
            }

            case eColorConversionCode::COLOR_BGR2YUV: {
                constexpr auto kernel = Kernels::Device::rgb_or_bgr_to_yuv<
                    uchar3, eSwizzle::ZYXW, ImageWrapper<uchar3>, ImageWrapper<uchar3>>;
                dim3 block = detail::GetBlockSize1D<kernel>();
                dim3 grid = detail::GetGridSize1D(width, height, samples, block);
                kernel<<<grid, block, 0, stream>>>(ImageWrapper<uchar3>(input), ImageWrapper<uchar3>(output), 128.0f);
                break;
            }

            case eColorConversionCode::COLOR_RGB2YUV: {
                constexpr auto kernel = Kernels::Device::rgb_or_bgr_to_yuv<
                    uchar3, eSwizzle::XYZW, ImageWrapper<uchar3>, ImageWrapper<uchar3>>;
                dim3 block = detail::GetBlockSize1D<kernel>();
                dim3 grid = detail::GetGridSize1D(width, height, samples, block);
                kernel<<<grid, block, 0, stream>>>(ImageWrapper<uchar3>(input), ImageWrapper<uchar3>(output), 128.0f);
                break;
            }

            case eColorConversionCode::COLOR_YUV2BGR: {
                constexpr auto kernel = Kernels::Device::yuv_to_rgb_or_bgr<
                    uchar3, eSwizzle::ZYXW, ImageWrapper<uchar3>, ImageWrapper<uchar3>>;
                dim3 block = detail::GetBlockSize1D<kernel>();
                dim3 grid = detail::GetGridSize1D(width, height, samples, block);
                kernel<<<grid, block, 0, stream>>>(ImageWrapper<uchar3>(input), ImageWrapper<uchar3>(output), 128.0f);
                break;
            }

            case eColorConversionCode::COLOR_YUV2RGB: {
                constexpr auto kernel = Kernels::Device::yuv_to_rgb_or_bgr<
                    uchar3, eSwizzle::XYZW, ImageWrapper<uchar3>, ImageWrapper<uchar3>>;
                dim3 block = detail::GetBlockSize1D<kernel>();
                dim3 grid = detail::GetGridSize1D(width, height, samples, block);
                kernel<<<grid, block, 0, stream>>>(ImageWrapper<uchar3>(input), ImageWrapper<uchar3>(output), 128.0f);
                break;
            }

            default:
                throw Exception("Not implemented", eStatusType::NOT_IMPLEMENTED);
        }
    } else {
        // Dispatch appropriate host kernel based on conversion code

        switch (conversionCode) {
            case eColorConversionCode::COLOR_BGR2GRAY:
                Kernels::Host::rgb_or_bgr_to_grayscale<uchar3, eSwizzle::ZYXW>(ImageWrapper<uchar3>(input),
                                                                               ImageWrapper<uchar1>(output));
                break;

            case eColorConversionCode::COLOR_RGB2GRAY:
                Kernels::Host::rgb_or_bgr_to_grayscale<uchar3, eSwizzle::XYZW>(ImageWrapper<uchar3>(input),
                                                                               ImageWrapper<uchar1>(output));
                break;

            case eColorConversionCode::COLOR_BGR2RGB:
            case eColorConversionCode::COLOR_RGB2BGR:
                Kernels::Host::reorder<uchar3, eSwizzle::ZYXW>(ImageWrapper<uchar3>(input),
                                                               ImageWrapper<uchar3>(output));
                break;

            case eColorConversionCode::COLOR_BGR2YUV:
                Kernels::Host::rgb_or_bgr_to_yuv<uchar3, eSwizzle::ZYXW>(ImageWrapper<uchar3>(input),
                                                                         ImageWrapper<uchar3>(output), 128.0f);
                break;

            case eColorConversionCode::COLOR_RGB2YUV:
                Kernels::Host::rgb_or_bgr_to_yuv<uchar3, eSwizzle::XYZW>(ImageWrapper<uchar3>(input),
                                                                         ImageWrapper<uchar3>(output), 128.0f);
                break;

            case eColorConversionCode::COLOR_YUV2BGR:
                Kernels::Host::yuv_to_rgb_or_bgr<uchar3, eSwizzle::ZYXW>(ImageWrapper<uchar3>(input),
                                                                         ImageWrapper<uchar3>(output), 128.0f);
                break;

            case eColorConversionCode::COLOR_YUV2RGB:
                Kernels::Host::yuv_to_rgb_or_bgr<uchar3, eSwizzle::XYZW>(ImageWrapper<uchar3>(input),
                                                                         ImageWrapper<uchar3>(output), 128.0f);
                break;

            default:
                throw Exception("Not implemented", eStatusType::NOT_IMPLEMENTED);
        }
    }
}

}  // namespace roccv