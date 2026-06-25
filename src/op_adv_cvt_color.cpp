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

#include "op_adv_cvt_color.hpp"

#include <hip/hip_runtime.h>

#include "common/validation_helpers.hpp"
#include "core/tensor.hpp"
#include "core/wrappers/image_wrapper.hpp"
#include "kernels/common/adv_cvt_color_coefficients.hpp"
#include "operator_types.h"
#include "kernels/device/adv_cvt_color_device.hpp"
#include "kernels/host/adv_cvt_color_host.hpp"

namespace roccv {
namespace {
constexpr float kDelta = 128.0f;

inline int64_t GetBatch(const Tensor &tensor) {
    int batchIdx = tensor.layout().batch_index();
    return batchIdx < 0 ? 1 : tensor.shape(batchIdx);
}

inline int64_t GetHeight(const Tensor &tensor) {
    return tensor.shape(tensor.layout().height_index());
}

inline int64_t GetWidth(const Tensor &tensor) {
    return tensor.shape(tensor.layout().width_index());
}

inline int64_t GetChannels(const Tensor &tensor) {
    return tensor.shape(tensor.layout().channels_index());
}

inline bool IsSemiPlanarToInterleaved(eColorConversionCode code) {
    switch (code) {
        case COLOR_YUV2RGB_NV12:
        case COLOR_YUV2BGR_NV12:
        case COLOR_YUV2RGB_NV21:
        case COLOR_YUV2BGR_NV21: return true;
        default: return false;
    }
}

inline bool IsInterleavedToSemiPlanar(eColorConversionCode code) {
    switch (code) {
        case COLOR_RGB2YUV_NV12:
        case COLOR_BGR2YUV_NV12:
        case COLOR_RGB2YUV_NV21:
        case COLOR_BGR2YUV_NV21: return true;
        default: return false;
    }
}

inline bool IsInterleaved444(eColorConversionCode code) {
    switch (code) {
        case COLOR_RGB2YUV:
        case COLOR_BGR2YUV:
        case COLOR_YUV2RGB:
        case COLOR_YUV2BGR: return true;
        default: return false;
    }
}

inline bool IsSupportedCode(eColorConversionCode code) {
    return IsInterleaved444(code) || IsSemiPlanarToInterleaved(code) || IsInterleavedToSemiPlanar(code);
}

inline bool IsBGRCode(eColorConversionCode code) {
    switch (code) {
        case COLOR_BGR2YUV:
        case COLOR_YUV2BGR:
        case COLOR_YUV2BGR_NV12:
        case COLOR_YUV2BGR_NV21:
        case COLOR_BGR2YUV_NV12:
        case COLOR_BGR2YUV_NV21: return true;
        default: return false;
    }
}

inline bool IsNV12Code(eColorConversionCode code) {
    switch (code) {
        case COLOR_YUV2RGB_NV12:
        case COLOR_YUV2BGR_NV12:
        case COLOR_RGB2YUV_NV12:
        case COLOR_BGR2YUV_NV12: return true;
        default: return false;
    }
}

inline Kernels::AdvCvtColorCoefficients GetCoefficients(eColorSpec spec) {
    switch (spec) {
        case BT601:
            return {0.299f, 0.587f, 0.114f, 0.564334086f, 0.713266762f, 1.402f, -0.344f, -0.714f, 1.772f};
        case BT709:
            return {0.2126f, 0.7152f, 0.0722f, 0.538909248f, 0.63500127f, 1.5748f, -0.187324f, -0.468124f, 1.8556f};
        case BT2020:
            return {0.2627f, 0.6780f, 0.0593f, 0.531519082f, 0.678150007f, 1.4746f, -0.16455f, -0.57135f, 1.8814f};
        default:
            throw Exception("Unsupported color specification.", eStatusType::INVALID_COMBINATION);
    }
}

inline int DivUp(int n, int d) {
    return (n + d - 1) / d;
}

}  // namespace

AdvCvtColor::AdvCvtColor() {}

AdvCvtColor::~AdvCvtColor() {}

void AdvCvtColor::operator()(hipStream_t stream, const Tensor &input, Tensor &output, eColorConversionCode conversionCode,
                             eColorSpec colorSpec, eDeviceType device) {
    CHECK_TENSOR_DEVICE(input, device);
    CHECK_TENSOR_DEVICE(output, device);

    CHECK_TENSOR_LAYOUT(input, eTensorLayout::TENSOR_LAYOUT_NHWC, eTensorLayout::TENSOR_LAYOUT_HWC);
    CHECK_TENSOR_LAYOUT(output, eTensorLayout::TENSOR_LAYOUT_NHWC, eTensorLayout::TENSOR_LAYOUT_HWC);
    CHECK_TENSOR_COMPARISON(input.layout() == output.layout());

    CHECK_TENSOR_DATATYPES(input, eDataType::DATA_TYPE_U8);
    CHECK_TENSOR_DATATYPES(output, eDataType::DATA_TYPE_U8);

    CHECK_TENSOR_COMPARISON(IsSupportedCode(conversionCode));

    const int64_t inBatch = GetBatch(input);
    const int64_t outBatch = GetBatch(output);
    const int64_t inWidth = GetWidth(input);
    const int64_t outWidth = GetWidth(output);
    const int64_t inHeight = GetHeight(input);
    const int64_t outHeight = GetHeight(output);
    const int64_t inChannels = GetChannels(input);
    const int64_t outChannels = GetChannels(output);

    CHECK_TENSOR_COMPARISON(inBatch == outBatch);
    CHECK_TENSOR_COMPARISON(inWidth == outWidth);

    if (IsInterleaved444(conversionCode)) {
        CHECK_TENSOR_COMPARISON(inHeight == outHeight);
        CHECK_TENSOR_COMPARISON(inChannels == 3);
        CHECK_TENSOR_COMPARISON(outChannels == 3);
    } else if (IsSemiPlanarToInterleaved(conversionCode)) {
        CHECK_TENSOR_COMPARISON(inChannels == 1);
        CHECK_TENSOR_COMPARISON(outChannels == 3 || outChannels == 4);
        CHECK_TENSOR_COMPARISON(inWidth % 2 == 0);
        CHECK_TENSOR_COMPARISON(inHeight % 3 == 0);
        CHECK_TENSOR_COMPARISON(outHeight == (inHeight * 2) / 3);
    } else {
        CHECK_TENSOR_COMPARISON(inChannels == 3 || inChannels == 4);
        CHECK_TENSOR_COMPARISON(outChannels == 1);
        CHECK_TENSOR_COMPARISON(inWidth % 2 == 0);
        CHECK_TENSOR_COMPARISON(inHeight % 2 == 0);
        CHECK_TENSOR_COMPARISON(outHeight == (inHeight * 3) / 2);
    }

    if (needsInt64Wrapper(input) || needsInt64Wrapper(output)) {
        throw Exception("Input or output tensor is too large for int32 indexing", eStatusType::INVALID_OPERATION);
    }

    Kernels::AdvCvtColorCoefficients coeff = GetCoefficients(colorSpec);
    const bool bgr = IsBGRCode(conversionCode);
    const int uidx = IsNV12Code(conversionCode) ? 0 : 1;

    if (device == eDeviceType::GPU) {
        dim3 blockSize(32, 16);

        if (IsInterleaved444(conversionCode)) {
            dim3 gridSize(DivUp(static_cast<int>(outWidth), blockSize.x), DivUp(static_cast<int>(outHeight), blockSize.y),
                          static_cast<unsigned int>(outBatch));

            switch (conversionCode) {
                case COLOR_BGR2YUV:
                    Kernels::Device::rgb_or_bgr_to_yuv_adv<uchar3, eSwizzle::ZYXW>
                        <<<gridSize, blockSize, 0, stream>>>(ImageWrapper<uchar3>(input), ImageWrapper<uchar3>(output),
                                                              coeff, kDelta);
                    break;
                case COLOR_RGB2YUV:
                    Kernels::Device::rgb_or_bgr_to_yuv_adv<uchar3, eSwizzle::XYZW>
                        <<<gridSize, blockSize, 0, stream>>>(ImageWrapper<uchar3>(input), ImageWrapper<uchar3>(output),
                                                              coeff, kDelta);
                    break;
                case COLOR_YUV2BGR:
                    Kernels::Device::yuv_to_rgb_or_bgr_adv<uchar3, eSwizzle::ZYXW>
                        <<<gridSize, blockSize, 0, stream>>>(ImageWrapper<uchar3>(input), ImageWrapper<uchar3>(output),
                                                              coeff, kDelta);
                    break;
                case COLOR_YUV2RGB:
                    Kernels::Device::yuv_to_rgb_or_bgr_adv<uchar3, eSwizzle::XYZW>
                        <<<gridSize, blockSize, 0, stream>>>(ImageWrapper<uchar3>(input), ImageWrapper<uchar3>(output),
                                                              coeff, kDelta);
                    break;
                default: throw Exception("Unsupported conversion code.", eStatusType::INVALID_COMBINATION);
            }
        } else if (IsSemiPlanarToInterleaved(conversionCode)) {
            dim3 gridSize(DivUp(static_cast<int>(outWidth), blockSize.x), DivUp(static_cast<int>(outHeight), blockSize.y),
                          static_cast<unsigned int>(outBatch));

            if (outChannels == 3) {
                if (bgr) {
                    Kernels::Device::nv12_or_nv21_to_rgb_or_bgr_adv<eSwizzle::ZYXW, ImageWrapper<uchar1>,
                                                                     ImageWrapper<uchar3>, uchar3>
                        <<<gridSize, blockSize, 0, stream>>>(ImageWrapper<uchar1>(input), ImageWrapper<uchar3>(output),
                                                              coeff, kDelta, uidx);
                } else {
                    Kernels::Device::nv12_or_nv21_to_rgb_or_bgr_adv<eSwizzle::XYZW, ImageWrapper<uchar1>,
                                                                     ImageWrapper<uchar3>, uchar3>
                        <<<gridSize, blockSize, 0, stream>>>(ImageWrapper<uchar1>(input), ImageWrapper<uchar3>(output),
                                                              coeff, kDelta, uidx);
                }
            } else {
                if (bgr) {
                    Kernels::Device::nv12_or_nv21_to_rgb_or_bgr_adv<eSwizzle::ZYXW, ImageWrapper<uchar1>,
                                                                     ImageWrapper<uchar4>, uchar4>
                        <<<gridSize, blockSize, 0, stream>>>(ImageWrapper<uchar1>(input), ImageWrapper<uchar4>(output),
                                                              coeff, kDelta, uidx);
                } else {
                    Kernels::Device::nv12_or_nv21_to_rgb_or_bgr_adv<eSwizzle::XYZW, ImageWrapper<uchar1>,
                                                                     ImageWrapper<uchar4>, uchar4>
                        <<<gridSize, blockSize, 0, stream>>>(ImageWrapper<uchar1>(input), ImageWrapper<uchar4>(output),
                                                              coeff, kDelta, uidx);
                }
            }
        } else {
            dim3 gridSize(DivUp(static_cast<int>(inWidth), blockSize.x), DivUp(static_cast<int>(inHeight), blockSize.y),
                          static_cast<unsigned int>(inBatch));

            if (inChannels == 3) {
                if (bgr) {
                    Kernels::Device::rgb_or_bgr_to_nv12_or_nv21_adv<uchar3, eSwizzle::ZYXW>
                        <<<gridSize, blockSize, 0, stream>>>(ImageWrapper<uchar3>(input), ImageWrapper<uchar1>(output),
                                                              coeff, kDelta, uidx);
                } else {
                    Kernels::Device::rgb_or_bgr_to_nv12_or_nv21_adv<uchar3, eSwizzle::XYZW>
                        <<<gridSize, blockSize, 0, stream>>>(ImageWrapper<uchar3>(input), ImageWrapper<uchar1>(output),
                                                              coeff, kDelta, uidx);
                }
            } else {
                if (bgr) {
                    Kernels::Device::rgb_or_bgr_to_nv12_or_nv21_adv<uchar4, eSwizzle::ZYXW>
                        <<<gridSize, blockSize, 0, stream>>>(ImageWrapper<uchar4>(input), ImageWrapper<uchar1>(output),
                                                              coeff, kDelta, uidx);
                } else {
                    Kernels::Device::rgb_or_bgr_to_nv12_or_nv21_adv<uchar4, eSwizzle::XYZW>
                        <<<gridSize, blockSize, 0, stream>>>(ImageWrapper<uchar4>(input), ImageWrapper<uchar1>(output),
                                                              coeff, kDelta, uidx);
                }
            }
        }
    } else {
        if (IsInterleaved444(conversionCode)) {
            switch (conversionCode) {
                case COLOR_BGR2YUV:
                    Kernels::Host::rgb_or_bgr_to_yuv_adv<uchar3, eSwizzle::ZYXW>(ImageWrapper<uchar3>(input),
                                                                                  ImageWrapper<uchar3>(output), coeff,
                                                                                  kDelta);
                    break;
                case COLOR_RGB2YUV:
                    Kernels::Host::rgb_or_bgr_to_yuv_adv<uchar3, eSwizzle::XYZW>(ImageWrapper<uchar3>(input),
                                                                                  ImageWrapper<uchar3>(output), coeff,
                                                                                  kDelta);
                    break;
                case COLOR_YUV2BGR:
                    Kernels::Host::yuv_to_rgb_or_bgr_adv<uchar3, eSwizzle::ZYXW>(ImageWrapper<uchar3>(input),
                                                                                  ImageWrapper<uchar3>(output), coeff,
                                                                                  kDelta);
                    break;
                case COLOR_YUV2RGB:
                    Kernels::Host::yuv_to_rgb_or_bgr_adv<uchar3, eSwizzle::XYZW>(ImageWrapper<uchar3>(input),
                                                                                  ImageWrapper<uchar3>(output), coeff,
                                                                                  kDelta);
                    break;
                default: throw Exception("Unsupported conversion code.", eStatusType::INVALID_COMBINATION);
            }
        } else if (IsSemiPlanarToInterleaved(conversionCode)) {
            if (outChannels == 3) {
                if (bgr) {
                    Kernels::Host::nv12_or_nv21_to_rgb_or_bgr_adv<eSwizzle::ZYXW, ImageWrapper<uchar1>,
                                                                   ImageWrapper<uchar3>, uchar3>(
                        ImageWrapper<uchar1>(input), ImageWrapper<uchar3>(output), coeff, kDelta, uidx);
                } else {
                    Kernels::Host::nv12_or_nv21_to_rgb_or_bgr_adv<eSwizzle::XYZW, ImageWrapper<uchar1>,
                                                                   ImageWrapper<uchar3>, uchar3>(
                        ImageWrapper<uchar1>(input), ImageWrapper<uchar3>(output), coeff, kDelta, uidx);
                }
            } else {
                if (bgr) {
                    Kernels::Host::nv12_or_nv21_to_rgb_or_bgr_adv<eSwizzle::ZYXW, ImageWrapper<uchar1>,
                                                                   ImageWrapper<uchar4>, uchar4>(
                        ImageWrapper<uchar1>(input), ImageWrapper<uchar4>(output), coeff, kDelta, uidx);
                } else {
                    Kernels::Host::nv12_or_nv21_to_rgb_or_bgr_adv<eSwizzle::XYZW, ImageWrapper<uchar1>,
                                                                   ImageWrapper<uchar4>, uchar4>(
                        ImageWrapper<uchar1>(input), ImageWrapper<uchar4>(output), coeff, kDelta, uidx);
                }
            }
        } else {
            if (inChannels == 3) {
                if (bgr) {
                    Kernels::Host::rgb_or_bgr_to_nv12_or_nv21_adv<uchar3, eSwizzle::ZYXW>(
                        ImageWrapper<uchar3>(input), ImageWrapper<uchar1>(output), coeff, kDelta, uidx);
                } else {
                    Kernels::Host::rgb_or_bgr_to_nv12_or_nv21_adv<uchar3, eSwizzle::XYZW>(
                        ImageWrapper<uchar3>(input), ImageWrapper<uchar1>(output), coeff, kDelta, uidx);
                }
            } else {
                if (bgr) {
                    Kernels::Host::rgb_or_bgr_to_nv12_or_nv21_adv<uchar4, eSwizzle::ZYXW>(
                        ImageWrapper<uchar4>(input), ImageWrapper<uchar1>(output), coeff, kDelta, uidx);
                } else {
                    Kernels::Host::rgb_or_bgr_to_nv12_or_nv21_adv<uchar4, eSwizzle::XYZW>(
                        ImageWrapper<uchar4>(input), ImageWrapper<uchar1>(output), coeff, kDelta, uidx);
                }
            }
        }
    }
}

}  // namespace roccv
