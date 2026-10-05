/*
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

// C++ confirmation of M20: operators documented for "NHWC, HWC" layouts, called with unbatched HWC tensors.
// cpp-probe.hwc::<Op>.<CPU|GPU>: pass when the call succeeds, fail when it throws.
#include <hip/hip_runtime.h>

#include <core/exception.hpp>
#include <core/tensor.hpp>
#include <functional>
#include <op_brightness_contrast.hpp>
#include <op_center_crop.hpp>
#include <op_composite.hpp>
#include <op_custom_crop.hpp>
#include <op_cvt_color.hpp>
#include <op_flip.hpp>
#include <op_gamma_contrast.hpp>
#include <op_remap.hpp>
#include <op_thresholding.hpp>

#include "check.hpp"

using namespace roccv;

static void run(const char* name, eDeviceType dev, const std::function<void()>& fn) {
    std::string id = std::string(name) + (dev == eDeviceType::GPU ? ".GPU" : ".CPU");
    try {
        fn();
        if (dev == eDeviceType::GPU) (void)hipDeviceSynchronize();
        vp_check("cpp-probe.hwc", id, "pass", "");
    } catch (const std::exception& e) {
        vp_check("cpp-probe.hwc", id, "fail", std::string("HWC input throws: ") + e.what());
    }
}

static TensorShape hwc(int64_t h, int64_t w, int64_t c) { return TensorShape(TensorLayout(TENSOR_LAYOUT_HWC), {h, w, c}); }

int main() {
    const int64_t H = 36, W = 60;
    for (eDeviceType dev : {eDeviceType::CPU, eDeviceType::GPU}) {
        Tensor in3(hwc(H, W, 3), DataType(DATA_TYPE_U8), dev);
        Tensor out3(hwc(H, W, 3), DataType(DATA_TYPE_U8), dev);
        Tensor in1(hwc(H, W, 1), DataType(DATA_TYPE_U8), dev);
        run("Flip", dev, [&] { Flip op; op(nullptr, in3, out3, 1, dev); });
        run("CvtColor", dev, [&] { CvtColor op; op(nullptr, in3, out3, eColorConversionCode::COLOR_BGR2RGB, dev); });
        run("GammaContrast", dev, [&] { GammaContrast op; op(nullptr, in3, out3, 2.0f, dev); });
        run("CenterCrop", dev, [&] {
            Tensor o(hwc(10, 20, 3), DataType(DATA_TYPE_U8), dev);
            CenterCrop op;
            op(nullptr, in3, o, Size2D{20, 10}, dev);
        });
        run("CustomCrop", dev, [&] {
            Tensor o(hwc(10, 10, 3), DataType(DATA_TYPE_U8), dev);
            CustomCrop op;
            op(nullptr, in3, o, Box_t{1, 1, 10, 10}, dev);
        });
        run("BrightnessContrast", dev, [&] {
            BrightnessContrast op;
            op(nullptr, in3, out3, std::nullopt, std::nullopt, std::nullopt, std::nullopt, dev);
        });
        run("Composite", dev, [&] { Composite op; op(nullptr, in3, in3, in1, out3, dev); });
        run("Threshold", dev, [&] {
            Tensor th(TensorShape(TensorLayout(TENSOR_LAYOUT_N), {1}), DataType(DATA_TYPE_F64), dev);
            Tensor mv(TensorShape(TensorLayout(TENSOR_LAYOUT_N), {1}), DataType(DATA_TYPE_F64), dev);
            Threshold op(eThresholdType::THRESH_BINARY, 1);
            op(nullptr, in3, out3, th, mv, dev);
        });
        run("Remap", dev, [&] {
            Tensor map(hwc(H, W, 2), DataType(DATA_TYPE_F32), dev);
            Remap op;
            op(nullptr, in3, out3, map, eInterpolationType::INTERP_TYPE_NEAREST, eInterpolationType::INTERP_TYPE_NEAREST,
               eRemapType::REMAP_ABSOLUTE, false, eBorderType::BORDER_TYPE_CONSTANT, make_float4(0, 0, 0, 0), dev);
        });
    }
    return 0;
}
