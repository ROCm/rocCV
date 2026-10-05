# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""HWC (unbatched) layout sweep (M20): every operator documented as 'NHWC, HWC', via the allocating and _into APIs.

Records hwc.<op>::<alloc|into>.<CPU|GPU>: pass when the call succeeds, fail when it raises.
Known (M20): CvtColor, GammaContrast, CenterCrop, CustomCrop, BrightnessContrast, Remap and Threshold raise
'TensorShape index out of bounds: -1' (Threshold: 'Invalid dimension: N'); Composite raises on the allocating path only.
"""
from __future__ import annotations

import numpy as np
import rocpycv as cv
from common import record, summary

GPU, CPU = cv.eDeviceType.GPU, cv.eDeviceType.CPU
R = np.random.default_rng(3)
H, W = 36, 60
BT, IT, CC = cv.eBorderType, cv.eInterpolationType, cv.eColorConversionCode
U8 = cv.eDataType.U8


def u8(c):
    return R.integers(0, 255, size=(H, W, c), dtype=np.uint8, endpoint=True)


def T(a, lay, dev):
    t = cv.from_dlpack(np.ascontiguousarray(a), lay)
    return t.copy_to(GPU) if dev == GPU else t


def out(shape, dt, dev):
    return cv.Tensor(list(shape), cv.HWC, dt, dev)


def boxes():
    return cv.BndBoxes([[cv.BndBox(cv.Box(2, 2, 10, 10), 1, cv.ColorRGBA(255, 0, 0, 255), cv.ColorRGBA(0, 0, 0, 0))]])


def zmap(d):
    return T(np.zeros((H, W, 2), np.float32), cv.HWC, d)


def base(d):
    return T(np.zeros((1, 1, 3), np.float32), cv.HWC, d)


def scale(d):
    return T(np.ones((1, 1, 3), np.float32), cv.HWC, d)


def nvec(v, d):
    return T(np.array([v]), cv.N, d)


CASES = {
    "advcvtcolor": (
        lambda d: cv.advcvtcolor(T(u8(3), cv.HWC, d), CC.COLOR_RGB2YUV, cv.eColorSpec.BT601, None, d),
        lambda d: cv.advcvtcolor_into(out((H, W, 3), U8, d), T(u8(3), cv.HWC, d), CC.COLOR_RGB2YUV, cv.eColorSpec.BT601, None, d)),
    "averageblur": (
        lambda d: cv.averageblur(T(u8(3), cv.HWC, d), (3, 3), (-1, -1), BT.REPLICATE, stream=None, device=d),
        lambda d: cv.averageblur_into(out((H, W, 3), U8, d), T(u8(3), cv.HWC, d), (3, 3), (-1, -1), BT.REPLICATE, stream=None,
                                      device=d)),
    "bilateral_filter": (
        lambda d: cv.bilateral_filter(T(u8(3), cv.HWC, d), 5, 50.0, 3.0, BT.REPLICATE, [0, 0, 0, 0], None, d),
        lambda d: cv.bilateral_filter_into(out((H, W, 3), U8, d), T(u8(3), cv.HWC, d), 5, 50.0, 3.0, BT.REPLICATE, [0, 0, 0, 0],
                                           None, d)),
    "bndbox": (
        lambda d: cv.bndbox(T(u8(3), cv.HWC, d), boxes(), None, d),
        lambda d: cv.bndbox_into(out((H, W, 3), U8, d), T(u8(3), cv.HWC, d), boxes(), None, d)),
    "brightness_contrast": (
        lambda d: cv.brightness_contrast(T(u8(3), cv.HWC, d), stream=None, device=d),
        lambda d: cv.brightness_contrast_into(out((H, W, 3), U8, d), T(u8(3), cv.HWC, d), stream=None, device=d)),
    "center_crop": (
        lambda d: cv.center_crop(T(u8(3), cv.HWC, d), (20, 10), None, d),
        lambda d: cv.center_crop_into(out((10, 20, 3), U8, d), T(u8(3), cv.HWC, d), (20, 10), None, d)),
    "composite": (
        lambda d: cv.composite(T(u8(3), cv.HWC, d), T(u8(3), cv.HWC, d), T(u8(1), cv.HWC, d), 3, None, d),
        lambda d: cv.composite_into(out((H, W, 3), U8, d), T(u8(3), cv.HWC, d), T(u8(3), cv.HWC, d), T(u8(1), cv.HWC, d), None,
                                    d)),
    "convert_to": (
        lambda d: cv.convert_to(T(u8(3), cv.HWC, d), cv.eDataType.F32, 1.0, 0.0, None, d),
        lambda d: cv.convert_to_into(out((H, W, 3), cv.eDataType.F32, d), T(u8(3), cv.HWC, d), 1.0, 0.0, None, d)),
    "copymakeborder": (
        lambda d: cv.copymakeborder(T(u8(3), cv.HWC, d), BT.CONSTANT, [0, 0, 0, 0], top=1, bottom=1, left=1, right=1,
                                    stream=None, device=d),
        lambda d: cv.copymakeborder_into(out((H + 2, W + 2, 3), U8, d), T(u8(3), cv.HWC, d), BT.CONSTANT, [0, 0, 0, 0], top=1,
                                         left=1, stream=None, device=d)),
    "custom_crop": (
        lambda d: cv.custom_crop(T(u8(3), cv.HWC, d), cv.Box(1, 1, 10, 10), None, d),
        lambda d: cv.custom_crop_into(out((10, 10, 3), U8, d), T(u8(3), cv.HWC, d), cv.Box(1, 1, 10, 10), None, d)),
    "cvtcolor": (
        lambda d: cv.cvtcolor(T(u8(3), cv.HWC, d), CC.COLOR_BGR2RGB, None, d),
        lambda d: cv.cvtcolor_into(out((H, W, 3), U8, d), T(u8(3), cv.HWC, d), CC.COLOR_BGR2RGB, None, d)),
    "cvtcolor_gray": (
        lambda d: cv.cvtcolor(T(u8(3), cv.HWC, d), CC.COLOR_BGR2GRAY, None, d),
        lambda d: cv.cvtcolor_into(out((H, W, 1), U8, d), T(u8(3), cv.HWC, d), CC.COLOR_BGR2GRAY, None, d)),
    "flip": (
        lambda d: cv.flip(T(u8(3), cv.HWC, d), 1, None, d),
        lambda d: cv.flip_into(out((H, W, 3), U8, d), T(u8(3), cv.HWC, d), 1, None, d)),
    "gamma_contrast": (
        lambda d: cv.gamma_contrast(T(u8(3), cv.HWC, d), 2.0, None, d),
        lambda d: cv.gamma_contrast_into(out((H, W, 3), U8, d), T(u8(3), cv.HWC, d), 2.0, None, d)),
    "gaussian": (
        lambda d: cv.gaussian(T(u8(3), cv.HWC, d), (3, 3), (1.0, 1.0), BT.REPLICATE, stream=None, device=d),
        lambda d: cv.gaussian_into(out((H, W, 3), U8, d), T(u8(3), cv.HWC, d), (3, 3), (1.0, 1.0), BT.REPLICATE, stream=None,
                                   device=d)),
    "histogram": (
        lambda d: cv.histogram(T(u8(1), cv.HWC, d), None, None, d),
        lambda d: cv.histogram_into(cv.Tensor([1, 256, 1], cv.HWC, cv.eDataType.S32, d), T(u8(1), cv.HWC, d), None, None, d)),
    "laplacian": (
        lambda d: cv.laplacian(T(u8(3), cv.HWC, d), 3, 1.0, BT.REPLICATE, stream=None, device=d),
        lambda d: cv.laplacian_into(out((H, W, 3), U8, d), T(u8(3), cv.HWC, d), 3, 1.0, BT.REPLICATE, stream=None, device=d)),
    "normalize": (
        lambda d: cv.normalize(T(u8(3), cv.HWC, d), base(d), scale(d), None, 1.0, 0.0, 0.0, None, d),
        lambda d: cv.normalize_into(out((H, W, 3), U8, d), T(u8(3), cv.HWC, d), base(d), scale(d), None, 1.0, 0.0, 0.0, None,
                                    d)),
    "reformat_HWC-CHW": (
        lambda d: cv.reformat(T(u8(3), cv.HWC, d), cv.CHW, None, d),
        lambda d: cv.reformat_into(cv.Tensor([3, H, W], cv.CHW, U8, d), T(u8(3), cv.HWC, d), None, d)),
    "remap": (
        lambda d: cv.remap(T(u8(3), cv.HWC, d), zmap(d), IT.NEAREST, IT.NEAREST, cv.REMAP_ABSOLUTE, False, BT.CONSTANT,
                           [0, 0, 0, 0], None, d),
        lambda d: cv.remap_into(out((H, W, 3), U8, d), T(u8(3), cv.HWC, d), zmap(d), IT.NEAREST, IT.NEAREST, cv.REMAP_ABSOLUTE,
                                False, BT.CONSTANT, [0, 0, 0, 0], None, d)),
    "resize": (
        lambda d: cv.resize(T(u8(3), cv.HWC, d), (2 * H, 2 * W, 3), IT.LINEAR, None, d),
        lambda d: cv.resize_into(out((2 * H, 2 * W, 3), U8, d), T(u8(3), cv.HWC, d), IT.LINEAR, None, d)),
    "rotate": (
        lambda d: cv.rotate(T(u8(3), cv.HWC, d), 30.0, (0.0, 0.0), IT.LINEAR, None, d),
        lambda d: cv.rotate_into(out((H, W, 3), U8, d), T(u8(3), cv.HWC, d), 30.0, (0.0, 0.0), IT.LINEAR, None, d)),
    "threshold": (
        lambda d: cv.threshold(T(u8(3), cv.HWC, d), nvec(100.0, d), nvec(255.0, d), 1, cv.eThresholdType.BINARY, None, d),
        lambda d: cv.threshold_into(out((H, W, 3), U8, d), T(u8(3), cv.HWC, d), nvec(100.0, d), nvec(255.0, d), 1,
                                    cv.eThresholdType.BINARY, None, d)),
    "warp_affine": (
        lambda d: cv.warp_affine(T(u8(3), cv.HWC, d), [1, 0, 1, 0, 1, 1], False, IT.LINEAR, BT.CONSTANT, [0, 0, 0, 0], None, d),
        lambda d: cv.warp_affine_into(out((H, W, 3), U8, d), T(u8(3), cv.HWC, d), [1, 0, 1, 0, 1, 1], False, IT.LINEAR,
                                      BT.CONSTANT, [0, 0, 0, 0], None, d)),
    "warp_perspective": (
        lambda d: cv.warp_perspective(T(u8(3), cv.HWC, d), [1, 0, 1, 0, 1, 1, 0, 0, 1], False, IT.LINEAR, BT.CONSTANT,
                                      [0, 0, 0, 0], None, d),
        lambda d: cv.warp_perspective_into(out((H, W, 3), U8, d), T(u8(3), cv.HWC, d), [1, 0, 1, 0, 1, 1, 0, 0, 1], False,
                                           IT.LINEAR, BT.CONSTANT, [0, 0, 0, 0], None, d)),
}


def main():
    for name, (alloc, into) in CASES.items():
        for path, fn in (("alloc", alloc), ("into", into)):
            for d in (CPU, GPU):
                try:
                    fn(d)
                    record(f"hwc.{name}", f"{path}.{d.name}", "pass", backend=d.name)
                except Exception as e:
                    record(f"hwc.{name}", f"{path}.{d.name}", "fail", f"HWC input raised: {str(e)[:200]}", backend=d.name)
    summary()


if __name__ == "__main__":
    main()
