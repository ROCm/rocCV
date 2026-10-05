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

"""Targeted checks for the rocCV low-severity items found in the manual QA session.

  api-doc::threshold_NW_layout.<dev>        op_thresholding.hpp documents thresh/maxVal as layout NW; the op must accept it
  precision.gamma_contrast::U32_<...>       U32 gamma within +-1 of a float64 reference (U32 is computed in float32)
  repro::warp_affine_nearest_source_pixels  GPU and CPU NEAREST warp pick the same source pixel (FMA rounding drift)
"""
from __future__ import annotations

import numpy as np
import rocpycv as cv
from common import record, summary

GPU, CPU = cv.eDeviceType.GPU, cv.eDeviceType.CPU


def to_np(t):
    return np.array(np.from_dlpack(t.copy_to(CPU) if t.device() == GPU else t))


def threshold_nw_layout():
    for dev in (CPU, GPU):
        img = cv.from_dlpack(np.zeros((2, 8, 8, 3), np.uint8), cv.NHWC)
        arr = np.array([[100.0], [200.0]])
        th, mv = cv.from_dlpack(arr, cv.NW), cv.from_dlpack(arr.copy(), cv.NW)
        if dev == GPU:
            img, th, mv = img.copy_to(dev), th.copy_to(dev), mv.copy_to(dev)
        try:
            cv.threshold(img, th, mv, 2, cv.eThresholdType.BINARY, None, dev)
            record("api-doc", f"threshold_NW_layout.{dev.name}", "pass", "documented NW thresh/maxVal accepted", backend=dev.name)
        except Exception as e:
            record("api-doc", f"threshold_NW_layout.{dev.name}", "fail",
                   f"op_thresholding.hpp documents thresh/maxVal as layout 'NW', but the op rejects it: {e}", backend=dev.name)


def gamma_u32_strict():
    rng = np.random.default_rng(77)
    for g in (0.4, 2.2):
        a = rng.integers(0, 2**32 - 1, size=(2, 37, 61, 3), dtype=np.uint32, endpoint=True)
        mx = float(np.iinfo(np.uint32).max)
        ref = np.clip(np.rint(np.power(a.astype(np.float64) / mx, g) * mx), 0, mx)
        for dev in (CPU, GPU):
            t = cv.from_dlpack(a, cv.NHWC)
            if dev == GPU:
                t = t.copy_to(GPU)
            try:
                out = to_np(cv.gamma_contrast(t, g, None, dev)).astype(np.float64)
            except Exception as e:
                record("precision.gamma_contrast", f"U32_gamma={g}.{dev.name}", "error", f"{type(e).__name__}: {e}", backend=dev.name)
                continue
            d = np.abs(out - ref)
            nb = int((d > 1).sum())
            status = "pass" if nb == 0 else "fail"
            record("precision.gamma_contrast", f"U32_gamma={g}.{dev.name}", status,
                   f"{nb}/{d.size} elements differ from the float64 reference by more than 1 (max {d.max():.0f})",
                   backend=dev.name)


def warp_affine_nearest_repro():
    h, w = 97, 131
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    src = np.stack([xx, yy, np.zeros_like(xx)], -1).astype(np.float32)[None]
    m = [0.9, 0.2, 5.3, -0.15, 1.1, -3.7]
    args = (m, True, cv.eInterpolationType.NEAREST, cv.eBorderType.CONSTANT, [-1, -1, -1, -1])
    t = cv.from_dlpack(src, cv.NHWC)
    try:
        oc = to_np(cv.warp_affine(t, *args, None, CPU))
        og = to_np(cv.warp_affine(t.copy_to(GPU), *args, None, GPU))
    except Exception as e:
        record("repro", "warp_affine_nearest_source_pixels", "error", f"{type(e).__name__}: {e}")
        return
    diff = (oc[0, ..., 0] != og[0, ..., 0]) | (oc[0, ..., 1] != og[0, ..., 1])
    n = int(diff.sum())
    msg = f"GPU and CPU chose different source pixels for {n}/{h * w} ({100 * n / (h * w):.3f}%) output pixels"
    if n:
        y, x = np.argwhere(diff)[0]
        msg += (f"; e.g. dst({x},{y}) exact src=({m[0] * x + m[1] * y + m[2]:.7f},{m[3] * x + m[4] * y + m[5]:.7f}) "
                f"CPU->({oc[0, y, x, 0]:.0f},{oc[0, y, x, 1]:.0f}) GPU->({og[0, y, x, 0]:.0f},{og[0, y, x, 1]:.0f})")
    record("repro", "warp_affine_nearest_source_pixels", "fail" if n else "pass", msg)


if __name__ == "__main__":
    threshold_nw_layout()
    gamma_u32_strict()
    warp_affine_nearest_repro()
    summary()
