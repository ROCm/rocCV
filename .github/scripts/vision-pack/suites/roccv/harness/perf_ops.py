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

"""Per-operator latency through the rocpycv *_into APIs (pre-allocated outputs, stream sync), 8 x 1080p RGB U8.

    perf_ops.py <perf.json>

Writes vp_perf metrics perf_ops.<op>.<GPU|CPU> (median ms) and records
  perf::perf_ops.<op>.<dev>       pass when the operator ran, error when it raised
  perf-audit::<check>             relative-speed checks for the low item "Gaussian and AverageBlur about 10x slower
                                  than comparable operators; Rotate 2.4x slower than WarpAffine"
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np
import rocpycv as cv
from common import record, summary

GPU, CPU = cv.eDeviceType.GPU, cv.eDeviceType.CPU
N, H, W, C = 8, 1080, 1920, 3
GPU_ITERS = int(os.environ.get("VP_ROCCV_PERF_GPU_ITERS", "30"))
CPU_ITERS = int(os.environ.get("VP_ROCCV_PERF_CPU_ITERS", "2"))
rng = np.random.default_rng(0)
img = rng.integers(0, 255, size=(N, H, W, C), dtype=np.uint8, endpoint=True)
BT, IT, CC = cv.eBorderType, cv.eInterpolationType, cv.eColorConversionCode
U8, F32 = cv.eDataType.U8, cv.eDataType.F32


def mk(a, lay, dev):
    t = cv.from_dlpack(np.ascontiguousarray(a), lay)
    return t.copy_to(dev) if dev == GPU else t


def empty(shape, lay, dt, dev):
    return cv.Tensor(list(shape), lay, dt, dev)


def cases(dev):
    s = cv.Stream()
    src = mk(img, cv.NHWC, dev)
    out = empty((N, H, W, C), cv.NHWC, U8, dev)
    outf = empty((N, H, W, C), cv.NHWC, F32, dev)
    out1 = empty((N, H, W, 1), cv.NHWC, U8, dev)
    mask = mk(img[..., :1], cv.NHWC, dev)
    th = mk(np.full(N, 100.0), cv.N, dev)
    mv = mk(np.full(N, 255.0), cv.N, dev)
    base = mk(np.full((1, 1, 1, C), 100, np.float32), cv.NHWC, dev)
    scl = mk(np.full((1, 1, 1, C), 0.02, np.float32), cv.NHWC, dev)
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    rmap = mk(np.tile(np.stack([W - 1 - xx, yy], -1).astype(np.float32)[None], (N, 1, 1, 1)), cv.NHWC, dev)
    half = empty((N, H // 2, W // 2, C), cv.NHWC, U8, dev)
    crop = empty((N, 224, 224, C), cv.NHWC, U8, dev)
    border = empty((N, H + 20, W + 20, C), cv.NHWC, U8, dev)
    nchw = empty((N, C, H, W), cv.NCHW, U8, dev)
    hist = empty((N, 256, 1), cv.HWC, cv.eDataType.S32, dev)
    nv12 = empty((N, H * 3 // 2, W, 1), cv.NHWC, U8, dev)
    boxes = mk(np.concatenate([rng.integers(0, 1500, (N, 1000, 2)), rng.integers(10, 200, (N, 1000, 2))], -1).astype(np.int16),
               cv.NWC, dev)
    scores = mk(rng.random((N, 1000)).astype(np.float32), cv.NW, dev)
    nmsout = empty((N, 1000), cv.NW, U8, dev)
    bb = cv.BndBoxes([[cv.BndBox(cv.Box(100 + 50 * i, 100, 300, 200), 4, cv.ColorRGBA(255, 0, 0, 255),
                                 cv.ColorRGBA(0, 255, 0, 64)) for i in range(10)] for _ in range(N)])
    r = math.radians(30)
    cx, cy = (W - 1) / 2, (H - 1) / 2
    shift = ((1 - math.cos(r)) * cx - math.sin(r) * cy, math.sin(r) * cx + (1 - math.cos(r)) * cy)
    return s, {
        "flip": lambda: cv.flip_into(out, src, -1, s, dev),
        "center_crop_224": lambda: cv.center_crop_into(crop, src, (224, 224), s, dev),
        "custom_crop_224": lambda: cv.custom_crop_into(crop, src, cv.Box(100, 100, 224, 224), s, dev),
        "copymakeborder_10_REFLECT": lambda: cv.copymakeborder_into(border, src, BT.REFLECT, [0, 0, 0, 0], top=10, left=10,
                                                                    stream=s, device=dev),
        "reformat_NHWC_NCHW": lambda: cv.reformat_into(nchw, src, s, dev),
        "convert_to_u8_f32": lambda: cv.convert_to_into(outf, src, 1 / 255.0, 0.0, s, dev),
        "normalize_u8": lambda: cv.normalize_into(out, src, base, scl, None, 1.0, 0.0, 0.0, s, dev),
        "threshold_BINARY": lambda: cv.threshold_into(out, src, th, mv, N, cv.eThresholdType.BINARY, s, dev),
        "gamma_contrast_2p2": lambda: cv.gamma_contrast_into(out, src, 2.2, s, dev),
        "brightness_contrast": lambda: cv.brightness_contrast_into(out, src, stream=s, device=dev),
        "cvtcolor_BGR2RGB": lambda: cv.cvtcolor_into(out, src, CC.COLOR_BGR2RGB, s, dev),
        "cvtcolor_BGR2GRAY": lambda: cv.cvtcolor_into(out1, src, CC.COLOR_BGR2GRAY, s, dev),
        "advcvtcolor_RGB2YUV_NV12_BT709": lambda: cv.advcvtcolor_into(nv12, src, CC.COLOR_RGB2YUV_NV12, cv.eColorSpec.BT709, s,
                                                                      dev),
        "composite": lambda: cv.composite_into(out, src, src, mask, s, dev),
        "histogram": lambda: cv.histogram_into(hist, out1, None, s, dev),
        "resize_LINEAR_half": lambda: cv.resize_into(half, src, IT.LINEAR, s, dev),
        "resize_CUBIC_half": lambda: cv.resize_into(half, src, IT.CUBIC, s, dev),
        "warp_affine_LINEAR": lambda: cv.warp_affine_into(out, src, [0.9, 0.2, 5.3, -0.15, 1.1, -3.7], False, IT.LINEAR,
                                                          BT.CONSTANT, [0, 0, 0, 0], s, dev),
        "warp_perspective_LINEAR": lambda: cv.warp_perspective_into(out, src, [1, 0.05, 3, 0.02, 0.95, -2, 5e-5, 3e-5, 1], False,
                                                                    IT.LINEAR, BT.CONSTANT, [0, 0, 0, 0], s, dev),
        "rotate_30deg_LINEAR": lambda: cv.rotate_into(out, src, 30.0, shift, IT.LINEAR, s, dev),
        "remap_LINEAR": lambda: cv.remap_into(out, src, rmap, IT.LINEAR, IT.NEAREST, cv.REMAP_ABSOLUTE, False, BT.CONSTANT,
                                              [0, 0, 0, 0], s, dev),
        "gaussian_5x5": lambda: cv.gaussian_into(out, src, (5, 5), (1.5, 1.5), BT.REPLICATE, stream=s, device=dev),
        "averageblur_5x5": lambda: cv.averageblur_into(out, src, (5, 5), (-1, -1), BT.REPLICATE, stream=s, device=dev),
        "laplacian_k3": lambda: cv.laplacian_into(out, src, 3, 1.0, BT.REPLICATE, stream=s, device=dev),
        "bilateral_d9": lambda: cv.bilateral_filter_into(out, src, 9, 50.0, 5.0, BT.REPLICATE, [0, 0, 0, 0], s, dev),
        "bndbox_10_per_image": lambda: cv.bndbox_into(out, src, bb, s, dev),
        "nms_1000_boxes": lambda: cv.nms_into(nmsout, boxes, scores, 0.3, 0.5, s, dev),
    }


def bench(dev, iters):
    s, cs = cases(dev)
    res = {}
    for name, fn in cs.items():
        try:
            fn()
            s.synchronize()
            ts = []
            for _ in range(iters):
                t0 = time.perf_counter()
                fn()
                s.synchronize()
                ts.append((time.perf_counter() - t0) * 1e3)
            res[name] = float(np.median(ts))
            record("perf", f"perf_ops.{name}.{dev.name}", "pass", f"median {res[name]:.3f} ms over {iters} runs",
                   sum(ts) / 1e3, dev.name)
        except Exception as e:
            record("perf", f"perf_ops.{name}.{dev.name}", "error", f"{type(e).__name__}: {str(e)[:200]}", backend=dev.name)
    return res


def ratio_check(g, name, num, den, limit):
    if num not in g or den not in g:
        record("perf-audit", name, "error", f"missing timing for {num} or {den}")
        return
    r = g[num] / g[den]
    record("perf-audit", name, "fail" if r > limit else "pass",
           f"GPU {num} {g[num]:.3f} ms / {den} {g[den]:.3f} ms = {r:.2f}x (limit {limit}x; 8x1080p U8, shared GPU)")


def main():
    g = bench(GPU, GPU_ITERS)
    c = bench(CPU, CPU_ITERS)
    metrics = [{"name": f"perf_ops.{k}.GPU", "value": round(v, 4), "unit": "ms", "lower_is_better": True, "backend": "GPU"}
               for k, v in g.items()]
    metrics += [{"name": f"perf_ops.{k}.CPU", "value": round(v, 3), "unit": "ms", "lower_is_better": True, "backend": "CPU"}
                for k, v in c.items()]
    with open(sys.argv[1], "w") as f:
        json.dump({"metrics": metrics, "batch": [N, H, W, C], "gpu_iters": GPU_ITERS, "cpu_iters": CPU_ITERS}, f, indent=1)
    ratio_check(g, "rotate_vs_warp_affine", "rotate_30deg_LINEAR", "warp_affine_LINEAR", 1.5)
    ratio_check(g, "gaussian_vs_laplacian", "gaussian_5x5", "laplacian_k3", 2.0)
    ratio_check(g, "averageblur_vs_laplacian", "averageblur_5x5", "laplacian_k3", 2.0)
    summary()


if __name__ == "__main__":
    main()
