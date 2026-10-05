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

"""Check the images written by the rocCV C++ samples.

    samples_check.py <data_dir> <out_dir>

samples.compare::<check>:
  <sample>_gpu_vs_cpu        GPU and CPU outputs agree within 1
  copy_make_border_mode<m>   border modes against numpy.pad
  center_crop_vs_input / custom_crop_vs_input   crops against the decoded input
  normalize_output_valid     normalize writes a full-size, non-constant image (the sample exits 0 even on errors)
  <sample>_batch_outputs     batch mode writes one image of the expected size per input
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
from common import record, summary
from PIL import Image

G = "samples.compare"


def load(p):
    return np.asarray(Image.open(p).convert("RGB")).astype(np.int16)


def exists(name, *paths):
    missing = [os.path.basename(p) for p in paths if not os.path.exists(p)]
    if missing:
        record(G, name, "fail", f"output missing: {', '.join(missing)}")
        return False
    return True


def main():
    d, o = sys.argv[1], sys.argv[2]
    for name, g, c in [("bilateral_filter", "bilateral_gpu.png", "bilateral_cpu.png"), ("bnd_box", "bndbox_gpu.png", "bndbox_cpu.png"),
                       ("center_crop", "center_crop_gpu.png", "center_crop_cpu.png"),
                       ("custom_crop", "custom_crop_gpu.png", "custom_crop_cpu.png"),
                       ("normalize", "normalize_gpu.bmp", "normalize_cpu.bmp"),
                       ("cropandresize", "cropresize_gpu.png", "cropresize_cpu.png")]:
        if not exists(f"{name}_gpu_vs_cpu", f"{o}/{g}", f"{o}/{c}"):
            continue
        a, b = load(f"{o}/{g}"), load(f"{o}/{c}")
        if a.shape != b.shape:
            record(G, f"{name}_gpu_vs_cpu", "fail", f"shape {a.shape} vs {b.shape}")
            continue
        diff = np.abs(a - b)
        record(G, f"{name}_gpu_vs_cpu", "pass" if diff.max() <= 1 else "fail",
               f"max|GPU-CPU|={int(diff.max())}, {int((diff > 0).sum())} values differ")

    src = load(f"{d}/mug_small.png")
    for m, npm in {0: "constant", 1: "edge", 2: "symmetric", 3: "reflect", 4: "wrap"}.items():
        name = f"copy_make_border_mode{m}"
        if not exists(name, f"{o}/cmb_mode{m}.png"):
            continue
        out = load(f"{o}/cmb_mode{m}.png")
        if m == 0:
            ref = np.pad(src, ((20, 20), (15, 15), (0, 0)), mode="constant")
            ref[:20] = ref[-20:] = [255, 0, 0]
            ref[:, :15] = ref[:, -15:] = [255, 0, 0]
        else:
            ref = np.pad(src, ((20, 20), (15, 15), (0, 0)), mode=npm)
        if out.shape != ref.shape:
            record(G, name, "fail", f"shape {out.shape}, expected {ref.shape}")
        else:
            mx = int(np.abs(out - ref).max())
            record(G, name, "pass" if mx == 0 else "fail", f"max|diff| vs numpy.pad({npm})={mx}")

    big = load(f"{d}/mug.jpg")
    h, w = big.shape[:2]
    x0, y0 = (w >> 1) - (640 >> 1), (h >> 1) - (480 >> 1)
    for name, f, ref in [("center_crop_vs_input", "center_crop_gpu.png", big[y0:y0 + 480, x0:x0 + 640]),
                         ("custom_crop_vs_input", "custom_crop_gpu.png", big[200:680, 100:740])]:
        if not exists(name, f"{o}/{f}"):
            continue
        out = load(f"{o}/{f}")
        if out.shape != ref.shape:
            record(G, name, "fail", f"shape {out.shape}, expected {ref.shape}")
        else:
            md = float(np.abs(out - ref).mean())
            record(G, name, "pass" if md <= 0.5 else "fail", f"mean|diff| vs the decoded input slice={md:.3f}")

    ok_all = True
    notes = []
    for f in ("normalize_gpu.bmp", "normalize_cpu.bmp", "normalize_files_gpu.bmp", "normalize_stddev_cpu.bmp"):
        p = f"{o}/{f}"
        if not os.path.exists(p):
            ok_all = False
            notes.append(f"{f} missing")
            continue
        img = load(p)
        good = img.shape[:2] == (h, w) and float(img.std()) > 0
        ok_all &= good
        notes.append(f"{f} {img.shape[1]}x{img.shape[0]} std={img.std():.1f}")
    record(G, "normalize_output_valid", "pass" if ok_all else "fail", "; ".join(notes))

    for name, sub, size in [("center_crop", "center_crop_batch", (320, 200)), ("copy_make_border", "cmb_batch", (654, 370)),
                            ("gamma_contrast", "gamma_batch", (640, 360)), ("cropandresize", "cropresize_batch", (150, 100))]:
        files = sorted(glob.glob(f"{o}/{sub}/*"))
        sizes = [Image.open(p).size for p in files]
        good = len(files) == 3 and all(s == size for s in sizes)
        record(G, f"{name}_batch_outputs", "pass" if good else "fail", f"{len(files)} files, sizes {sizes}, expected 3 x {size}")
    summary()


if __name__ == "__main__":
    main()
