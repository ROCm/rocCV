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

"""Build the inputs for the rocCV samples from the shipped test image (share/roccv/test/data/mug_422.jpg, 3840x2160).

    samples_data.py <out_dir>

Writes mug.jpg (the shipped image), mug_small.png (640x360), mug_flip.jpg, mask.png (grayscale),
batch/ (three 640x360 images), batch_hd/ (three 1280x720 images) and the normalize base/scale files.
"""
from __future__ import annotations

import os
import shutil
import sys

import numpy as np
from PIL import Image


def main():
    out = sys.argv[1]
    src = os.path.join(os.environ["ROCM_PATH"], "share", "roccv", "test", "data", "mug_422.jpg")
    os.makedirs(os.path.join(out, "batch"), exist_ok=True)
    os.makedirs(os.path.join(out, "batch_hd"), exist_ok=True)
    shutil.copyfile(src, os.path.join(out, "mug.jpg"))
    im = Image.open(src).convert("RGB")
    w, h = im.size
    im.transpose(Image.Transpose.FLIP_LEFT_RIGHT).save(os.path.join(out, "mug_flip.jpg"), quality=95)
    yy, xx = np.mgrid[0:h, 0:w]
    mask = (((xx - w / 2) ** 2 + (yy - h / 2) ** 2) < (min(w, h) / 3) ** 2).astype(np.uint8) * 255
    Image.fromarray(mask, "L").save(os.path.join(out, "mask.png"))
    small = im.resize((640, 360), Image.Resampling.BILINEAR)
    small.save(os.path.join(out, "mug_small.png"))
    a = np.asarray(small)
    small.save(os.path.join(out, "batch", "a.png"))
    Image.fromarray(np.ascontiguousarray(a[::-1])).save(os.path.join(out, "batch", "b.png"))
    Image.fromarray(np.ascontiguousarray(a[:, ::-1])).save(os.path.join(out, "batch", "c.jpg"), quality=95)
    hd = np.asarray(im.resize((1280, 720), Image.Resampling.BILINEAR))
    Image.fromarray(hd).save(os.path.join(out, "batch_hd", "a.png"))
    Image.fromarray(np.ascontiguousarray(hd[::-1])).save(os.path.join(out, "batch_hd", "b.png"))
    Image.fromarray(np.ascontiguousarray(hd[:, ::-1])).save(os.path.join(out, "batch_hd", "c.png"))
    with open(os.path.join(out, "base.txt"), "w") as f:
        f.write("1\n1\n120.0\n110.0\n115.0\n")
    with open(os.path.join(out, "scale.txt"), "w") as f:
        f.write("1\n1\n80.0\n75.0\n65.0\n")
    print(f"sample inputs ready in {out}")


if __name__ == "__main__":
    main()
