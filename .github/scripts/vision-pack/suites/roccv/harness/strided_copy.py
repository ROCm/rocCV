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

"""Tensor.copy_to() of non-contiguous (strided) DLPack-imported views (H17).

For each numpy view: strided-copy.<view>::<check>, checks
  from_dlpack       np.from_dlpack(tensor) equals the view (import honours strides; control)
  copy_to_CPU       tensor.copy_to(CPU) equals the view
  copy_to_GPU       tensor.copy_to(GPU).copy_to(CPU) equals the view
  flip_CPU / flip_GPU   a horizontal flip of the view is correct on each device (skip for 2-channel views)
Known (H17): copy_to silently corrupts H crops, ROIs, channel slices, W steps and transposes; no error is raised.
"""
from __future__ import annotations

import numpy as np
import rocpycv as cv
from common import record, summary

GPU, CPU = cv.eDeviceType.GPU, cv.eDeviceType.CPU

base = np.random.default_rng(0).integers(0, 255, size=(2, 10, 12, 3), dtype=np.uint8, endpoint=True)
VIEWS = {
    "contiguous": base,
    "h_crop": base[:, 2:7],
    "w_crop": base[:, :, 3:9],
    "hw_roi": base[:, 2:7, 3:9],
    "hw_roi_single": base[0:1, 2:7, 3:9],
    "batch_step": base[::2],
    "channel_slice": base[..., :2],
    "w_step2": base[:, :, ::2],
    "hw_transpose": base.transpose(0, 2, 1, 3),
}


def check(view_name, check_name, fn, expected, backend=""):
    try:
        got = np.asarray(fn())
    except Exception as e:
        record(f"strided-copy.{view_name}", check_name, "error", f"{type(e).__name__}: {e}", backend=backend)
        return
    if got.shape != expected.shape:
        record(f"strided-copy.{view_name}", check_name, "fail", f"shape {got.shape} != {expected.shape}", backend=backend)
    elif np.array_equal(got, expected):
        record(f"strided-copy.{view_name}", check_name, "pass", backend=backend)
    else:
        record(f"strided-copy.{view_name}", check_name, "fail",
               f"{int((got != expected).sum())}/{expected.size} elements wrong (view strides {expected.strides}); no error raised",
               backend=backend)


def main():
    for name, v in VIEWS.items():
        try:
            t = cv.from_dlpack(v, cv.NHWC)
        except Exception as e:
            for c in ("from_dlpack", "copy_to_CPU", "copy_to_GPU", "flip_CPU", "flip_GPU"):
                record(f"strided-copy.{name}", c, "error", f"from_dlpack raised {type(e).__name__}: {e}")
            continue
        check(name, "from_dlpack", lambda t=t: np.from_dlpack(t), v, "CPU")
        check(name, "copy_to_CPU", lambda t=t: np.from_dlpack(t.copy_to(CPU)), v, "CPU")
        check(name, "copy_to_GPU", lambda t=t: np.from_dlpack(t.copy_to(GPU).copy_to(CPU)), v, "GPU")
        ref = v[:, :, ::-1, :]
        if v.shape[-1] not in (1, 3, 4):
            for c in ("flip_CPU", "flip_GPU"):
                record(f"strided-copy.{name}", c, "skip", f"Flip does not support {v.shape[-1]} channels")
            continue
        check(name, "flip_CPU", lambda t=t: np.from_dlpack(cv.flip(t, 1, None, CPU)), ref, "CPU")
        check(name, "flip_GPU", lambda t=t: np.from_dlpack(cv.flip(t.copy_to(GPU), 1, None, GPU).copy_to(CPU)), ref, "GPU")
    summary()


if __name__ == "__main__":
    main()
