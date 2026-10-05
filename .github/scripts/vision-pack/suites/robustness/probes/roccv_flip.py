#!/usr/bin/env python3
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

"""rocpycv Flip (codes 1, 0, -1) against numpy and judge the outcome (see verdict.py).

    roccv_flip.py --device cpu|gpu --mode correct|honest|error [--copy-to]

gpu: host -> copy_to(GPU) -> flip on a Stream -> synchronize -> copy_to(CPU).
cpu: flip on the host tensor from from_dlpack, stream=None, read back with np.from_dlpack; with
--copy-to the input and output also go through Tensor.copy_to(CPU), which needs HIP (L5).
"""
from __future__ import annotations

import argparse

import numpy as np
from verdict import finish


def expected(src: np.ndarray, code: int) -> np.ndarray:
    if code > 0:
        return src[:, :, ::-1, :]
    if code == 0:
        return src[:, ::-1, :, :]
    return src[:, ::-1, ::-1, :]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", required=True, choices=["cpu", "gpu"])
    ap.add_argument("--mode", required=True, choices=["correct", "honest", "error"])
    ap.add_argument("--copy-to", action="store_true")
    a = ap.parse_args()
    try:
        import rocpycv as cv
    except Exception as e:  # noqa: BLE001 - an import failure is reported as a clean error
        finish(a.mode, "clean_error", f"import rocpycv: {type(e).__name__}: {e}")
    gpu, cpu = cv.eDeviceType.GPU, cv.eDeviceType.CPU
    src = ((np.arange(2 * 5 * 7, dtype=np.uint8).reshape(2, 5, 7, 1) * 3) % 251).astype(np.uint8)
    wrong = 0
    try:
        for code in (1, 0, -1):
            if a.device == "gpu":
                t = cv.from_dlpack(src, cv.eTensorLayout.NHWC).copy_to(gpu)
                s = cv.Stream()
                out = cv.flip(t, code, s, gpu)
                s.synchronize()
                got = np.from_dlpack(out.copy_to(cpu)).reshape(src.shape)
            else:
                t = cv.from_dlpack(src, cv.eTensorLayout.NHWC)
                if a.copy_to:
                    t = t.copy_to(cpu)
                out = cv.flip(t, code, None, cpu)
                if a.copy_to:
                    out = out.copy_to(cpu)
                got = np.from_dlpack(out).reshape(src.shape)
            bad = int((got != expected(src, code)).sum())
            print(f"[{a.device} flip={code:2d}] mismatches={bad}/{src.size} first={got.ravel()[:3].tolist()}", flush=True)
            wrong += bad
    except Exception as e:  # noqa: BLE001 - any exception is the library reporting an error
        finish(a.mode, "clean_error", f"{type(e).__name__}: {e}")
    finish(a.mode, "wrong" if wrong else "correct", f"{wrong} wrong elements over 3 flip codes")


if __name__ == "__main__":
    main()
