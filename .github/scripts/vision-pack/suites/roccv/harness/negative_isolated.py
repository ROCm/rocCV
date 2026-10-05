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

"""Negative and out-of-bounds inputs, each probe in its own subprocess so a device fault cannot poison later checks.

negative.<CPU|GPU>::<probe>, with an expectation per probe:
  reject    invalid input: a clean exception passes, silently accepting it fails
  tolerate  implementation-defined input: accepting or rejecting both pass
A crash, signal, timeout or GPU memory access fault is an error either way.
oob.<CPU|GPU>::<check> look at the output of accepted inputs.

Known: H18 custom_crop accepts a negative origin (GPU memory access fault that poisons the process; CPU reads
host memory before the buffer); low items: negative padding, singular warp matrices and a NaN rotation angle are accepted.
The GPU negative-origin probes (and the far out-of-bounds CPU one) are deliberate crash probes: they only run when
VP_ROCCV_CRASH_PROBES=1 (run.sh sets it when vp_crash_tests_allowed), otherwise they are recorded as skip.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import time

from common import record, summary

CRASH_OK = os.environ.get("VP_ROCCV_CRASH_PROBES", "0") == "1"
TIMEOUT = int(os.environ.get("VP_ROCCV_PROBE_TIMEOUT", "300"))

PRE = """
import numpy as np, rocpycv as cv
GPU, CPU = cv.eDeviceType.GPU, cv.eDeviceType.CPU
D = {dev}
def T(a, lay=cv.NHWC):
    t = cv.from_dlpack(np.ascontiguousarray(a), lay)
    return t.copy_to(GPU) if D == GPU else t
def N(t):
    return np.from_dlpack(t.copy_to(CPU)) if t.device() == GPU else np.from_dlpack(t)
t = T(np.arange(16*16*3, dtype=np.uint32).astype(np.uint8).reshape(1, 16, 16, 3))
"""

# name -> (expectation, statement producing r, crash_probe_devices)
PROBES = {
    "custom_crop_Box(-5,-5,4,4)": ("reject", "r = cv.custom_crop(t, cv.Box(-5, -5, 4, 4), None, D)", {"GPU"}),
    "custom_crop_Box(-1000,0,4,4)": ("reject", "r = cv.custom_crop(t, cv.Box(-1000, 0, 4, 4), None, D)", {"GPU"}),
    "custom_crop_Box(0,-100000,4,4)": ("reject", "r = cv.custom_crop(t, cv.Box(0, -100000, 4, 4), None, D)", {"GPU", "CPU"}),
    "custom_crop_Box(2,2,0,0)": ("reject", "r = cv.custom_crop(t, cv.Box(2, 2, 0, 0), None, D)", set()),
    "center_crop_(0,0)": ("reject", "r = cv.center_crop(t, (0, 0), None, D)", set()),
    "gaussian_even_kernel_(4,4)": ("reject", "r = cv.gaussian(t, (4, 4), (1.0, 1.0), cv.eBorderType.CONSTANT, stream=None, device=D)", set()),
    "gaussian_sigma=0_k=(3,3)": ("tolerate", "r = cv.gaussian(t, (3, 3), (0.0, 0.0), cv.eBorderType.CONSTANT, stream=None, device=D)", set()),
    "averageblur_zero_kernel_(0,0)": ("reject", "r = cv.averageblur(t, (0, 0), (-1, -1), cv.eBorderType.CONSTANT, stream=None, device=D)", set()),
    "averageblur_anchor_outside_kernel": ("reject", "r = cv.averageblur(t, (3, 3), (7, 7), cv.eBorderType.CONSTANT, stream=None, device=D)", set()),
    "laplacian_ksize=5": ("reject", "r = cv.laplacian(t, 5, 1.0, cv.eBorderType.CONSTANT, stream=None, device=D)", set()),
    "bilateral_diameter=-1_sigma=0": ("tolerate", "r = cv.bilateral_filter(t, -1, 0.0, 0.0, cv.eBorderType.CONSTANT, [0,0,0,0], None, D)", set()),
    "threshold_2_thresholds_for_batch_of_1": ("reject", "v = T(np.array([1.0, 2.0]), cv.N); r = cv.threshold(t, v, v, 2, cv.eThresholdType.BINARY, None, D)", set()),
    "warp_affine_singular_matrix": ("reject", "r = cv.warp_affine(t, [0, 0, 0, 0, 0, 0], False, cv.eInterpolationType.LINEAR, cv.eBorderType.CONSTANT, [0,0,0,0], None, D)", set()),
    "warp_perspective_singular_matrix": ("reject", "r = cv.warp_perspective(t, [0]*9, False, cv.eInterpolationType.LINEAR, cv.eBorderType.CONSTANT, [0,0,0,0], None, D)", set()),
    "warp_affine_xform_too_short": ("reject", "r = cv.warp_affine(t, [1, 0, 0], False, cv.eInterpolationType.LINEAR, cv.eBorderType.CONSTANT, [0,0,0,0], None, D)", set()),
    "copymakeborder_negative_top": ("reject", "r = cv.copymakeborder(t, cv.eBorderType.CONSTANT, [0,0,0,0], top=-1, bottom=0, left=0, right=0, stream=None, device=D)", set()),
    "copymakeborder_REFLECT_pad_larger_than_image": ("tolerate", "r = cv.copymakeborder(t, cv.eBorderType.REFLECT, [0,0,0,0], top=40, bottom=40, left=40, right=40, stream=None, device=D)", set()),
    "reformat_NHWC_to_NW": ("reject", "r = cv.reformat(t, cv.NW, None, D)", set()),
    "histogram_3_channel_input": ("reject", "r = cv.histogram(t, None, None, D)", set()),
    "convert_to_4S16": ("reject", "r = cv.convert_to(t, getattr(cv.eDataType, '4S16'), 1.0, 0.0, None, D)", set()),
    "resize_to_1x1_CUBIC": ("tolerate", "r = cv.resize(t, (1, 1, 1, 3), cv.eInterpolationType.CUBIC, None, D)", set()),
    "rotate_NaN_angle": ("reject", "r = cv.rotate(t, float('nan'), (0.0, 0.0), cv.eInterpolationType.LINEAR, None, D)", set()),
    "remap_map_smaller_than_output": ("tolerate", "r = cv.remap(t, T(np.zeros((1, 4, 4, 2), np.float32)), cv.eInterpolationType.NEAREST, cv.eInterpolationType.NEAREST, cv.REMAP_ABSOLUTE, False, cv.eBorderType.CONSTANT, [0,0,0,0], None, D)", set()),
    "nms_iou=2.0_score=-1": ("reject", "r = cv.nms(T(np.zeros((1, 4, 4), np.int16), cv.NWC), T(np.zeros((1, 4), np.float32), cv.NW), -1.0, 2.0, None, D)", set()),
    "bndbox_box_outside_image": ("tolerate", "r = cv.bndbox(t, cv.BndBoxes([[cv.BndBox(cv.Box(-50, -50, 500, 500), 3, cv.ColorRGBA(1,2,3,255), cv.ColorRGBA(0,0,0,0))]]), None, D)", set()),
    "bndbox_huge_thickness": ("tolerate", "r = cv.bndbox(t, cv.BndBoxes([[cv.BndBox(cv.Box(2, 2, 8, 8), 100000, cv.ColorRGBA(1,2,3,255), cv.ColorRGBA(0,0,0,0))]]), None, D)", set()),
    "bndbox_fewer_box_lists_than_batch": ("tolerate", "t2 = T(np.zeros((3, 16, 16, 3), np.uint8)); r = cv.bndbox(t2, cv.BndBoxes([[cv.BndBox(cv.Box(2, 2, 8, 8), 1, cv.ColorRGBA(1,2,3,255), cv.ColorRGBA(0,0,0,0))]]), None, D)", set()),
}
POST = """
N(r)
print("@@ACCEPTED", r.shape())
"""

# Output checks on accepted inputs: the snippet prints "@@RESULT <pass|fail> <message>".
OOB = {
    "custom_crop_negative_origin_content": ({"CPU"}, """
a = np.full((1, 64, 64, 3), 200, np.uint8)
try:
    out = N(cv.custom_crop(T(a), cv.Box(-8, -8, 16, 16), None, D))
except Exception as e:
    print("@@RESULT pass rejected:", e)
else:
    bad = int((out != 200).sum())
    print("@@RESULT", "fail" if bad else "pass", f"crop(-8,-8,16,16) of an all-200 image: {bad}/{out.size} output values are not 200 (out-of-bounds memory)")
"""),
    "copymakeborder_pad_larger_than_image_vs_numpy": ({"CPU", "GPU"}, """
modes = {cv.eBorderType.REPLICATE: "edge", cv.eBorderType.REFLECT: "symmetric", cv.eBorderType.REFLECT101: "reflect", cv.eBorderType.WRAP: "wrap"}
a = np.random.default_rng(0).integers(0, 255, size=(1, 16, 16, 3), dtype=np.uint8, endpoint=True)
res = []
for bm, npm in modes.items():
    ref = np.pad(a, ((0, 0), (40, 40), (40, 40), (0, 0)), mode=npm)
    out = N(cv.copymakeborder(T(a), bm, [0,0,0,0], top=40, bottom=40, left=40, right=40, stream=None, device=D))
    res.append((bm.name, int((out != ref).sum())))
print("@@RESULT", "fail" if any(n for _, n in res) else "pass", "mismatches vs numpy: " + ", ".join(f"{m}={n}" for m, n in res))
"""),
    "bndbox_missing_box_lists_leave_images_unchanged": ({"CPU", "GPU"}, """
src = T(np.zeros((3, 16, 16, 3), np.uint8))
bb = cv.BndBoxes([[cv.BndBox(cv.Box(2, 2, 8, 8), 1, cv.ColorRGBA(255, 0, 0, 255), cv.ColorRGBA(0, 0, 0, 0))]])
out = N(cv.bndbox(src, bb, None, D))
nz = [int((out[i] != 0).any(-1).sum()) for i in range(3)]
print("@@RESULT", "pass" if nz[0] > 0 and nz[1] == 0 and nz[2] == 0 else "fail", f"non-zero pixels per image {nz}")
"""),
}

FAULT_MARKERS = ("memory access fault", "page not present", "hiperrorillegaladdress", "illegal address")


def run_snippet(code):
    t0 = time.perf_counter()
    try:
        p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        return None, "", f"timeout after {TIMEOUT}s", time.perf_counter() - t0
    return p.returncode, p.stdout, p.stderr, time.perf_counter() - t0


def classify_crash(rc, out, err):
    low = (out + err).lower()
    if rc is None:
        return f"{err}"
    if any(m in low for m in FAULT_MARKERS):
        return f"GPU memory access fault (rc {rc}): " + " / ".join(ln for ln in err.splitlines() if ln.strip())[-400:]
    if rc < 0 or rc > 1:
        return f"process died (rc {rc}): {err.strip()[-400:]}"
    return ""


def main():
    for name, (expect, stmt, crash_devs) in PROBES.items():
        for dev in ("CPU", "GPU"):
            group = f"negative.{dev}"
            if dev in crash_devs and not CRASH_OK:
                record(group, name, "skip", "deliberate crash probe (out-of-bounds access); runs only where crash tests are "
                       "allowed (CI container)", backend=dev)
                continue
            code = (PRE.replace("{dev}", dev) + "try:\n" + textwrap.indent(stmt + "\n" + POST, "    ")
                    + "except Exception as e:\n    print('@@REJECTED', type(e).__name__ + ':', str(e)[:200])\n")
            rc, out, err, dur = run_snippet(code)
            crash = classify_crash(rc, out, err)
            last = next((ln for ln in reversed(out.splitlines()) if ln.startswith("@@")), "")
            if crash:
                record(group, name, "error", crash + (f" | {last}" if last else ""), dur, dev)
            elif last.startswith("@@REJECTED"):
                record(group, name, "pass", "rejected: " + last[len("@@REJECTED "):], dur, dev)
            elif last.startswith("@@ACCEPTED"):
                if expect == "reject":
                    record(group, name, "fail", "invalid input accepted without validation -> shape " + last[len("@@ACCEPTED "):],
                           dur, dev)
                else:
                    record(group, name, "pass", "accepted (implementation-defined) -> shape " + last[len("@@ACCEPTED "):], dur, dev)
            else:
                record(group, name, "error", f"probe produced no verdict (rc {rc}): {err.strip()[-300:]}", dur, dev)

    for name, (devs, body) in OOB.items():
        for dev in sorted(devs):
            rc, out, err, dur = run_snippet(PRE.replace("{dev}", dev) + textwrap.dedent(body))
            crash = classify_crash(rc, out, err)
            last = next((ln for ln in reversed(out.splitlines()) if ln.startswith("@@RESULT")), "")
            if crash:
                record(f"oob.{dev}", name, "error", crash, dur, dev)
            elif last:
                _, status, msg = (last.split(" ", 2) + [""])[:3]
                record(f"oob.{dev}", name, status if status in ("pass", "fail") else "error", msg, dur, dev)
            else:
                record(f"oob.{dev}", name, "error", f"no verdict (rc {rc}): {err.strip()[-300:]}", dur, dev)
    summary()


if __name__ == "__main__":
    main()
