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

"""GPU-vs-CPU value agreement for every rocpycv operator, plus a numpy reference where one is simple.

Ported from the manual QA session (same cases, inputs, seeds and tolerances). Each case records
  gpu-vs-cpu.<op>::<label>         GPU and CPU outputs agree within the case tolerance
  numpy-ref.<op>::<label>          CPU and GPU outputs both match the numpy reference (when one exists)
  gpu-vs-cpu.reject.<op>::<label>  a documented-unsupported configuration raises on both devices
Known: WarpAffine NEAREST differs between GPU and CPU (FMA rounding, low item); HWC CvtColor raises (M20).
"""
from __future__ import annotations

import math
import time

import numpy as np
import rocpycv as cv
from common import record, summary

GPU, CPU = cv.eDeviceType.GPU, cv.eDeviceType.CPU
DT = cv.eDataType
BT = cv.eBorderType
IT = cv.eInterpolationType
NP_OF = {DT.U8: np.uint8, DT.S8: np.int8, DT.U16: np.uint16, DT.S16: np.int16, DT.U32: np.uint32,
         DT.S32: np.int32, DT.F32: np.float32, DT.F64: np.float64}
RNG = np.random.default_rng(1234)


def rand_img(n, h, w, c, npdt, lo=None, hi=None):
    if np.issubdtype(npdt, np.floating):
        lo = 0.0 if lo is None else lo
        hi = 1.0 if hi is None else hi
        return (RNG.random((n, h, w, c)) * (hi - lo) + lo).astype(npdt)
    info = np.iinfo(npdt)
    lo = info.min if lo is None else lo
    hi = info.max if hi is None else hi
    return RNG.integers(lo, hi, size=(n, h, w, c), dtype=npdt, endpoint=True)


def to_np(t):
    if t.device() == GPU:
        t = t.copy_to(CPU)
    return np.array(np.from_dlpack(t))


def diff_stats(a, b):
    if a.shape != b.shape:
        return None, f"shape {a.shape} vs {b.shape}"
    af, bf = a.astype(np.float64), b.astype(np.float64)
    d = np.abs(af - bf)
    d[np.isnan(af) & np.isnan(bf)] = 0
    return (float(np.nanmax(d)) if d.size else 0.0), None


def _exceed(a, b, tol):
    return np.abs(a.astype(np.float64) - b.astype(np.float64)) > tol


def _examples(a, b, mask):
    idx = np.argwhere(mask)[:3].tolist()
    return f"first idx {idx} got={[a[tuple(i)].item() for i in idx]} ref={[b[tuple(i)].item() for i in idx]}"


def _attempt(op, fn, inputs, tol, max_bad_frac, golden, golden_tol, golden_bad_frac):
    """One comparison. Returns {group: (status, message, seconds)}."""
    out = {}
    t0 = time.perf_counter()
    try:
        cpu_in = [None if a is None else cv.from_dlpack(np.ascontiguousarray(a), lay) for a, lay in inputs]
        gpu_in = [None if t is None else t.copy_to(GPU) for t in cpu_in]
        tc = time.perf_counter()
        oc = to_np(fn(cpu_in, CPU))
        tg = time.perf_counter()
        og = to_np(fn(gpu_in, GPU))
        te = time.perf_counter()
    except Exception as e:  # the operator raised: both checks of this case are errors
        msg = f"{type(e).__name__}: {str(e)[:300]}"
        out[f"gpu-vs-cpu.{op}"] = ("error", msg, time.perf_counter() - t0)
        if golden is not None:
            out[f"numpy-ref.{op}"] = ("error", "operator raised before the reference comparison: " + msg, 0.0)
        return out
    timing = f"cpu {1e3 * (tg - tc):.2f} ms, gpu {1e3 * (te - tg):.2f} ms"
    mx, err = diff_stats(og, oc)
    if err:
        out[f"gpu-vs-cpu.{op}"] = ("fail", f"GPU/CPU {err}", te - t0)
    else:
        bad = _exceed(og, oc, tol)
        nb = int(bad.sum())
        if nb / max(oc.size, 1) > max_bad_frac:
            out[f"gpu-vs-cpu.{op}"] = (
                "fail", f"GPU vs CPU: {nb}/{oc.size} elems exceed tol {tol} (max {mx}); {_examples(og, oc, bad)} ({timing})",
                te - t0)
        else:
            out[f"gpu-vs-cpu.{op}"] = ("pass", f"max|GPU-CPU|={mx} ({timing})", te - t0)
    if golden is None:
        return out
    try:
        ref = golden([a for a, _ in inputs])
    except Exception as e:
        out[f"numpy-ref.{op}"] = ("error", f"reference raised {type(e).__name__}: {e}", 0.0)
        return out
    gt = tol if golden_tol is None else golden_tol
    gbf = max_bad_frac if golden_bad_frac is None else golden_bad_frac
    notes, ok = [], True
    for name, arr in (("CPU", oc), ("GPU", og)):
        m2, e2 = diff_stats(arr, ref)
        if e2:
            ok = False
            notes.append(f"{name} vs numpy {e2}")
            continue
        bad = _exceed(arr, ref, gt)
        nb2 = int(bad.sum())
        if nb2 / max(ref.size, 1) > gbf:
            ok = False
            notes.append(f"{name} vs numpy: {nb2}/{ref.size} exceed tol {gt} (max {m2}); {_examples(arr, ref, bad)}")
        else:
            notes.append(f"{name} max|diff|={m2}")
    out[f"numpy-ref.{op}"] = ("pass" if ok else "fail", "; ".join(notes), 0.0)
    return out


def run_case(op, label, fn, inputs, tol=0.0, max_bad_frac=0.0, golden=None, golden_tol=None, golden_bad_frac=None):
    """inputs: list of (np.ndarray | None, layout). fn(list_of_tensors, device) -> tensor.

    A failing case is re-run once with fresh device copies; a pass on the retry is recorded as flaky.
    """
    args = (op, fn, inputs, tol, max_bad_frac, golden, golden_tol, golden_bad_frac)
    first = _attempt(*args)
    if all(s == "pass" for s, _, _ in first.values()):
        for group, (status, msg, dur) in first.items():
            record(group, label, status, msg, dur)
        return
    second = _attempt(*args)
    for group, (status, msg, dur) in second.items():
        s1, m1, d1 = first.get(group, (status, msg, dur))
        if s1 == "pass":
            record(group, label, s1, m1, d1)
        elif status == "pass":
            record(group, label, "flaky", f"passed on retry; first attempt {s1}: {m1}", dur, attempts=2)
        else:
            record(group, label, status, msg, dur, attempts=2)


def sat(x, npdt):
    if np.issubdtype(npdt, np.floating):
        return x.astype(npdt)
    info = np.iinfo(npdt)
    return np.clip(np.rint(x), info.min, info.max).astype(npdt)


def expect_reject(op, label, fn, inputs):
    """Documented-unsupported configuration: both devices must raise (not crash, not silently succeed)."""
    notes, ok = [], True
    for dev in (CPU, GPU):
        try:
            ts = [cv.from_dlpack(np.ascontiguousarray(a), lay) for a, lay in inputs]
            if dev == GPU:
                ts = [t.copy_to(GPU) for t in ts]
            fn(ts, dev)
            ok = False
            notes.append(f"{dev.name}: accepted a documented-unsupported config")
        except Exception as e:
            notes.append(f"{dev.name} raised {type(e).__name__}: {str(e)[:80]}")
    record(f"gpu-vs-cpu.reject.{op}", label, "pass" if ok else "fail", "; ".join(notes))


SHAPES = [(1, 1, 1), (1, 7, 3), (3, 37, 61), (2, 480, 640), (1, 1081, 1919)]


def main():
    # ---------------- documented-unsupported dtypes ----------------
    expect_reject("flip", "S16 not in documented dtypes", lambda t, d: cv.flip(t[0], 1, None, d),
                  [(rand_img(1, 7, 3, 3, np.int16), cv.NHWC)])
    expect_reject("laplacian", "S16 not in documented dtypes",
                  lambda t, d: cv.laplacian(t[0], 3, 1.0, BT.CONSTANT, stream=None, device=d),
                  [(rand_img(1, 7, 3, 3, np.int16), cv.NHWC)])
    expect_reject("resize", "U16 not in documented dtypes",
                  lambda t, d: cv.resize(t[0], (1, 14, 6, 3), IT.LINEAR, None, d),
                  [(rand_img(1, 7, 3, 3, np.uint16), cv.NHWC)])
    expect_reject("cvtcolor", "F32 not in documented dtypes",
                  lambda t, d: cv.cvtcolor(t[0], cv.eColorConversionCode.COLOR_BGR2RGB, None, d),
                  [(rand_img(1, 7, 3, 3, np.float32), cv.NHWC)])

    # ---------------- flip ----------------
    flip_gold = {1: lambda x: x[0][:, :, ::-1], 0: lambda x: x[0][:, ::-1], -1: lambda x: x[0][:, ::-1, ::-1]}
    for (n, h, w) in SHAPES:
        for c, dt in ((1, DT.U8), (3, DT.U8), (4, DT.F32), (3, DT.S32)):
            for code in (-1, 0, 1):
                a = rand_img(n, h, w, c, NP_OF[dt])
                run_case("flip", f"{n}x{h}x{w}x{c} {dt.name} code={code}",
                         lambda t, d, code=code: cv.flip(t[0], code, None, d), [(a, cv.NHWC)], golden=flip_gold[code])

    # ---------------- center_crop / custom_crop ----------------
    for (n, h, w, c, dt, cw, ch) in [(1, 100, 80, 3, DT.U8, 31, 17), (3, 37, 61, 1, DT.F32, 60, 36),
                                     (2, 480, 640, 4, DT.U16, 224, 224), (1, 9, 9, 3, DT.U8, 9, 9),
                                     (1, 10, 10, 3, DT.U8, 1, 1)]:
        a = rand_img(n, h, w, c, NP_OF[dt])
        x0, y0 = (w >> 1) - (cw >> 1), (h >> 1) - (ch >> 1)
        run_case("center_crop", f"{n}x{h}x{w}x{c} {dt.name} -> {cw}x{ch}",
                 lambda t, d, cw=cw, ch=ch: cv.center_crop(t[0], (cw, ch), None, d), [(a, cv.NHWC)],
                 golden=lambda x, x0=x0, y0=y0, cw=cw, ch=ch: x[0][:, y0:y0 + ch, x0:x0 + cw])
    for (n, h, w, c, dt, bx) in [(1, 100, 80, 3, DT.U8, (5, 7, 31, 17)), (3, 37, 61, 1, DT.F32, (0, 0, 61, 37)),
                                 (2, 480, 640, 4, DT.S32, (600, 400, 40, 80)), (1, 50, 50, 3, DT.U8, (49, 49, 1, 1))]:
        a = rand_img(n, h, w, c, NP_OF[dt])
        x, y, bw, bh = bx
        run_case("custom_crop", f"{n}x{h}x{w}x{c} {dt.name} box={','.join(map(str, bx))}",
                 lambda t, d, bx=bx: cv.custom_crop(t[0], cv.Box(*bx), None, d), [(a, cv.NHWC)],
                 golden=lambda xx, x=x, y=y, bw=bw, bh=bh: xx[0][:, y:y + bh, x:x + bw])

    # ---------------- copymakeborder ----------------
    np_pad = {BT.REPLICATE: "edge", BT.REFLECT: "symmetric", BT.REFLECT101: "reflect", BT.WRAP: "wrap"}
    top, bottom, left, right = 2, 3, 1, 4
    bval = [11.0, 22.0, 33.0, 44.0]

    def cmb_gold(x, bm, c):
        pw = ((0, 0), (top, bottom), (left, right), (0, 0))
        if bm == BT.CONSTANT:
            out = np.pad(x[0], pw, mode="constant")
            mask = np.ones(out.shape[:3], bool)
            mask[:, top:top + x[0].shape[1], left:left + x[0].shape[2]] = False
            for ch in range(c):
                out[..., ch][mask] = bval[ch]
            return out
        return np.pad(x[0], pw, mode=np_pad[bm])

    for (n, h, w, c, dt) in [(1, 7, 3, 3, DT.U8), (2, 37, 61, 4, DT.F32), (1, 480, 640, 1, DT.U16)]:
        for bm in (BT.CONSTANT, BT.REPLICATE, BT.REFLECT, BT.REFLECT101, BT.WRAP):
            a = rand_img(n, h, w, c, NP_OF[dt])
            run_case("copymakeborder", f"{n}x{h}x{w}x{c} {dt.name} {bm.name}",
                     lambda t, d, bm=bm: cv.copymakeborder(t[0], bm, bval, top=top, bottom=bottom, left=left,
                                                           right=right, stream=None, device=d),
                     [(a, cv.NHWC)], golden=lambda x, bm=bm, c=c: cmb_gold(x, bm, c))

    # ---------------- reformat ----------------
    for (n, h, w, c, dt) in [(1, 7, 3, 3, DT.U8), (3, 37, 61, 4, DT.F32), (2, 480, 640, 3, DT.S16)]:
        a = rand_img(n, h, w, c, NP_OF[dt])
        run_case("reformat", f"NHWC->NCHW {n}x{h}x{w}x{c} {dt.name}", lambda t, d: cv.reformat(t[0], cv.NCHW, None, d),
                 [(a, cv.NHWC)], golden=lambda x: x[0].transpose(0, 3, 1, 2))
        b = np.ascontiguousarray(a.transpose(0, 3, 1, 2))
        run_case("reformat", f"NCHW->NHWC {n}x{c}x{h}x{w} {dt.name}", lambda t, d: cv.reformat(t[0], cv.NHWC, None, d),
                 [(b, cv.NCHW)], golden=lambda x: x[0].transpose(0, 2, 3, 1))
        run_case("reformat", f"HWC->CHW {h}x{w}x{c} {dt.name}", lambda t, d: cv.reformat(t[0], cv.CHW, None, d),
                 [(a[0], cv.HWC)], golden=lambda x: x[0].transpose(2, 0, 1))

    # ---------------- convert_to ----------------
    for (src, dst, alpha, beta) in [(DT.U8, DT.F32, 1 / 255.0, 0.0), (DT.U8, DT.S8, 1.0, -128.0),
                                    (DT.F32, DT.U8, 255.0, 0.5), (DT.S16, DT.U8, 0.5, 10.0),
                                    (DT.U8, DT.U16, 257.0, 0.0), (DT.S32, DT.F32, 1e-3, 0.0),
                                    (DT.F32, DT.S16, 1000.0, -500.0), (DT.U16, DT.U8, 1 / 257.0, 0.0)]:
        a = rand_img(2, 37, 61, 3, NP_OF[src])
        flt = dst in (DT.F32, DT.F64)
        run_case("convert_to", f"{src.name}->{dst.name} a={alpha:g} b={beta:g}",
                 lambda t, d, dst=dst, alpha=alpha, beta=beta: cv.convert_to(t[0], dst, alpha, beta, None, d),
                 [(a, cv.NHWC)], tol=(1e-5 if flt else 0),
                 golden=lambda x, dst=dst, alpha=alpha, beta=beta: sat(x[0].astype(np.float64) * alpha + beta, NP_OF[dst]),
                 golden_tol=(0.5 if flt else 1))

    # ---------------- threshold ----------------
    def thr_ref(x, t, m, typ):
        x = x.astype(np.float64)
        if typ == cv.eThresholdType.BINARY:
            return np.where(x > t, m, 0)
        if typ == cv.eThresholdType.BINARY_INV:
            return np.where(x > t, 0, m)
        if typ == cv.eThresholdType.TRUNC:
            return np.where(x > t, t, x)
        if typ == cv.eThresholdType.TOZERO:
            return np.where(x > t, x, 0)
        return np.where(x > t, 0, x)

    def thr_gold(x, th, mv, typ, dt):
        return np.stack([thr_ref(x[0][i], th[i], mv[i], typ) for i in range(x[0].shape[0])]).astype(NP_OF[dt])

    for dt in (DT.U8, DT.U16, DT.S16, DT.F32, DT.F64):
        for typ in (cv.eThresholdType.BINARY, cv.eThresholdType.BINARY_INV, cv.eThresholdType.TRUNC,
                    cv.eThresholdType.TOZERO, cv.eThresholdType.TOZERO_INV):
            n = 3
            a = rand_img(n, 37, 61, 3, NP_OF[dt])
            if dt in (DT.F32, DT.F64):
                th, mv = np.array([0.25, 0.5, 0.75]), np.array([1.0, 0.8, 0.6])
            elif dt == DT.S16:
                th, mv = np.array([-1000.0, 0.0, 1000.0]), np.array([30000.0, 20000.0, 10000.0])
            else:
                th, mv = np.array([50.0, 100.0, 200.0]), np.array([255.0, 128.0, 77.0])
            run_case("threshold", f"{dt.name} {typ.name} per-sample thresh",
                     lambda t, d, typ=typ, n=n: cv.threshold(t[0], t[1], t[2], n, typ, None, d),
                     [(a, cv.NHWC), (th.astype(np.float64), cv.N), (mv.astype(np.float64), cv.N)],
                     golden=lambda x, th=th, mv=mv, typ=typ, dt=dt: thr_gold(x, th, mv, typ, dt),
                     golden_tol=(1e-6 if dt in (DT.F32, DT.F64) else 0))

    # ---------------- histogram ----------------
    for (n, h, w) in [(1, 1, 1), (1, 45, 23), (3, 67, 85), (4, 480, 640)]:
        a = rand_img(n, h, w, 1, np.uint8)
        run_case("histogram", f"{n}x{h}x{w} U8 no-mask", lambda t, d: cv.histogram(t[0], None, None, d), [(a, cv.NHWC)],
                 golden=lambda x: np.stack([np.bincount(x[0][i].ravel(), minlength=256)
                                            for i in range(x[0].shape[0])])[..., None])
        m = RNG.integers(0, 2, size=(n, h, w, 1), dtype=np.uint8)
        run_case("histogram", f"{n}x{h}x{w} U8 masked", lambda t, d: cv.histogram(t[0], t[1], None, d),
                 [(a, cv.NHWC), (m, cv.NHWC)],
                 golden=lambda x: np.stack([np.bincount(x[0][i].ravel()[x[1][i].ravel() != 0], minlength=256)
                                            for i in range(x[0].shape[0])])[..., None])

    # ---------------- cvtcolor ----------------
    def gray(x, r, g, b):
        return sat(x[0][..., r:r + 1] * 0.299 + x[0][..., g:g + 1] * 0.587 + x[0][..., b:b + 1] * 0.114, np.uint8)

    cvt_gold = {"COLOR_BGR2RGB": lambda x: x[0][..., ::-1], "COLOR_RGB2BGR": lambda x: x[0][..., ::-1],
                "COLOR_RGB2GRAY": lambda x: gray(x, 0, 1, 2), "COLOR_BGR2GRAY": lambda x: gray(x, 2, 1, 0)}
    for code in cv.eColorConversionCode.__members__.values():
        if "NV" in code.name:
            continue
        a = rand_img(2, 37, 61, 3, np.uint8)
        run_case("cvtcolor", f"{code.name} 2x37x61 U8", lambda t, d, code=code: cv.cvtcolor(t[0], code, None, d),
                 [(a, cv.NHWC)], tol=1, golden=cvt_gold.get(code.name), golden_tol=1)

    # ---------------- advcvtcolor ----------------
    for spec in cv.eColorSpec.__members__.values():
        for code in cv.eColorConversionCode.__members__.values():
            if "GRAY" in code.name or code.name in ("COLOR_RGB2BGR", "COLOR_BGR2RGB"):
                continue
            h, w = 36, 60
            if code.name.startswith("COLOR_YUV2") and "NV" in code.name:
                a = rand_img(2, h * 3 // 2, w, 1, np.uint8)
            else:
                a = rand_img(2, h, w, 3, np.uint8)
            run_case("advcvtcolor", f"{code.name} {spec.name}",
                     lambda t, d, code=code, spec=spec: cv.advcvtcolor(t[0], code, spec, None, d), [(a, cv.NHWC)], tol=1)

    # ---------------- gamma_contrast ----------------
    def gamma_gold(x, g, npdt, c):
        if np.issubdtype(npdt, np.floating):
            r = np.power(x[0].astype(np.float64), g)
            if c == 4:
                r[..., 3] = x[0][..., 3]
            return r
        mx = float(np.iinfo(npdt).max)
        r = np.power(x[0].astype(np.float64) / mx, g) * mx
        if c == 4:
            r[..., 3] = x[0][..., 3]
        return sat(r, npdt)

    for dt in (DT.U8, DT.U16, DT.U32, DT.F32):
        for c in (1, 3, 4):
            for g in (0.4, 1.0, 2.2):
                a = rand_img(2, 37, 61, c, NP_OF[dt])
                # U32 is processed in float32 (24-bit mantissa): allow ~2 float32 ULPs at 2^32 (512), as in the manual run.
                run_case("gamma_contrast", f"{dt.name} c={c} gamma={g}", lambda t, d, g=g: cv.gamma_contrast(t[0], g, None, d),
                         [(a, cv.NHWC)], tol=(1e-5 if dt == DT.F32 else 512 if dt == DT.U32 else 1),
                         golden=lambda x, g=g, npdt=NP_OF[dt], c=c: gamma_gold(x, g, npdt, c),
                         golden_tol=(1e-4 if dt == DT.F32 else 512 if dt == DT.U32 else 1))

    # ---------------- normalize ----------------
    gs, gsh, eps = 1.5, 3.0, 0.01

    def norm_gold(x, flags, dt):
        s = x[2].astype(np.float64)
        if flags == 1:
            s = 1.0 / np.sqrt(s * s + eps)
        return sat((x[0].astype(np.float64) - x[1]) * s * gs + gsh, NP_OF[dt])

    for dt in (DT.U8, DT.S16, DT.F32):
        for c in (1, 3):
            for flags in (None, 1):
                a = rand_img(2, 37, 61, c, NP_OF[dt])
                base = (RNG.random((1, 1, 1, c)) * 50).astype(np.float32)
                scale = (RNG.random((1, 1, 1, c)) * 2 + 0.1).astype(np.float32)
                run_case("normalize", f"{dt.name} c={c} flags={flags}",
                         lambda t, d, flags=flags: cv.normalize(t[0], t[1], t[2], flags, gs, gsh, eps, None, d),
                         [(a, cv.NHWC), (base, cv.NHWC), (scale, cv.NHWC)], tol=(1e-4 if dt == DT.F32 else 1),
                         golden=lambda x, flags=flags, dt=dt: norm_gold(x, flags, dt), golden_tol=(1e-3 if dt == DT.F32 else 1))

    # ---------------- composite ----------------
    def comp_gold(x, dt, oc_):
        if dt == DT.U8:
            m = x[2].astype(np.float64) / 255.0
            r = sat(x[0] * m + x[1] * (1 - m), np.uint8)
            if oc_ == 4:
                r = np.concatenate([r, np.full(r.shape[:3] + (1,), 255, np.uint8)], -1)
        else:
            m = x[2].astype(np.float64)
            r = x[0] * m + x[1] * (1 - m)
            if oc_ == 4:
                r = np.concatenate([r, np.ones(r.shape[:3] + (1,))], -1)
        return r

    for dt in (DT.U8, DT.F32):
        for oc_ in (3, 4):
            fg, bg = rand_img(2, 37, 61, 3, NP_OF[dt]), rand_img(2, 37, 61, 3, NP_OF[dt])
            mk = rand_img(2, 37, 61, 1, NP_OF[dt])
            run_case("composite", f"{dt.name} out_c={oc_}", lambda t, d, oc_=oc_: cv.composite(t[0], t[1], t[2], oc_, None, d),
                     [(fg, cv.NHWC), (bg, cv.NHWC), (mk, cv.NHWC)], tol=(1e-5 if dt == DT.F32 else 1),
                     golden=lambda x, dt=dt, oc_=oc_: comp_gold(x, dt, oc_), golden_tol=(1e-4 if dt == DT.F32 else 1))

    # ---------------- brightness_contrast ----------------
    for dt in (DT.U8, DT.U16, DT.S16, DT.S32, DT.F32):
        pdt = np.float64 if dt == DT.S32 else np.float32
        a = rand_img(3, 37, 61, 3, NP_OF[dt])
        params = [np.array([1.2, 0.8, 1.0], pdt), np.array([1.1, 0.9, 1.3], pdt), np.array([5.0, -3.0, 0.0], pdt),
                  np.array([100.0, 50.0, 0.0], pdt)]
        if dt == DT.F32:
            params[2] = np.array([0.05, -0.03, 0.0], pdt)
            params[3] = np.array([0.5, 0.25, 0.0], pdt)
        run_case("brightness_contrast", f"{dt.name} per-sample params",
                 lambda t, d: cv.brightness_contrast(t[0], t[1], t[2], t[3], t[4], stream=None, device=d),
                 [(a, cv.NHWC)] + [(p, cv.N) for p in params], tol=(1e-4 if dt == DT.F32 else 1))
        run_case("brightness_contrast", f"{dt.name} defaults (identity)",
                 lambda t, d: cv.brightness_contrast(t[0], stream=None, device=d), [(a, cv.NHWC)],
                 tol=(1e-6 if dt == DT.F32 else 0), golden=lambda x: x[0], golden_tol=(1e-6 if dt == DT.F32 else 0))

    # ---------------- resize ----------------
    def nearest_gold(x, ih, iw, oh, ow):
        # rocCV convention (shipped C++ golden): pixel-centre mapping src = (x+0.5)*scale-0.5, lroundf, clamp.
        def idx(n_out, n_in):
            s = np.float32(n_in) / np.float32(n_out)
            v = (np.arange(n_out, dtype=np.float32) + np.float32(0.5)) * s - np.float32(0.5)
            r = np.where(v >= 0, np.floor(v + 0.5), np.ceil(v - 0.5)).astype(int)
            return np.clip(r, 0, n_in - 1)
        return x[0][:, idx(oh, ih)][:, :, idx(ow, iw)]

    for dt in (DT.U8, DT.F32):
        for interp in (IT.NEAREST, IT.LINEAR, IT.CUBIC):
            for (ih, iw, oh, ow) in [(37, 61, 74, 122), (480, 640, 224, 224), (7, 3, 1, 1), (1, 1, 5, 9), (1081, 1919, 540, 960)]:
                a = rand_img(2, ih, iw, 3, NP_OF[dt])
                gold = None
                if interp == IT.NEAREST:
                    gold = lambda x, ih=ih, iw=iw, oh=oh, ow=ow: nearest_gold(x, ih, iw, oh, ow)  # noqa: E731
                run_case("resize", f"{dt.name} {interp.name} {ih}x{iw}->{oh}x{ow}",
                         lambda t, d, interp=interp, oh=oh, ow=ow: cv.resize(t[0], (2, oh, ow, 3), interp, None, d),
                         [(a, cv.NHWC)], tol=(1e-4 if dt == DT.F32 else 1), golden=gold, golden_tol=0, golden_bad_frac=0.0)
    for interp in (IT.NEAREST, IT.LINEAR, IT.CUBIC):
        a = rand_img(2, 37, 61, 3, np.uint8)
        run_case("resize", f"U8 {interp.name} identity 37x61->37x61",
                 lambda t, d, interp=interp: cv.resize(t[0], (2, 37, 61, 3), interp, None, d), [(a, cv.NHWC)],
                 golden=lambda x: x[0])

    # ---------------- HWC (unbatched) layout on a few operators ----------------
    a3 = rand_img(1, 45, 67, 3, np.uint8)[0]
    run_case("flip", "HWC 45x67x3 U8 code=1", lambda t, d: cv.flip(t[0], 1, None, d), [(a3, cv.HWC)],
             golden=lambda x: x[0][:, ::-1])
    run_case("resize", "HWC 45x67x3 U8 LINEAR ->90x134", lambda t, d: cv.resize(t[0], (90, 134, 3), IT.LINEAR, None, d),
             [(a3, cv.HWC)], tol=1)
    run_case("gaussian", "HWC 45x67x3 U8 k=(3,3)",
             lambda t, d: cv.gaussian(t[0], (3, 3), (1.0, 1.0), BT.REPLICATE, stream=None, device=d), [(a3, cv.HWC)], tol=1)
    run_case("cvtcolor", "HWC BGR2RGB", lambda t, d: cv.cvtcolor(t[0], cv.eColorConversionCode.COLOR_BGR2RGB, None, d),
             [(a3, cv.HWC)], golden=lambda x: x[0][..., ::-1])
    run_case("warp_affine", "HWC LINEAR",
             lambda t, d: cv.warp_affine(t[0], [0.9, 0.2, 5.3, -0.15, 1.1, -3.7], False, IT.LINEAR, BT.CONSTANT,
                                         [0, 0, 0, 0], None, d), [(a3, cv.HWC)], tol=1)
    run_case("copymakeborder", "HWC REPLICATE",
             lambda t, d: cv.copymakeborder(t[0], BT.REPLICATE, [0, 0, 0, 0], top=1, bottom=2, left=3, right=4,
                                            stream=None, device=d),
             [(a3, cv.HWC)], golden=lambda x: np.pad(x[0], ((1, 2), (3, 4), (0, 0)), mode="edge"))

    # ---------------- warp_affine / warp_perspective / rotate ----------------
    m_aff = [0.9, 0.2, 5.3, -0.15, 1.1, -3.7]
    m_per = [1.0, 0.05, 3.0, 0.02, 0.95, -2.0, 0.0005, 0.0003, 1.0]
    for dt in (DT.U8, DT.F32):
        for interp in (IT.NEAREST, IT.LINEAR, IT.CUBIC):
            for bm in (BT.CONSTANT, BT.REPLICATE, BT.REFLECT, BT.WRAP):
                a = rand_img(2, 97, 131, 3, NP_OF[dt])
                run_case("warp_affine", f"{dt.name} {interp.name} {bm.name}",
                         lambda t, d, interp=interp, bm=bm: cv.warp_affine(t[0], m_aff, False, interp, bm, [10, 20, 30, 40],
                                                                           None, d),
                         [(a, cv.NHWC)], tol=(1e-3 if dt == DT.F32 else 1), max_bad_frac=0.0)
                run_case("warp_perspective", f"{dt.name} {interp.name} {bm.name}",
                         lambda t, d, interp=interp, bm=bm: cv.warp_perspective(t[0], m_per, False, interp, bm,
                                                                                [10, 20, 30, 40], None, d),
                         [(a, cv.NHWC)], tol=(1e-3 if dt == DT.F32 else 1), max_bad_frac=0.0)
            for ang in (17.0, 90.0, -45.0):
                a = rand_img(2, 97, 131, 3, NP_OF[dt])
                cx, cy = (131 - 1) / 2, (97 - 1) / 2
                r = math.radians(ang)
                shift = ((1 - math.cos(r)) * cx - math.sin(r) * cy, math.sin(r) * cx + (1 - math.cos(r)) * cy)
                if interp == IT.CUBIC:
                    continue
                run_case("rotate", f"{dt.name} {interp.name} angle={ang}",
                         lambda t, d, ang=ang, shift=shift, interp=interp: cv.rotate(t[0], ang, shift, interp, None, d),
                         [(a, cv.NHWC)], tol=(1e-3 if dt == DT.F32 else 1))
    a = rand_img(2, 97, 131, 3, np.uint8)
    run_case("warp_affine", "identity NEAREST",
             lambda t, d: cv.warp_affine(t[0], [1, 0, 0, 0, 1, 0], False, IT.NEAREST, BT.CONSTANT, [0, 0, 0, 0], None, d),
             [(a, cv.NHWC)], golden=lambda x: x[0])
    run_case("warp_perspective", "identity LINEAR",
             lambda t, d: cv.warp_perspective(t[0], [1, 0, 0, 0, 1, 0, 0, 0, 1], False, IT.LINEAR, BT.CONSTANT,
                                              [0, 0, 0, 0], None, d),
             [(a, cv.NHWC)], golden=lambda x: x[0])
    run_case("rotate", "angle=0 NEAREST", lambda t, d: cv.rotate(t[0], 0.0, (0.0, 0.0), IT.NEAREST, None, d),
             [(a, cv.NHWC)], golden=lambda x: x[0])
    run_case("warp_affine", "integer translate (+5,+3) NEAREST",
             lambda t, d: cv.warp_affine(t[0], [1, 0, 5, 0, 1, 3], False, IT.NEAREST, BT.CONSTANT, [0, 0, 0, 0], None, d),
             [(a, cv.NHWC)], golden=lambda x: np.pad(x[0], ((0, 0), (3, 0), (5, 0), (0, 0)))[:, :97, :131])

    # ---------------- remap ----------------
    for interp in (IT.NEAREST, IT.LINEAR):
        for bm in (BT.CONSTANT, BT.REPLICATE, BT.REFLECT, BT.WRAP):
            n, h, w = 2, 53, 71
            a = rand_img(n, h, w, 3, np.uint8)
            yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
            mp = np.stack([w - 1 - xx + 0.3 * np.sin(yy / 5.0), yy + 0.2 * np.cos(xx / 7.0)], -1).astype(np.float32)
            mp = np.tile(mp[None], (n, 1, 1, 1))
            run_case("remap", f"ABSOLUTE {interp.name} {bm.name}",
                     lambda t, d, interp=interp, bm=bm: cv.remap(t[0], t[1], interp, IT.NEAREST, cv.REMAP_ABSOLUTE, False,
                                                                 bm, [1, 2, 3, 4], None, d),
                     [(a, cv.NHWC), (mp, cv.NHWC)], tol=1)
    a = rand_img(2, 53, 71, 3, np.uint8)
    yy, xx = np.meshgrid(np.arange(53), np.arange(71), indexing="ij")
    mp = np.tile(np.stack([71 - 1 - xx, yy], -1).astype(np.float32)[None], (2, 1, 1, 1))
    run_case("remap", "ABSOLUTE integer mirror map == flip",
             lambda t, d: cv.remap(t[0], t[1], IT.NEAREST, IT.NEAREST, cv.REMAP_ABSOLUTE, False, BT.CONSTANT, [0, 0, 0, 0],
                                   None, d),
             [(a, cv.NHWC), (mp, cv.NHWC)], golden=lambda x: x[0][:, :, ::-1])

    # ---------------- filters: gaussian / averageblur / laplacian / bilateral ----------------
    for dt in (DT.U8, DT.U16, DT.S16, DT.F32):
        for bm in (BT.CONSTANT, BT.REPLICATE, BT.REFLECT, BT.REFLECT101, BT.WRAP):
            a = rand_img(2, 67, 85, 3, NP_OF[dt])
            ftol = 1e-4 if dt == DT.F32 else 1
            run_case("gaussian", f"{dt.name} k=(5,7) s=(1.3,2.1) {bm.name}",
                     lambda t, d, bm=bm: cv.gaussian(t[0], (5, 7), (1.3, 2.1), bm, stream=None, device=d), [(a, cv.NHWC)],
                     tol=ftol)
            run_case("averageblur", f"{dt.name} k=(3,5) anchor=(-1,-1) {bm.name}",
                     lambda t, d, bm=bm: cv.averageblur(t[0], (3, 5), (-1, -1), bm, stream=None, device=d), [(a, cv.NHWC)],
                     tol=ftol)
            for k in ((1, 3) if dt != DT.S16 else ()):
                run_case("laplacian", f"{dt.name} ksize={k} scale=0.5 {bm.name}",
                         lambda t, d, bm=bm, k=k: cv.laplacian(t[0], k, 0.5, bm, stream=None, device=d), [(a, cv.NHWC)],
                         tol=ftol)
    for dt in (DT.U8, DT.F32):
        for bm in (BT.CONSTANT, BT.REPLICATE, BT.REFLECT, BT.WRAP):
            a = rand_img(2, 67, 85, 3, NP_OF[dt])
            sc = 50.0 if dt == DT.U8 else 0.2
            run_case("bilateral_filter", f"{dt.name} d=5 sc={sc} ss=3 {bm.name}",
                     lambda t, d, bm=bm, sc=sc: cv.bilateral_filter(t[0], 5, sc, 3.0, bm, [0, 0, 0, 0], None, d),
                     [(a, cv.NHWC)], tol=(1e-4 if dt == DT.F32 else 1))
    a = rand_img(1, 20, 24, 1, np.uint8)

    def box_ref(x):
        p = np.pad(x[0].astype(np.float64), ((0, 0), (1, 1), (1, 1), (0, 0)), mode="edge")
        return sat(sum(p[:, dy:dy + 20, dx:dx + 24] for dy in range(3) for dx in range(3)) / 9.0, np.uint8)

    run_case("averageblur", "golden 3x3 REPLICATE u8",
             lambda t, d: cv.averageblur(t[0], (3, 3), (-1, -1), BT.REPLICATE, stream=None, device=d), [(a, cv.NHWC)],
             tol=1, golden=box_ref, golden_tol=1)

    # ---------------- nms ----------------
    for (n, nb) in [(1, 4), (3, 64), (20, 837), (1000, 34)]:
        xy = RNG.integers(0, 300, size=(n, nb, 2), dtype=np.int16)
        wh = RNG.integers(5, 80, size=(n, nb, 2), dtype=np.int16)
        boxes = np.concatenate([xy, wh], -1).astype(np.int16)
        scores = RNG.random((n, nb)).astype(np.float32)
        run_case("nms", f"{n} batches x {nb} boxes", lambda t, d: cv.nms(t[0], t[1], 0.3, 0.5, None, d),
                 [(boxes, cv.NWC), (scores, cv.NW)])

    # ---------------- bndbox ----------------
    for c in (3, 4):
        a = rand_img(3, 60, 80, c, np.uint8)
        bb = cv.BndBoxes([[cv.BndBox(cv.Box(5, 5, 30, 20), 2, cv.ColorRGBA(255, 0, 0, 255), cv.ColorRGBA(0, 255, 0, 128))],
                          [cv.BndBox(cv.Box(0, 0, 80, 60), 1, cv.ColorRGBA(0, 0, 255, 255), cv.ColorRGBA(0, 0, 0, 0)),
                           cv.BndBox(cv.Box(40, 30, 50, 50), 3, cv.ColorRGBA(9, 9, 9, 200), cv.ColorRGBA(200, 100, 50, 64))],
                          [cv.BndBox(cv.Box(70, 50, 5, 5), 0, cv.ColorRGBA(1, 2, 3, 4), cv.ColorRGBA(250, 250, 250, 255))]])
        run_case("bndbox", f"3x60x80x{c} U8 mixed boxes", lambda t, d, bb=bb: cv.bndbox(t[0], bb, None, d), [(a, cv.NHWC)],
                 tol=1)
    summary()


if __name__ == "__main__":
    main()
