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

"""DLPack / tensor interop probes for rocpycv.

  dlpack_interop.py            numpy-only probes (dlpack::<name>), ported from the manual QA session
  dlpack_interop.py --torch    exchange with PyTorch (dlpack.torch::<name>); needs a ROCm torch (extended image)
Known (H17): the noncontig_* probes and noncontig_view_into_operator fail (strided copies are corrupted).
"""
from __future__ import annotations

import gc
import sys

import numpy as np
import rocpycv as cv
from common import record, summary

GPU, CPU = cv.eDeviceType.GPU, cv.eDeviceType.CPU
NP_OF = {
    cv.eDataType.U8: np.uint8, cv.eDataType.S8: np.int8, cv.eDataType.U16: np.uint16, cv.eDataType.S16: np.int16,
    cv.eDataType.U32: np.uint32, cv.eDataType.S32: np.int32, cv.eDataType.F32: np.float32, cv.eDataType.F64: np.float64,
}
GROUP = "dlpack"


def check(name, fn):
    try:
        detail = fn()
        record(GROUP, name, "pass", str(detail or ""))
    except AssertionError as e:
        record(GROUP, name, "fail", f"AssertionError: {e}")
    except Exception as e:
        record(GROUP, name, "error", f"{type(e).__name__}: {e}")


def rand(shape, npdt, seed=0):
    rng = np.random.default_rng(seed)
    if np.issubdtype(npdt, np.floating):
        return rng.random(shape).astype(npdt)
    info = np.iinfo(npdt)
    return rng.integers(info.min, info.max, size=shape, dtype=npdt, endpoint=True)


def to_np(t):
    if t.device() == GPU:
        t = t.copy_to(CPU)
    return np.from_dlpack(t)


def expect_raise(fn, label):
    try:
        r = fn()
    except Exception as e:
        return f"{label}: raised {type(e).__name__}: {str(e)[:160]}"
    raise AssertionError(f"{label}: accepted silently -> {r.dtype() if hasattr(r, 'dtype') else r}")


def numpy_probes():
    for rdt, npdt in NP_OF.items():
        def rt(rdt=rdt, npdt=npdt):
            a = rand((2, 5, 7, 3), npdt)
            t = cv.from_dlpack(a, cv.NHWC)
            assert t.dtype() == rdt, f"dtype {t.dtype()} != {rdt}"
            assert t.shape() == [2, 5, 7, 3], t.shape()
            assert t.device() == CPU
            b = np.from_dlpack(t)
            assert b.dtype == npdt and b.shape == a.shape
            assert np.array_equal(a, b)
            return f"strides={t.strides()}"
        check(f"roundtrip_cpu_{rdt.name}", rt)

    for rdt, npdt in NP_OF.items():
        def h2d2h(npdt=npdt):
            a = rand((3, 17, 33, 4), npdt, seed=1)
            g = cv.from_dlpack(a, cv.NHWC).copy_to(GPU)
            assert g.device() == GPU
            assert g.__dlpack_device__()[0] == 10, f"GPU dlpack device {g.__dlpack_device__()}"
            assert np.array_equal(a, np.from_dlpack(g.copy_to(CPU)))
            assert np.array_equal(to_np(g.copy_to(GPU)), a), "GPU->GPU copy mismatch"
            return f"gpu dlpack_device={g.__dlpack_device__()}"
        check(f"h2d2h_{rdt.name}", h2d2h)

    check("dlpack_device_cpu", lambda: str(cv.Tensor([2, 2], cv.NW, cv.eDataType.U8, CPU).__dlpack_device__()))
    check("dlpack_device_gpu", lambda: str(cv.Tensor([2, 2], cv.NW, cv.eDataType.U8, GPU).__dlpack_device__()))

    def zero_copy():
        a = np.zeros((1, 4, 4, 1), np.uint8)
        t = cv.from_dlpack(a, cv.NHWC)
        a[0, 1, 2, 0] = 77
        b = np.from_dlpack(t)
        same_ptr = b.__array_interface__["data"][0] == a.__array_interface__["data"][0]
        return f"numpy->tensor zero-copy={bool(b[0, 1, 2, 0] == 77)}; same pointer={same_ptr}; writeable={b.flags.writeable}"
    check("zero_copy_semantics", zero_copy)

    def lifetime():
        a = rand((1, 64, 64, 3), np.uint8, seed=5)
        ref = a.copy()
        t = cv.from_dlpack(a, cv.NHWC)
        del a
        gc.collect()
        _junk = [np.full((1, 64, 64, 3), 0xAB, np.uint8) for _ in range(50)]
        assert np.array_equal(np.from_dlpack(t), ref), "data changed after source array was dropped"
        g = t.copy_to(GPU)
        del t
        gc.collect()
        assert np.array_equal(to_np(g), ref)
    check("lifetime_after_source_deleted", lifetime)

    def self_import_gpu():
        a = rand((2, 8, 8, 3), np.float32, seed=6)
        g2 = cv.from_dlpack(cv.from_dlpack(a, cv.NHWC).copy_to(GPU), cv.NHWC)
        assert g2.device() == GPU, g2.device()
        assert np.array_equal(to_np(g2), a)
        return f"device={g2.device()}"
    check("from_dlpack_of_gpu_tensor", self_import_gpu)

    def self_import_cpu():
        a = rand((2, 8, 8, 3), np.int16, seed=7)
        assert np.array_equal(np.from_dlpack(cv.from_dlpack(cv.from_dlpack(a, cv.NHWC), cv.NHWC)), a)
    check("from_dlpack_of_cpu_tensor", self_import_cpu)

    def np_from_gpu():
        g = cv.Tensor([1, 4, 4, 1], cv.NHWC, cv.eDataType.U8, GPU)
        try:
            np.from_dlpack(g)
        except (BufferError, RuntimeError, TypeError, ValueError) as e:
            return f"raised {type(e).__name__}: {e}"
        raise AssertionError("np.from_dlpack(GPU tensor) unexpectedly succeeded")
    check("np_from_dlpack_gpu_raises", np_from_gpu)

    def noncontig(view_fn, label):
        v = view_fn(rand((2, 10, 12, 3), np.uint8, seed=8))
        try:
            t = cv.from_dlpack(v, cv.NHWC)
        except Exception as e:
            return f"{label}: rejected with {type(e).__name__}: {e}"
        got = np.from_dlpack(t.copy_to(GPU).copy_to(CPU))
        if got.shape != v.shape:
            raise AssertionError(f"{label}: shape {got.shape} != {v.shape}")
        if not np.array_equal(got, v):
            raise AssertionError(f"{label}: accepted non-contiguous view (strides={t.strides()}) but GPU round-trip data "
                                 "differs from the view")
        return f"{label}: accepted, strides={t.strides()}, data correct"
    check("noncontig_slice_w", lambda: noncontig(lambda b: b[:, :, ::2, :], "step-2 W slice"))
    check("noncontig_crop", lambda: noncontig(lambda b: b[:, 2:7, 3:9, :], "ROI crop"))
    check("noncontig_channel_slice", lambda: noncontig(lambda b: b[..., :2], "channel slice"))
    check("noncontig_transpose_hw", lambda: noncontig(lambda b: b.transpose(0, 2, 1, 3), "H/W transpose"))

    def noncontig_op():
        v = rand((1, 10, 12, 3), np.uint8, seed=9)[:, 2:7, 3:9, :]
        t = cv.from_dlpack(v, cv.NHWC)
        ref = v[:, :, ::-1, :]
        assert np.array_equal(np.from_dlpack(cv.flip(t, 1, None, CPU)), ref), "CPU flip of ROI view wrong"
        assert np.array_equal(to_np(cv.flip(t.copy_to(GPU), 1, None, GPU)), ref), "GPU flip of ROI view wrong"
        return "flip on ROI view correct on both devices"
    check("noncontig_view_into_operator", noncontig_op)

    for npdt in [np.float16, np.int64, np.uint64, np.bool_, np.complex64]:
        nm = np.dtype(npdt).name
        check(f"unsupported_dtype_{nm}",
              lambda npdt=npdt, nm=nm: expect_raise(lambda: cv.from_dlpack(np.zeros((1, 2, 2, 1), npdt), cv.NHWC), nm))
    check("rank_mismatch_3d_as_NHWC", lambda: expect_raise(lambda: cv.from_dlpack(np.zeros((2, 2, 1), np.uint8), cv.NHWC),
                                                           "3D as NHWC"))
    check("rank_mismatch_4d_as_HWC", lambda: expect_raise(lambda: cv.from_dlpack(np.zeros((1, 2, 2, 1), np.uint8), cv.HWC),
                                                          "4D as HWC"))
    check("from_dlpack_non_dlpack_object", lambda: expect_raise(lambda: cv.from_dlpack([1, 2, 3], cv.N), "python list"))

    def layouts():
        out = []
        for lay, shape in [(cv.NHWC, [2, 3, 4, 3]), (cv.HWC, [3, 4, 3]), (cv.NCHW, [2, 3, 4, 5]), (cv.CHW, [3, 4, 5]),
                           (cv.NC, [4, 5]), (cv.NW, [4, 5]), (cv.NWC, [2, 5, 4]), (cv.N, [7])]:
            for dev in (CPU, GPU):
                t = cv.Tensor(shape, lay, cv.eDataType.F32, dev)
                assert t.shape() == shape and t.layout() == lay and t.ndim() == len(shape)
            out.append(f"{lay.name}:{t.strides()}")
        return "; ".join(out)
    check("tensor_ctor_layouts", layouts)

    def reshape():
        a = rand((2, 4, 6, 3), np.uint8, seed=10)
        for dev in (CPU, GPU):
            r = cv.from_dlpack(a, cv.NHWC).copy_to(dev).reshape([8, 18], cv.NW)
            assert r.shape() == [8, 18] and r.layout() == cv.NW
            assert np.array_equal(to_np(r), a.reshape(8, 18))
        return "reshape preserves data on CPU and GPU"
    check("tensor_reshape", reshape)
    check("tensor_reshape_bad_size", lambda: expect_raise(
        lambda: cv.Tensor([2, 4, 6, 3], cv.NHWC, cv.eDataType.U8, GPU).reshape([5, 5], cv.NW), "reshape to wrong size"))
    check("tensor_ctor_bad_rank", lambda: expect_raise(lambda: cv.Tensor([2, 3], cv.NHWC, cv.eDataType.U8, GPU),
                                                       "2 dims for NHWC"))
    check("tensor_ctor_negative_dim", lambda: expect_raise(lambda: cv.Tensor([1, -4, 4, 1], cv.NHWC, cv.eDataType.U8, GPU),
                                                           "negative dim"))
    check("tensor_ctor_zero_dim", lambda: expect_raise(lambda: cv.Tensor([1, 0, 4, 1], cv.NHWC, cv.eDataType.U8, GPU),
                                                       "zero dim"))

    def dlpack_kwargs():
        t = cv.Tensor([1, 2, 2, 1], cv.NHWC, cv.eDataType.U8, CPU)
        notes = [f"__dlpack__() -> {type(t.__dlpack__()).__name__}"]
        t.__dlpack__(stream=None)
        notes.append("stream=None ok")
        try:
            t.__dlpack__(max_version=(1, 0))
            notes.append("max_version accepted")
        except TypeError:
            notes.append("max_version -> TypeError (legacy-only producer)")
        cv.Tensor([1, 2, 2, 1], cv.NHWC, cv.eDataType.U8, GPU).__dlpack__(stream=1)
        notes.append("GPU stream=1 ok")
        return "; ".join(notes)
    check("dlpack_protocol_kwargs", dlpack_kwargs)

    def multiple_exports():
        t = cv.from_dlpack(np.arange(16, dtype=np.uint8).reshape(1, 4, 4, 1), cv.NHWC)
        assert np.array_equal(np.from_dlpack(t), np.from_dlpack(t))
        return "multiple exports from one tensor ok"
    check("multiple_exports", multiple_exports)

    def large():
        a = np.random.default_rng(11).integers(0, 255, size=(8, 2160, 3840, 4), dtype=np.uint8, endpoint=True)
        assert np.array_equal(to_np(cv.from_dlpack(a, cv.NHWC).copy_to(GPU)), a)
        return f"{a.nbytes / 1e6:.0f} MB ok"
    check("large_roundtrip_265MB", large)


def torch_probes():
    global GROUP
    GROUP = "dlpack.torch"
    import torch

    gpu_ok = torch.cuda.is_available() and getattr(torch.version, "hip", None)
    a = rand((2, 16, 24, 3), np.uint8, seed=20)

    def cpu_torch_to_roccv():
        tt = torch.from_numpy(a.copy())
        t = cv.from_dlpack(tt, cv.NHWC)
        assert t.device() == CPU
        assert np.array_equal(np.from_dlpack(t), a)
        return f"torch {torch.__version__}"
    check("torch_cpu_to_roccv", cpu_torch_to_roccv)

    def roccv_cpu_to_torch():
        tt = torch.from_dlpack(cv.from_dlpack(a, cv.NHWC))
        assert tt.device.type == "cpu"
        assert np.array_equal(tt.numpy(), a)
    check("roccv_cpu_to_torch", roccv_cpu_to_torch)

    if not gpu_ok:
        for n in ("torch_gpu_to_roccv", "roccv_gpu_to_torch", "roccv_op_on_torch_gpu_tensor", "torch_gpu_noncontig_to_roccv"):
            record(GROUP, n, "blocked", f"torch {torch.__version__} has no ROCm GPU (hip={getattr(torch.version, 'hip', None)})")
        return

    def torch_gpu_to_roccv():
        tg = torch.from_numpy(a.copy()).to("cuda")
        t = cv.from_dlpack(tg, cv.NHWC)
        assert t.device() == GPU, t.device()
        assert np.array_equal(to_np(t), a)
    check("torch_gpu_to_roccv", torch_gpu_to_roccv)

    def roccv_gpu_to_torch():
        tt = torch.from_dlpack(cv.from_dlpack(a, cv.NHWC).copy_to(GPU))
        assert tt.device.type == "cuda", tt.device
        assert np.array_equal(tt.cpu().numpy(), a)
    check("roccv_gpu_to_torch", roccv_gpu_to_torch)

    def op_on_torch_tensor():
        tg = torch.from_numpy(a.copy()).to("cuda")
        out = cv.flip(cv.from_dlpack(tg, cv.NHWC), 1, None, GPU)
        res = torch.from_dlpack(out)
        torch.cuda.synchronize()
        assert np.array_equal(res.cpu().numpy(), a[:, :, ::-1])
    check("roccv_op_on_torch_gpu_tensor", op_on_torch_tensor)

    def noncontig():
        tg = torch.from_numpy(a.copy()).to("cuda")[:, 2:10, 3:15, :]
        t = cv.from_dlpack(tg, cv.NHWC)
        got = to_np(cv.flip(t, 1, None, GPU))
        assert np.array_equal(got, a[:, 2:10, 3:15, :][:, :, ::-1]), "flip of a strided torch GPU view is wrong"
    check("torch_gpu_noncontig_to_roccv", noncontig)


if __name__ == "__main__":
    if "--torch" in sys.argv:
        torch_probes()
    else:
        numpy_probes()
    summary()
