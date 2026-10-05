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

"""Shared helpers for the rocCV harnesses: result records and small numpy/rocpycv utilities.

Records go straight to $VP_RESULTS through build_tools/results/emit.py (one process
writes thousands of records). Without $VP_RESULTS (a developer running a harness by
hand) records are only printed.
"""
from __future__ import annotations

import os
import re
import sys

_REPO = os.environ.get("VP_REPO", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
sys.path.insert(0, os.path.join(_REPO, "build_tools", "results"))

from emit import append_record  # noqa: E402

SUITE = os.environ.get("VP_SUITE", "roccv")
RESULTS = os.environ.get("VP_RESULTS", "")
LOG = os.environ.get("VP_HARNESS_LOG", "")

_seen: set[str] = set()
_counts: dict[str, int] = {}


def slug(name: str) -> str:
    """Stable, ID-safe name: no '::' (the ID separator) and no whitespace."""
    s = str(name).replace("::", ".")
    s = re.sub(r"\s+", "_", s.strip())
    return s[:400]


def record(group: str, name: str, status: str, message: str = "", duration: float = 0.0, backend: str = "",
           attempts: int = 1) -> str:
    test_id = f"{group}::{slug(name)}"
    if test_id in _seen:
        n = 2
        while f"{test_id}#{n}" in _seen:
            n += 1
        test_id = f"{test_id}#{n}"
    _seen.add(test_id)
    _counts[status] = _counts.get(status, 0) + 1
    print(f"{status.upper():7s} {test_id}  {message[:300]}", flush=True)
    if RESULTS:
        append_record(RESULTS, SUITE, test_id, status, message=message, duration_s=duration, backend=backend, log=LOG,
                      attempts=attempts)
    return test_id


def summary() -> None:
    print("SUMMARY " + " ".join(f"{k}={v}" for k, v in sorted(_counts.items())), flush=True)


def backend_of(text: str) -> str:
    g, c = bool(re.search(r"\bGPU\b", text)), bool(re.search(r"\bCPU\b", text))
    return "GPU" if g and not c else "CPU" if c and not g else ""
