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

"""Convert roccv_bench output, collapsed by analyze_results.py --export, into vp_perf metrics.

    bench_to_perf.py <results_clean.csv> <perf.json>

One metric per benchmark configuration: bench.<category>.<type>.<params>.<shape>, the median of the kept samples
in ms (roccv_bench reports seconds). Prints the number of metrics; exits 1 when there are none.
"""
from __future__ import annotations

import csv
import json
import re
import sys

FIXED = {"category", "name", "samples", "height", "width", "runs", "warmupRuns", "shape", "n_total", "n_kept", "n_dropped",
         "mean", "median", "std", "q1", "q3", "min", "max", "read_memory_bytes", "written_memory_bytes", "gpu", "cpu",
         "cpu_threads"}


def clean(v: str) -> str:
    """eBorderType::BORDER_TYPE_REFLECT -> BORDER_TYPE_REFLECT, 3.0 -> 3 (pandas turns int columns with gaps into floats)."""
    v = v.rsplit("::", 1)[-1]
    if re.fullmatch(r"-?\d+\.0", v):
        v = v[:-2]
    return v.replace(".", "p")


def main():
    rows = list(csv.DictReader(open(sys.argv[1], newline="")))
    metrics = []
    for r in rows:
        try:
            med = float(r["median"])
        except (KeyError, ValueError):
            continue
        if med != med:
            continue
        params = ".".join(f"{k}={clean(v)}" for k, v in sorted(r.items()) if k not in FIXED and v not in ("", None))
        name = ".".join(p for p in ("bench", r.get("category", "?"), r.get("name", "?"), params, r.get("shape", "")) if p)
        metrics.append({"name": name.replace(" ", ""), "value": round(med * 1e3, 5), "unit": "ms", "lower_is_better": True,
                        "backend": "GPU" if r.get("name") == "GPU" else "CPU"})
    json.dump({"metrics": metrics, "source": "roccv_bench + analyze_results.py --export (tukey-upper)"},
              open(sys.argv[2], "w"), indent=1)
    print(f"{len(metrics)} metrics")
    return 0 if metrics else 1


if __name__ == "__main__":
    sys.exit(main())
