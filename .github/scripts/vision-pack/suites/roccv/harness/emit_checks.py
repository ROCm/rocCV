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

"""Turn "@@VPCHECK<TAB>group<TAB>name<TAB>status<TAB>message" lines (C++ probes) into result records.

    emit_checks.py <log> [--expect group::name ...]

Expected checks that never printed a line (the probe crashed first) are recorded as error.
"""
from __future__ import annotations

import sys

from common import record, summary


def main():
    log, expect = sys.argv[1], []
    if "--expect" in sys.argv:
        expect = sys.argv[sys.argv.index("--expect") + 1:]
    seen = set()
    for line in open(log, encoding="utf-8", errors="replace"):
        if not line.startswith("@@VPCHECK\t"):
            continue
        parts = line.rstrip("\n").split("\t")
        _, group, name, status = parts[:4]
        msg = parts[4] if len(parts) > 4 else ""
        if status not in ("pass", "fail", "error", "skip"):
            status, msg = "error", f"bad status {status!r}: {msg}"
        record(group, name, status, msg, backend="GPU" if name.endswith("GPU") else "CPU" if name.endswith("CPU") else "")
        seen.add(f"{group}::{name}")
    for e in expect:
        if e not in seen:
            group, name = e.split("::", 1)
            record(group, name, "error", f"the probe ended before reporting this check (see {log.rsplit('/', 1)[-1]})")
    summary()


if __name__ == "__main__":
    main()
