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

"""Per-case results for the rocCV C++ tests (custom TEST_CASE harness, not gtest).

    ctest_cases.py sites --src <test/cpp>                         print the TEST_CASE call sites per test binary
    ctest_cases.py run --src <test/cpp> --build <dir> --mode instr|stock --logs <dir> [--timeout S]

The installed sources contain TEST_CASE(...) call sites (1,656 for 0.2.0); some run in loops, so the binaries
execute more cases than there are sites (1,704). Two modes:

  instr  binaries built with instr/test_helpers.hpp print one @@VPCASE marker per executed case, so every
         execution gets its own record; a repeated execution of one site gets a "#<n>" suffix.
  stock  shipped binaries print only "Test Failed:" blocks; every site gets one record (pass unless a block
         names it). Used when the instrumented build is unavailable.

In both modes the static sites are the reference list: a site that never reported (the binary crashed, timed out
or exited early) is recorded as error. Records: case.<binary>::<call text>, e.g.
roccv::case.test_op_flip::TestCorrectness<uchar3>(1,480,360,0,eDeviceType.GPU).
"""
from __future__ import annotations

import argparse
import os
import re
import signal
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "harness"))
from common import backend_of, record, summary  # noqa: E402

MARKER = "@@VPCASE\t"


# ---------------------------------------------------------------------------
# static TEST_CASE sites
# ---------------------------------------------------------------------------

def strip_comments(src: str) -> str:
    """Replace comments with spaces (newlines kept) so offsets and line numbers stay valid."""
    out, i, n = [], 0, len(src)
    while i < n:
        c = src[i]
        if c in "\"'":
            j = i + 1
            while j < n and src[j] != c:
                j += 2 if src[j] == "\\" else 1
            out.append(src[i:j + 1])
            i = j + 1
        elif src.startswith("//", i):
            j = src.find("\n", i)
            j = n if j < 0 else j
            out.append(" " * (j - i))
            i = j
        elif src.startswith("/*", i):
            j = src.find("*/", i + 2)
            j = n if j < 0 else j + 2
            out.append(re.sub(r"[^\n]", " ", src[i:j]))
            i = j
        else:
            out.append(c)
            i += 1
    return "".join(out)


def collapse(text: str) -> str:
    """Whitespace handling of the preprocessor's #call: runs collapse to one space (not inside literals)."""
    out, i, n, pending = [], 0, len(text), False
    while i < n:
        c = text[i]
        if c in "\"'":
            j = i + 1
            while j < n and text[j] != c:
                j += 2 if text[j] == "\\" else 1
            if pending and out:
                out.append(" ")
            pending = False
            out.append(text[i:j + 1])
            i = j + 1
        elif c.isspace():
            pending = True
            i += 1
        else:
            if pending and out:
                out.append(" ")
            pending = False
            out.append(c)
            i += 1
    return "".join(out).strip()


def case_name(call: str) -> str:
    """ID-friendly form of a call: no whitespace around punctuation, '::' -> '.'."""
    s = collapse(call)
    s = re.sub(r"\s*([,(){}\[\]<>=+*/&|-])\s*", r"\1", s)
    s = s.replace("::", ".")
    return re.sub(r"\s+", "_", s)


def find_sites(path: Path) -> list[dict]:
    src = strip_comments(path.read_text(encoding="utf-8", errors="replace"))
    sites = []
    for m in re.finditer(r"(?<![A-Za-z0-9_])TEST_CASE\s*\(", src):
        if src[:m.start()].rstrip().endswith("#define"):
            continue
        i, depth, n = m.end(), 1, len(src)
        while i < n and depth:
            c = src[i]
            if c in "\"'":
                j = i + 1
                while j < n and src[j] != c:
                    j += 2 if src[j] == "\\" else 1
                i = j + 1
                continue
            depth += c == "("
            depth -= c == ")"
            i += 1
        call = src[m.end():i - 1]
        sites.append({"start": src.count("\n", 0, m.start()) + 1, "end": src.count("\n", 0, i) + 1,
                      "text": collapse(call), "name": case_name(call)})
    return sites


def all_sites(src_root: Path) -> dict[str, list[dict]]:
    out = {}
    for p in sorted((src_root / "src" / "tests").rglob("*.cpp")):
        sites = find_sites(p)
        seen: dict[str, int] = defaultdict(int)
        for s in sites:  # distinct sites with identical text: "@<k>" from the second one on
            seen[s["name"]] += 1
            s["id"] = s["name"] if seen[s["name"]] == 1 else f"{s['name']}@{seen[s['name']]}"
        out[p.stem] = sites
    return out


# ---------------------------------------------------------------------------
# running and parsing
# ---------------------------------------------------------------------------

def run_binary(exe: Path, cwd: Path, log: Path, timeout: int) -> tuple[int | None, str, float]:
    t0 = time.perf_counter()
    with open(log, "w", encoding="utf-8", errors="replace") as f:
        f.write(f"### {exe}\n")
        f.flush()
        try:
            p = subprocess.Popen([str(exe)], cwd=cwd, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
            rc = p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            p.wait()
            rc = None
    return rc, log.read_text(encoding="utf-8", errors="replace"), time.perf_counter() - t0


def site_for(sites: list[dict], line: int, text: str) -> dict | None:
    cands = [s for s in sites if s["start"] <= line <= s["end"]]
    if len(cands) == 1:
        return cands[0]
    norm = case_name(text)
    cands = cands or sites
    return next((s for s in cands if s["name"] == norm), cands[0] if len(cands) == 1 else None)


def parse_failed_blocks(out: str) -> list[tuple[str, int, str]]:
    blocks = []
    for m in re.finditer(r"Test Failed: (.*?)\n\s*Line: \[[^\]]*?:(\d+)\][^\n]*\n\s*Reason: ([^\n]*)", out):
        blocks.append((m.group(1), int(m.group(2)), m.group(3)))
    return blocks


def ended(rc: int | None) -> str:
    if rc is None:
        return "timed out"
    if rc < 0:
        return f"killed by signal {-rc}"
    return f"exit {rc}"


def report_binary(name: str, sites: list[dict], rc: int | None, out: str, dur: float, mode: str) -> tuple[int, bool]:
    group = f"case.{name}"
    abnormal = rc is None or rc < 0 or rc > 1
    reported: dict[str, int] = defaultdict(int)
    n = 0
    markers = [ln.split("\t") for ln in out.splitlines() if ln.startswith(MARKER)]
    if mode == "instr" and markers:
        for parts in markers:
            _, status, ms, line, call = (parts + [""] * 6)[:5]
            reason = parts[5] if len(parts) > 5 else ""
            site = site_for(sites, int(line or 0), call)
            base = site["id"] if site else case_name(call)
            reported[base] += 1
            rid = base if reported[base] == 1 else f"{base}#{reported[base]}"
            try:
                secs = float(ms) / 1000.0
            except ValueError:
                secs = 0.0
            record(group, rid, "pass" if status == "PASS" else "fail", reason[:1500], secs, backend_of(call))
            n += 1
    else:
        failed = defaultdict(list)
        for call, line, reason in parse_failed_blocks(out):
            site = site_for(sites, line, call)
            failed[site["id"] if site else case_name(call)].append(reason)
        if rc in (0, 1) and (rc == 0 or failed):
            for s in sites:
                reasons = failed.pop(s["id"], [])
                record(group, s["id"], "fail" if reasons else "pass", " | ".join(reasons)[:1500], 0.0, backend_of(s["text"]))
                reported[s["id"]] += 1
                n += 1
        else:
            for sid, reasons in failed.items():
                record(group, sid, "fail", " | ".join(reasons)[:1500], 0.0, backend_of(sid))
                reported[sid] += 1
                n += 1
    missing = [s for s in sites if s["id"] not in reported]
    for s in missing:
        why = "crashed" if abnormal else "ended"
        record(group, s["id"], "error", f"never reported: the binary {why} ({ended(rc)} after {dur:.1f}s) before this "
               "case ran", 0.0, backend_of(s["text"]))
        n += 1
    fails = sum(1 for ln in out.splitlines() if ln.startswith("Test Failed: "))
    if rc == 1 and not fails and not missing:
        record(group, "exit-status", "error", "binary exited 1 without reporting a failed case")
        n += 1
    return n, bool(markers)


def cmd_sites(a) -> int:
    sites = all_sites(Path(a.src))
    for name, ss in sites.items():
        print(f"{name}\t{len(ss)}")
    print(f"TOTAL\t{sum(len(s) for s in sites.values())}")
    return 0


def cmd_run(a) -> int:
    src, build, logs = Path(a.src), Path(a.build), Path(a.logs)
    logs.mkdir(parents=True, exist_ok=True)
    sites = all_sites(src)
    exes = {p.name: p for p in (build / "bin" / "tests").rglob("*") if p.is_file() and os.access(p, os.X_OK)}
    total_sites = sum(len(s) for s in sites.values())
    total = 0
    any_marker = False
    mode = a.mode
    for name in sorted(sites):
        if a.only and name not in a.only.split(","):
            continue
        if name not in exes:
            for s in sites[name]:
                record(f"case.{name}", s["id"], "error", "test binary was not built", 0.0, backend_of(s["text"]))
            total += len(sites[name])
            continue
        log = logs / f"{name}.log"
        rc, out, dur = run_binary(exes[name], build, log, a.timeout)
        if mode == "instr" and MARKER not in out and rc == 0:
            mode = "stock"
            record("cases", "instrumentation", "error",
                   f"{name} printed no @@VPCASE markers: the per-case overlay was not applied; falling back to stock parsing")
        n, has = report_binary(name, sites[name], rc, out, dur, mode)
        any_marker |= has
        total += n
        print(f"{name}: {ended(rc)} in {dur:.1f}s, {n} cases", flush=True)
    if not a.only:
        record("cases", "static_sites", "pass" if total_sites else "error",
               f"{total_sites} TEST_CASE sites in {len(sites)} test sources; {total} per-case records ({mode} mode)")
    summary()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("sites")
    p.add_argument("--src", required=True)
    p = sub.add_parser("run")
    p.add_argument("--src", required=True)
    p.add_argument("--build", required=True)
    p.add_argument("--logs", required=True)
    p.add_argument("--mode", choices=("instr", "stock"), default="instr")
    p.add_argument("--timeout", type=int, default=900)
    p.add_argument("--only", default="", help="comma-separated test binaries (quick tier)")
    a = ap.parse_args()
    return cmd_sites(a) if a.cmd == "sites" else cmd_run(a)


if __name__ == "__main__":
    sys.exit(main())
