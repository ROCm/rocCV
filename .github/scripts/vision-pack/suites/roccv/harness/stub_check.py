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

"""Shipped type stub ($ROCM_PATH/lib/rocpycv.pyi) versus the rocpycv module (M22).

stub::<check>:
  syntax                   the stub parses as Python
  functions_declared       every module function is declared in the stub
  functions_exist          every stub function exists in the module
  classes_declared / classes_exist
  constants_declared / constants_exist
  all_names_exist          every __all__ entry exists in the module
  class_members_exist      stub class members exist on the module classes
  signatures               stub parameter names match the module docstrings
Known (M22): the stub is not valid Python (required parameter after a default in copymakeborder[_into]) and lacks
averageblur, brightness_contrast, gaussian, laplacian and their _into variants.
"""
from __future__ import annotations

import ast
import inspect
import os
import re

import rocpycv
from common import record, summary

STUB = os.path.join(os.environ["ROCM_PATH"], "lib", "rocpycv.pyi")


def split_top(s):
    parts, depth, cur = [], 0, ""
    for ch in s:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    if cur.strip():
        parts.append(cur)
    return parts


def strip_default(p):
    depth = 0
    for i, ch in enumerate(p):
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == "=" and depth == 0:
            return p[:i].rstrip(), True
    return p, False


def repaired_stub(text):
    """Drop default values so the stub parses; return (text, [invalid signature descriptions])."""
    invalid, lines = [], []
    for lineno, line in enumerate(text.splitlines(), 1):
        if line.lstrip().startswith("def ") and "(" in line and ") ->" in line:
            head, rest = line.split("(", 1)
            params_str, tail = rest.rsplit(") ->", 1)
            seen_default, bad, new_params = False, [], []
            for p in split_top(params_str):
                name_part, has_def = strip_default(p)
                pname = name_part.split(":")[0].strip()
                if pname.startswith("*") or pname == "/":
                    seen_default = False if pname == "*" else seen_default
                elif has_def:
                    seen_default = True
                elif seen_default:
                    bad.append(pname)
                new_params.append(name_part)
            if bad:
                invalid.append(f"line {lineno} {head.strip()[4:]}: required {bad} after a default")
            line = head + "(" + ",".join(new_params) + ") ->" + tail
        lines.append(line)
    return "\n".join(lines), invalid


def diff_check(name, missing, what):
    if missing:
        record("stub", name, "fail", f"{len(missing)} {what}: {', '.join(sorted(missing))}")
    else:
        record("stub", name, "pass")


def main():
    if not os.path.exists(STUB):
        record("stub", "syntax", "fail", f"{STUB} is not shipped")
        return
    text = open(STUB, encoding="utf-8").read()
    fixed, invalid = repaired_stub(text)
    try:
        ast.parse(text)
        record("stub", "syntax", "pass")
    except SyntaxError as e:
        record("stub", "syntax", "fail", f"SyntaxError line {e.lineno}: {e.msg}; " + "; ".join(invalid))
    try:
        tree = ast.parse(fixed)
    except SyntaxError as e:
        record("stub", "structure", "error", f"stub unparsable even without defaults: line {e.lineno}: {e.msg}")
        return

    funcs, classes, consts, all_names = set(), {}, set(), []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            funcs.add(node.name)
        elif isinstance(node, ast.ClassDef):
            members = set()
            for sub in node.body:
                if isinstance(sub, ast.FunctionDef):
                    members.add(sub.name)
                elif isinstance(sub, ast.AnnAssign) and isinstance(sub.target, ast.Name):
                    members.add(sub.target.id)
                elif isinstance(sub, ast.Assign):
                    members.update(t.id for t in sub.targets if isinstance(t, ast.Name))
            classes[node.name] = members
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == "__all__":
                all_names = ast.literal_eval(node.value)
            else:
                consts.add(node.target.id)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    if t.id == "__all__":
                        all_names = ast.literal_eval(node.value)
                    else:
                        consts.add(t.id)

    names = {n for n in dir(rocpycv) if not n.startswith("__")}
    mfuncs = {n for n in names if callable(getattr(rocpycv, n)) and not inspect.isclass(getattr(rocpycv, n))}
    mclasses = {n for n in names if inspect.isclass(getattr(rocpycv, n))}
    mconsts = names - mfuncs - mclasses

    diff_check("functions_declared", mfuncs - funcs, "module functions missing from the stub")
    diff_check("functions_exist", funcs - mfuncs, "stub functions missing from the module")
    diff_check("classes_declared", mclasses - set(classes), "module classes missing from the stub")
    diff_check("classes_exist", set(classes) - mclasses, "stub classes missing from the module")
    diff_check("constants_declared", mconsts - consts, "module constants missing from the stub")
    diff_check("constants_exist", consts - mconsts, "stub constants missing from the module")
    diff_check("all_names_exist", set(all_names) - names, "__all__ entries missing from the module")

    missing_members = []
    for cname, members in classes.items():
        if cname in mclasses:
            cls = getattr(rocpycv, cname)
            missing_members += [f"{cname}.{m}" for m in members if not hasattr(cls, m)]
    diff_check("class_members_exist", set(missing_members), "stub class members missing from the module")

    sig_diffs = []
    for fname in sorted(funcs & mfuncs):
        doc = (getattr(rocpycv, fname).__doc__ or "").strip()
        m = re.match(r"^\s*\w+\((.*)\)\s*->", doc.splitlines()[0] if doc else "")
        if not m:
            continue
        mod_params = [p.split(":")[0].strip() for p in split_top(m.group(1))]
        mod_params = [p for p in mod_params if p not in ("*", "/")]
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == fname)
        stub_params = [a.arg for a in node.args.args + node.args.kwonlyargs]
        if mod_params and mod_params != stub_params:
            sig_diffs.append(f"{fname}: stub {stub_params} vs module {mod_params}")
    diff_check("signatures", set(sig_diffs), "signature mismatches")
    summary()


if __name__ == "__main__":
    main()
