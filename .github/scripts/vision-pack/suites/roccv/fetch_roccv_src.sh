#!/usr/bin/env bash
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

# Fetch the rocCV source at one commit into the shared download cache (used to build roccv_bench, which is
# not installed). Safe for concurrent jobs: the fill runs under a flock and lands with an atomic rename.
#
#   fetch_roccv_src.sh <commit> <url> <cache_dir>
#
# Prints the source directory (<cache_dir>/roccv-src/<commit>) on success.
set -uo pipefail

commit="${1:?commit}"
url="${2:?url}"
cache="${3:?cache dir}"
dest="${cache}/roccv-src/${commit}"
mkdir -p "${cache}/roccv-src" || exit 2

(
  flock -w 1800 9 || { echo "could not lock ${cache}/roccv-src.lock" >&2; exit 3; }
  if [[ -f "${dest}/benchmarks/CMakeLists.txt" ]]; then
    exit 0
  fi
  part="${dest}.part"
  rm -rf "${part}"
  mkdir -p "${part}"
  git -C "${part}" init -q && git -C "${part}" remote add origin "${url}" || exit 4
  ok=0
  for attempt in 1 2 3 4; do
    if timeout -k 30 600 git -C "${part}" -c http.lowSpeedLimit=1000 -c http.lowSpeedTime=60 \
        fetch -q --depth 1 origin "${commit}"; then
      ok=1
      break
    fi
    echo "fetch attempt ${attempt} failed; retrying" >&2
    sleep $((attempt * 15))
  done
  [[ "${ok}" == 1 ]] || { rm -rf "${part}"; exit 5; }
  git -C "${part}" -c advice.detachedHead=false checkout -q FETCH_HEAD || exit 6
  [[ "$(git -C "${part}" rev-parse HEAD)" == "${commit}" ]] || { echo "fetched HEAD is not ${commit}" >&2; exit 7; }
  mv "${part}" "${dest}"
) 9>"${cache}/roccv-src.lock" || exit $?

echo "${dest}"
