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

# Robustness (gpu_access: all), rocCV checks: unsupported-GPU negatives for the
# rocpycv and C++ Flip, with the same probes on the chosen GPU as controls.
# Without an unsupported GPU (VP_UNSUPPORTED_GPUS empty) the suite records only
# unsupported-gpu::none-present.
#
# Device indices come from enumerating GPU agents with rocminfo in this
# environment (ROCR_VISIBLE_DEVICES unset), never from host indices: inside a
# container ROCr numbers only the GPUs whose render nodes were passed in.
# Every probe that uses a GPU pins exactly one with ROCR_VISIBLE_DEVICES, and
# GPU work runs serially.
set -uo pipefail
source "${VP_REPO}/build_tools/lib/vp.sh"
vp_init robustness

if ! vp_tier_ge comprehensive; then
  vp_skip "tier::below-comprehensive" "robustness runs in the comprehensive tier and above"
  vp_finish
  exit 0
fi

CR="${VP_SUITE_DIR}/checked_run.py"
PROBES="${VP_SUITE_DIR}/probes"
export TMPDIR="${VP_WORK}/tmp"
mkdir -p "${TMPDIR}"

fresh_dir() {
  local d="${VP_WORK}/cwd/$1"
  rm -rf "${d}"
  mkdir -p "${d}"
  printf '%s' "${d}"
}

# cr <id> [checked_run options] -- cmd... ; probes report a crashed child with exit 70.
cr() {
  local id="$1"; shift
  "${VP_PY}" "${CR}" --id "${id}" --timeout 300 --error-rc 70 "$@"
}

# ---------------------------------------------------------------------------
# GPU agents in ROCr order, as this process sees them.
# ---------------------------------------------------------------------------
mapfile -t AGENTS < <(env -u ROCR_VISIBLE_DEVICES timeout -k 5 60 rocminfo 2>/dev/null \
  | awk '/^  Name:/ { n = $2 } /^ *Device Type:/ { if ($3 == "GPU") print n }')
vp__log "GPU agents (ROCr order): ${AGENTS[*]:-none}; VP_GFX=${VP_GFX:-none}; VP_UNSUPPORTED_GPUS=${VP_UNSUPPORTED_GPUS:-none}"

# agent_index <gfx> <occurrence>: index of the n-th (0-based) GPU agent with that gfx, or empty.
agent_index() {
  local gfx="$1" want="$2" i seen=0
  for i in "${!AGENTS[@]}"; do
    if [[ "${AGENTS[$i]}" == "${gfx}" ]]; then
      if [[ "${seen}" == "${want}" ]]; then echo "${i}"; return; fi
      seen=$((seen + 1))
    fi
  done
}

CHOSEN_IDX=""
[[ -n "${VP_GFX}" ]] && CHOSEN_IDX="$(agent_index "${VP_GFX}" 0)"

# ---------------------------------------------------------------------------
# Unsupported-GPU negatives (M21), with the same probes on the chosen GPU as controls.
# ---------------------------------------------------------------------------
LAUNCH_PROBE=""
LAUNCH_PROBE_WHY=""
build_launch_probe() {
  [[ -n "${LAUNCH_PROBE}${LAUNCH_PROBE_WHY}" ]] && return
  local cxx="${ROCM_PATH}/lib/llvm/bin/amdclang++" b="${VP_WORK}/roccv-launch-probe"
  if ! command -v cmake >/dev/null 2>&1 || ! command -v ninja >/dev/null 2>&1 || [[ ! -x "${cxx}" ]]; then
    LAUNCH_PROBE_WHY="needs cmake, ninja and ${cxx}"
    return
  fi
  rm -rf "${b}"
  if vp_run "unsupported-gpu.build::roccv-launch-probe.configure" --timeout 300 -- \
       cmake -S "${PROBES}/roccv_launch" -B "${b}" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER="${cxx}" \
       -DROCM_PATH="${ROCM_PATH}" "-DCMAKE_PREFIX_PATH=${ROCM_PATH};${ROCM_PATH}/lib/cmake" \
     && vp_run "unsupported-gpu.build::roccv-launch-probe.build" --timeout 600 -- cmake --build "${b}"; then
    LAUNCH_PROBE="${b}/launch_probe"
  else
    LAUNCH_PROBE_WHY="the rocCV launch probe did not build"
  fi
}

# gpu_probes <group> <agent index> <mode>
gpu_probes() {
  local g="$1" idx="$2" mode="$3" tag="${1//[^A-Za-z0-9]/_}"
  local -a env=(--backend GPU --env "ROCR_VISIBLE_DEVICES=${idx}")
  cr "${g}::roccv-flip-zeros" "${env[@]}" --cwd "$(fresh_dir "${tag}_roccv")" \
    -- "${VP_PY}" "${PROBES}/roccv_flip.py" --device gpu --mode "${mode}"
  build_launch_probe
  if [[ -n "${LAUNCH_PROBE}" ]]; then
    cr "${g}::roccv-cpp-launch-error" "${env[@]}" --cwd "$(fresh_dir "${tag}_roccv_cpp")" -- "${LAUNCH_PROBE}" "${mode}"
  else
    vp_blocked "${g}::roccv-cpp-launch-error" "${LAUNCH_PROBE_WHY}"
  fi
}

if [[ -z "${VP_UNSUPPORTED_GPUS:-}" ]]; then
  vp_skip "unsupported-gpu::none-present" "no unsupported GPU on this runner"
elif [[ ${#AGENTS[@]} -eq 0 ]]; then
  vp_result "unsupported-gpu::agent-enumeration" error "rocminfo lists no GPU agents although VP_UNSUPPORTED_GPUS=${VP_UNSUPPORTED_GPUS}"
else
  if [[ -n "${CHOSEN_IDX}" ]]; then
    gpu_probes "unsupported-gpu.control" "${CHOSEN_IDX}" correct
  else
    vp_result "unsupported-gpu.control::device-visible" error "${VP_GFX:-no chosen GPU} is not among the GPU agents (${AGENTS[*]})"
  fi
  declare -A occurrence=()
  IFS=, read -r -a unsupported <<<"${VP_UNSUPPORTED_GPUS}"
  for u in "${unsupported[@]}"; do
    gfx="${u%%:*}"
    [[ -n "${gfx}" ]] || continue
    n="${occurrence[${gfx}]:-0}"
    occurrence[${gfx}]=$((n + 1))
    group="unsupported-gpu.${gfx}"
    [[ "${n}" -gt 0 ]] && group="${group}.${n}"
    idx="$(agent_index "${gfx}" "${n}")"
    if [[ -z "${idx}" ]]; then
      vp_result "${group}::device-visible" error "${gfx} (${u}) is not among the GPU agents here (${AGENTS[*]})"
      continue
    fi
    vp__log "${group}: ROCR_VISIBLE_DEVICES=${idx}"
    gpu_probes "${group}" "${idx}" honest
  done
fi

vp_finish
exit 0
