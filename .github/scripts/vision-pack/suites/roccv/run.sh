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

# rocCV suite: shipped C++ ctest with per-case results, pybind ctest, pytest over all 25 files, harvested harnesses
# from the manual QA session, samples and performance.
#
#   quick          import smoke, 5 C++ ctests, a 3-file pytest subset
#   standard       + all 40 C++ ctests, per-case results (instrumented build), pybind ctest (20), full pytest, audits
#   comprehensive  + GPU-vs-CPU/numpy harness, API checks, HWC (M20), strided copy (H17), DLPack, isolated negative
#                    tests (H18), C++ probes (H19, M20, M21 control), stub check (M22), samples, perf_ops
#   full           + roccv_bench from the rocCV source at the manifest commit; with VP_EXTENDED=1 also DLPack with
#                    torch and the PyTorch classification sample (otherwise recorded as blocked)
#
# Test IDs: roccv::<group>::<name>; each harness under harness/ documents the groups it records.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VP_REPO="${VP_REPO:-$(cd "${HERE}/../.." && pwd)}"
source "${VP_REPO}/build_tools/lib/vp.sh"
vp_init roccv

SHARE="${ROCM_PATH}/share/roccv"
CPP_SRC="${SHARE}/test/cpp"
PYB_SRC="${SHARE}/test/pybind"
PYT_DIR="${PYB_SRC}/python"
HPY="${HERE}/harness:${PYTHONPATH}"
JOBS="${VP_BUILD_JOBS:-$(nproc)}"

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

# harness <name> <timeout> <script> [args...]: run a Python harness; it writes its own records, and
# roccv::harness::<name> records whether the harness itself completed.
harness() {
  local name="$1" t="$2" script="$3"
  shift 3
  local id="harness::${name}"
  vp_run "${id}" --timeout "${t}" --cwd "${VP_WORK}" --env "PYTHONPATH=${HPY}" \
    --env "VP_HARNESS_LOG=logs/$(vp__slug "${id}").log" --env "VP_ROCCV_CRASH_PROBES=${CRASH_PROBES}" \
    -- "${VP_PY}" "${HERE}/harness/${script}" "$@"
}

# emit_checks <log> <expected group::name...>: records from a C++ probe's @@VPCHECK lines.
emit_checks() {
  local log="$1"
  shift
  PYTHONPATH="${HPY}" VP_HARNESS_LOG="${log#"${VP_OUT}"/}" "${VP_PY}" "${HERE}/harness/emit_checks.py" "${log}" --expect "$@" \
    >"${log}.emit" 2>&1 || vp_result "harness::emit-checks" error "could not parse ${log}"
}

# run_expect <id> <timeout> <expected-output|-> [--backend B] -- cmd...: pass = exit 0 and the output exists.
run_expect() {
  local id="$1" t="$2" want="$3" backend=""
  shift 3
  if [[ "${1:-}" == --backend ]]; then backend="$2"; shift 2; fi
  [[ "${1:-}" == -- ]] && shift
  local log rc t0 t1 dur status msg=""
  log="${VP_OUT}/logs/$(vp__slug "${id}").log"
  printf '### %s\n### cmd: %s\n' "${id}" "$(printf '%q ' "$@")" >"${log}"
  t0="$(date +%s.%N)"
  (cd "${SW}" && exec timeout -k 10 "$(vp__scale_timeout "${t}")" "$@") >>"${log}" 2>&1 </dev/null
  rc=$?
  t1="$(date +%s.%N)"
  dur="$(awk -v a="${t0}" -v b="${t1}" 'BEGIN { printf "%.3f", b - a }')"
  if [[ "${rc}" -eq 124 || "${rc}" -eq 137 ]]; then
    status=error; msg="timeout (rc ${rc})"
  elif [[ "${rc}" -gt 128 ]]; then
    status=error; msg="killed by signal $((rc - 128))"
  elif [[ "${rc}" -ne 0 ]]; then
    status=fail; msg="exit ${rc}"
  elif [[ "${want}" != - && ! -e "${want}" ]]; then
    status=fail; msg="exit 0 but no output ${want##*/}"
  else
    status=pass
  fi
  [[ "${status}" != pass ]] && msg="${msg}; $(tail -c 800 "${log}" | tr '\n' ' ' | tr -s ' ')"
  vp_result "${id}" "${status}" "${msg}" "${dur}" "${log}" "${backend}" 1 "$(printf '%q ' "$@")"
}

# run_badinput <id> -- cmd...: invalid arguments must give a clean non-zero exit (not 0, not a signal).
run_badinput() {
  local id="$1"
  shift
  [[ "${1:-}" == -- ]] && shift
  local log rc status msg
  log="${VP_OUT}/logs/$(vp__slug "${id}").log"
  (cd "${SW}" && exec timeout -k 10 120 "$@") >"${log}" 2>&1 </dev/null
  rc=$?
  if [[ "${rc}" -eq 0 ]]; then
    status=fail; msg="exit 0 on invalid input"
  elif [[ "${rc}" -eq 124 || "${rc}" -eq 137 ]]; then
    status=error; msg="timeout"
  elif [[ "${rc}" -gt 128 ]]; then
    status=error; msg="aborted with signal $((rc - 128)) instead of reporting the error"
  else
    status=pass; msg="exit ${rc}"
  fi
  vp_result "${id}" "${status}" "${msg}; $(tail -c 400 "${log}" | tr '\n' ' ' | tr -s ' ')" 0 "${log}" "" 1 "$(printf '%q ' "$@")"
}

# run_help <id> -- cmd...: -h must exit 0 and print usage.
run_help() {
  local id="$1"
  shift
  [[ "${1:-}" == -- ]] && shift
  local log rc
  log="${VP_OUT}/logs/$(vp__slug "${id}").log"
  (cd "${SW}" && exec timeout -k 10 60 "$@") >"${log}" 2>&1 </dev/null
  rc=$?
  if [[ "${rc}" -eq 0 ]] && grep -qiE 'usage|options' "${log}"; then
    vp_result "${id}" pass "exit 0 with usage text" 0 "${log}"
  else
    vp_result "${id}" fail "exit ${rc}; $(head -c 300 "${log}" | tr '\n' ' ')" 0 "${log}"
  fi
}

block_all() { # <why> <id...>
  local why="$1" id
  shift
  for id in "$@"; do vp_blocked "${id}" "${why}"; done
}

CRASH_PROBES=0
if vp_crash_tests_allowed; then CRASH_PROBES=1; fi

if [[ ! -d "${CPP_SRC}" || ! -d "${PYT_DIR}" ]]; then
  vp_result "setup::installed-tests" error "rocCV test sources not found under ${SHARE}/test (is amdrocm-roccv-test in the prefix?)"
fi

# ---------------------------------------------------------------------------
# quick: import smoke, C++ ctest subset, pytest subset
# ---------------------------------------------------------------------------

vp_run "import::rocpycv" --timeout 120 --backend CPU -- \
  "${VP_PY}" -c 'import numpy, rocpycv; print(rocpycv.__file__); t = rocpycv.Tensor([1, 2, 2, 1], rocpycv.NHWC, rocpycv.eDataType.U8, rocpycv.eDeviceType.CPU); print(t.shape())'
vp_run "import::rocpycv-gpu" --timeout 120 --backend GPU -- \
  "${VP_PY}" -c 'import numpy as np, rocpycv as cv; g = cv.from_dlpack(np.arange(12, dtype=np.uint8).reshape(1, 2, 2, 3), cv.NHWC).copy_to(cv.eDeviceType.GPU); r = np.from_dlpack(cv.flip(g, 1, None, cv.eDeviceType.GPU).copy_to(cv.eDeviceType.CPU)); assert r.tolist() == np.arange(12, dtype=np.uint8).reshape(1, 2, 2, 3)[:, :, ::-1].tolist(), r; print("flip ok")'

CPP_B="${VP_WORK}/cpp-tests"
CPP_BUILT=0
if [[ -d "${CPP_SRC}" ]]; then
  vp_cmake_build "build.cpp-tests" "${CPP_SRC}" "${CPP_B}" -DBACKEND=HIP -DCMAKE_EXE_LINKER_FLAGS=-pthread && CPP_BUILT=1
fi

# A timeout wrapper so pytest (run by vp_pytest through VP_PY) can never hang the suite.
PYT_WRAP="${VP_WORK}/bin/python-timeout"
mkdir -p "${VP_WORK}/bin" "${VP_WORK}/pytest"
printf '#!/bin/sh\nexec timeout -k 30 %s %q "$@"\n' "${VP_ROCCV_PYTEST_TIMEOUT:-2400}" "$(command -v "${VP_PY}")" >"${PYT_WRAP}"
chmod +x "${PYT_WRAP}"
run_pytest() { # <pytest args...>
  (cd "${VP_WORK}/pytest" && VP_PY="${PYT_WRAP}" vp_pytest pytest --rootdir "${PYT_DIR}" "$@")
}

if ! vp_tier_ge standard; then
  if [[ "${CPP_BUILT}" == 1 ]]; then
    vp_ctest ctest "${CPP_B}" -R '^(test_tensor|test_op_flip|test_op_resize|test_op_custom_crop|test_op_warp_affine)$'
  fi
  run_pytest "${PYT_DIR}/test_op_flip.py" "${PYT_DIR}/test_op_resize.py" "${PYT_DIR}/test_op_custom_crop.py"
  vp_finish
  exit 0
fi

# ---------------------------------------------------------------------------
# standard: all C++ ctests + per-case results, pybind ctest, full pytest, audits
# ---------------------------------------------------------------------------

if [[ "${CPP_BUILT}" == 1 ]]; then
  vp_ctest ctest "${CPP_B}"
fi

# Per-case results: the same installed sources, with instr/test_helpers.hpp shadowing the shipped header so
# every executed TEST_CASE prints a marker; falls back to the stock binaries' "Test Failed:" output.
INSTR_B="${VP_WORK}/cpp-tests-instr"
case_mode=instr case_build="${INSTR_B}"
if [[ -d "${CPP_SRC}" ]]; then
  if ! vp_cmake_build "build.cpp-tests-instr" "${CPP_SRC}" "${INSTR_B}" -DBACKEND=HIP -DCMAKE_EXE_LINKER_FLAGS=-pthread \
      "-DCMAKE_CXX_FLAGS=-iquote ${HERE}/instr"; then
    case_mode=stock case_build="${CPP_B}"
  fi
  if [[ "${case_mode}" == instr || "${CPP_BUILT}" == 1 ]]; then
    vp_run "harness::ctest-cases" --timeout 3600 --env "PYTHONPATH=${HPY}" \
      --env "VP_HARNESS_LOG=logs/cases" -- \
      "${VP_PY}" "${HERE}/ctest_cases.py" run --src "${CPP_SRC}" --build "${case_build}" --mode "${case_mode}" \
      --logs "${VP_OUT}/logs/cases" --timeout 900
  fi
fi

PYB_B="${VP_WORK}/pybind-tests"
if [[ -d "${PYB_SRC}" ]]; then
  vp_run "build.pybind-tests::configure" --timeout 300 -- \
    cmake -S "${PYB_SRC}" -B "${PYB_B}" -DROCM_PATH="${ROCM_PATH}" -DPython3_EXECUTABLE="$(command -v "${VP_PY}")" \
    && vp_ctest pybind-ctest "${PYB_B}"
fi

run_pytest "${PYT_DIR}"
harness audit-tests 300 audit_tests.py

if ! vp_tier_ge comprehensive; then
  vp_finish
  exit 0
fi

# ---------------------------------------------------------------------------
# comprehensive: harvested harnesses, C++ probes, samples, perf_ops
# ---------------------------------------------------------------------------

harness gpu-vs-cpu 2400 gpu_vs_cpu.py
harness api-checks 600 api_checks.py
harness hwc-layout 600 hwc_layout.py
harness strided-copy 300 strided_copy.py
harness dlpack 900 dlpack_interop.py
harness negative-isolated 3600 negative_isolated.py
harness stub-check 300 stub_check.py

PROBE_B="${VP_WORK}/cpp-probe"
PROBE_IDS_HWC=()
for dev in CPU GPU; do
  for op in Flip CvtColor GammaContrast CenterCrop CustomCrop BrightnessContrast Composite Threshold Remap; do
    PROBE_IDS_HWC+=("cpp-probe.hwc::${op}.${dev}")
  done
done
PROBE_IDS_WRAP=(cpp-probe.wrap::host_buffer_device cpp-probe.wrap::host_buffer_cpu_op cpp-probe.strided::copyToAsync_CPU
  cpp-probe.strided::copyToAsync_GPU cpp-probe.strided::copyToHostAsync_packed)
if vp_cmake_build "build.cpp-probe" "${HERE}/cpp_probe" "${PROBE_B}"; then
  vp_run "harness::cpp-probe.hwc" --timeout 300 -- "${PROBE_B}/hwc_probe"
  emit_checks "${VP_OUT}/logs/$(vp__slug harness::cpp-probe.hwc).log" "${PROBE_IDS_HWC[@]}"
  vp_run "harness::cpp-probe.wrap-host" --timeout 300 -- "${PROBE_B}/wrap_probe" host
  emit_checks "${VP_OUT}/logs/$(vp__slug harness::cpp-probe.wrap-host).log" "${PROBE_IDS_WRAP[@]}"
  vp_run "harness::cpp-probe.launch" --timeout 300 -- "${PROBE_B}/launch_probe"
  emit_checks "${VP_OUT}/logs/$(vp__slug harness::cpp-probe.launch).log" cpp-probe.launch::flip_launch_status
  if [[ "${CRASH_PROBES}" == 1 ]]; then
    vp_run "harness::cpp-probe.wrap-gpu" --timeout 300 -- "${PROBE_B}/wrap_probe" gpu
    emit_checks "${VP_OUT}/logs/$(vp__slug harness::cpp-probe.wrap-gpu).log" cpp-probe.wrap::caller_owned_gpu_memory
  else
    vp_skip cpp-probe.wrap::caller_owned_gpu_memory "deliberate double-free probe (H19); runs only where crash tests are allowed"
  fi
else
  block_all "the C++ probes did not build (see build.cpp-probe)" "${PROBE_IDS_HWC[@]}" "${PROBE_IDS_WRAP[@]}" \
    cpp-probe.launch::flip_launch_status cpp-probe.wrap::caller_owned_gpu_memory
fi

# --- samples -----------------------------------------------------------------
SAMPLES_B="${VP_WORK}/samples-build"
SD="${VP_WORK}/sample-data"
SO="${VP_WORK}/sample-out"
SW="${VP_WORK}/samples-run"
BIN="${SAMPLES_B}/bin/samples"
mkdir -p "${SO}" "${SW}"

CPP_SAMPLE_IDS=() HELP_IDS=() BAD_IDS=()
declare -A SAMPLE_CMD SAMPLE_OUT
add_sample() { # <label> <output> <args...>
  local label="$1" out="$2"
  shift 2
  CPP_SAMPLE_IDS+=("samples.cpp::${label}")
  SAMPLE_OUT["${label}"]="${out}"
  SAMPLE_CMD["${label}"]="$(printf '%q ' "$@")"
}
add_sample bilateral_filter.GPU "${SO}/bilateral_gpu.png" bilateral_filter -i "${SD}/mug.jpg" -o "${SO}/bilateral_gpu.png"
add_sample bilateral_filter.CPU "${SO}/bilateral_cpu.png" bilateral_filter -i "${SD}/mug.jpg" -o "${SO}/bilateral_cpu.png" -C
add_sample bnd_box.GPU "${SO}/bndbox_gpu.png" bnd_box -i "${SD}/mug.jpg" -o "${SO}/bndbox_gpu.png"
add_sample bnd_box.CPU "${SO}/bndbox_cpu.png" bnd_box -i "${SD}/mug.jpg" -o "${SO}/bndbox_cpu.png" -C
add_sample center_crop.GPU "${SO}/center_crop_gpu.png" center_crop -i "${SD}/mug.jpg" -o "${SO}/center_crop_gpu.png" -c 640,480
add_sample center_crop.CPU "${SO}/center_crop_cpu.png" center_crop -i "${SD}/mug.jpg" -o "${SO}/center_crop_cpu.png" -c 640,480 -C
add_sample center_crop.batch "${SO}/center_crop_batch" center_crop -i "${SD}/batch" -o "${SO}/center_crop_batch" -c 320,200
add_sample composite.GPU "${SO}/composite_gpu.png" composite -b "${SD}/mug.jpg" -f "${SD}/mug_flip.jpg" -m "${SD}/mask.png" \
  -o "${SO}/composite_gpu.png"
for m in 0 1 2 3 4; do
  add_sample "copy_make_border.mode${m}" "${SO}/cmb_mode${m}.png" copy_make_border -i "${SD}/mug_small.png" \
    -o "${SO}/cmb_mode${m}.png" -t 20 -l 15 -c 255,0,0,255 -m "${m}"
done
add_sample copy_make_border.batch "${SO}/cmb_batch" copy_make_border -i "${SD}/batch" -o "${SO}/cmb_batch" -t 5 -l 7
add_sample custom_crop.GPU "${SO}/custom_crop_gpu.png" custom_crop -i "${SD}/mug.jpg" -o "${SO}/custom_crop_gpu.png" -c 100,200,640,480
add_sample custom_crop.CPU "${SO}/custom_crop_cpu.png" custom_crop -i "${SD}/mug.jpg" -o "${SO}/custom_crop_cpu.png" \
  -c 100,200,640,480 -C
add_sample gamma_contrast.GPU "${SO}/gamma_gpu.png" gamma_contrast -i "${SD}/mug.jpg" -o "${SO}/gamma_gpu.png" -g 2.2
add_sample gamma_contrast.batch "${SO}/gamma_batch" gamma_contrast -i "${SD}/batch" -o "${SO}/gamma_batch" -g 0.5
add_sample normalize.GPU "${SO}/normalize_gpu.bmp" normalize -i "${SD}/mug.jpg" -o "${SO}/normalize_gpu.bmp"
add_sample normalize.CPU "${SO}/normalize_cpu.bmp" normalize -i "${SD}/mug.jpg" -o "${SO}/normalize_cpu.bmp" -cpu
add_sample normalize.files.GPU "${SO}/normalize_files_gpu.bmp" normalize -i "${SD}/mug.jpg" -o "${SO}/normalize_files_gpu.bmp" \
  -base_file "${SD}/base.txt" -scale_file "${SD}/scale.txt" -global_scale 64 -global_shift 128
add_sample normalize.stddev.CPU "${SO}/normalize_stddev_cpu.bmp" normalize -i "${SD}/mug.jpg" \
  -o "${SO}/normalize_stddev_cpu.bmp" -base_file "${SD}/base.txt" -scale_file "${SD}/scale.txt" -stddev_scale 1 \
  -global_scale 64 -global_shift 128 -cpu
for interp in 0 1 2; do
  for b in 0 1 2 3 4; do
    add_sample "warp_perspective.I${interp}_b${b}" "${SO}/warp_persp_I${interp}_b${b}.png" warp_perspective \
      -i "${SD}/mug.jpg" -o "${SO}/warp_persp_I${interp}_b${b}.png" -I "${interp}" -b "${b}"
  done
done
add_sample cropandresize.GPU "${SO}/cropresize_gpu.png" roccv_cropandresize_app -i "${SD}/mug.jpg" -o "${SO}/cropresize_gpu.png"
add_sample cropandresize.CPU "${SO}/cropresize_cpu.png" roccv_cropandresize_app -i "${SD}/mug.jpg" -o "${SO}/cropresize_cpu.png" -C
add_sample cropandresize.nearest "${SO}/cropresize_nearest.png" roccv_cropandresize_app -i "${SD}/mug.jpg" \
  -o "${SO}/cropresize_nearest.png" -I 0 -r 1280,720 -c 0,0,1920,1080
add_sample cropandresize.batch "${SO}/cropresize_batch" roccv_cropandresize_app -i "${SD}/batch" -o "${SO}/cropresize_batch" \
  -c 10,10,300,200 -r 150,100
SAMPLE_BINS=(bilateral_filter bnd_box center_crop composite copy_make_border custom_crop gamma_contrast normalize
  warp_perspective roccv_cropandresize_app)
for s in "${SAMPLE_BINS[@]}"; do HELP_IDS+=("samples.help::${s}"); done
BAD_IDS=(samples.badinput::center_crop_missing_input samples.badinput::center_crop_crop_too_big
  samples.badinput::custom_crop_out_of_bounds samples.badinput::normalize_missing_input
  samples.badinput::normalize_unknown_option)
COMPARE_IDS=()
for n in bilateral_filter bnd_box center_crop custom_crop normalize cropandresize; do COMPARE_IDS+=("samples.compare::${n}_gpu_vs_cpu"); done
for m in 0 1 2 3 4; do COMPARE_IDS+=("samples.compare::copy_make_border_mode${m}"); done
COMPARE_IDS+=(samples.compare::center_crop_vs_input samples.compare::custom_crop_vs_input samples.compare::normalize_output_valid)
for n in center_crop copy_make_border gamma_contrast cropandresize; do COMPARE_IDS+=("samples.compare::${n}_batch_outputs"); done
PY_SAMPLE_IDS=(samples.py::cropandresize samples.py::multi_op_1)

samples_ok=1
if ! vp_require_py "samples.data::generate" PIL; then
  samples_ok=0
elif ! vp_run "samples.data::generate" --timeout 300 --env "PYTHONPATH=${HPY}" -- "${VP_PY}" "${HERE}/harness/samples_data.py" "${SD}"; then
  samples_ok=0
fi
opencv_cpp=0
if [[ "${samples_ok}" == 1 ]] && vp_run "build.samples::configure" --timeout 600 -- \
    cmake -S "${SHARE}/samples" -B "${SAMPLES_B}" -G Ninja -DCMAKE_BUILD_TYPE=Release -DROCM_PATH="${ROCM_PATH}" \
    "-DCMAKE_PREFIX_PATH=${ROCM_PATH};${ROCM_PATH}/lib/cmake;${ROCM_PATH}/lib/llvm"; then
  if grep -q "require OpenCV, but it was not found" "${VP_OUT}/logs/$(vp__slug build.samples::configure).log"; then
    vp_blocked "build.samples::opencv" "libopencv-dev not found: the C++ samples are not built"
  elif vp_run "build.samples::build" --timeout 1800 -- cmake --build "${SAMPLES_B}" --parallel "${JOBS}"; then
    opencv_cpp=1
  fi
fi
if [[ "${opencv_cpp}" == 1 ]]; then
  args=()
  for id in "${CPP_SAMPLE_IDS[@]}"; do
    label="${id#samples.cpp::}"
    eval "args=(${SAMPLE_CMD[${label}]})"
    backend=GPU
    [[ " ${args[*]} " == *" -C "* || " ${args[*]} " == *" -cpu "* ]] && backend=CPU
    run_expect "${id}" 300 "${SAMPLE_OUT[${label}]}" --backend "${backend}" -- "${BIN}/${args[0]}" "${args[@]:1}"
  done
  for s in "${SAMPLE_BINS[@]}"; do run_help "samples.help::${s}" -- "${BIN}/${s}" -h; done
  if [[ "${CRASH_PROBES}" == 1 ]]; then
    run_badinput samples.badinput::center_crop_missing_input -- "${BIN}/center_crop" -i "${SD}/does_not_exist.jpg" -o "${SO}/neg1.png"
    run_badinput samples.badinput::center_crop_crop_too_big -- "${BIN}/center_crop" -i "${SD}/mug_small.png" -o "${SO}/neg2.png" \
      -c 5000,5000
    run_badinput samples.badinput::custom_crop_out_of_bounds -- "${BIN}/custom_crop" -i "${SD}/mug_small.png" -o "${SO}/neg3.png" \
      -c 600,300,100,100
  else
    for id in "${BAD_IDS[@]:0:3}"; do
      vp_skip "${id}" "bad input makes the samples abort (SIGABRT, core dump); runs only where crash tests are allowed"
    done
  fi
  # normalize parses its own options: a missing input exits 1 cleanly, an unknown option is silently ignored.
  run_badinput samples.badinput::normalize_missing_input -- "${BIN}/normalize" -i "${SD}/does_not_exist.jpg" -o "${SO}/neg4.bmp"
  run_badinput samples.badinput::normalize_unknown_option -- "${BIN}/normalize" -bogus_option 3 -i "${SD}/mug_small.png" \
    -o "${SO}/neg5.bmp"
  harness samples-check 600 samples_check.py "${SD}" "${SO}"
else
  block_all "C++ samples unavailable (needs libopencv-dev and Pillow; see build.samples)" \
    "${CPP_SAMPLE_IDS[@]}" "${HELP_IDS[@]}" "${BAD_IDS[@]}" "${COMPARE_IDS[@]}"
fi
if [[ "${samples_ok}" == 1 ]] && "${VP_PY}" -c 'import cv2' >/dev/null 2>&1; then
  mkdir -p "${SO}/py_cropandresize" "${SO}/py_multi_op_1"
  run_expect samples.py::cropandresize 300 "${SO}/py_cropandresize/output_2.png" --backend GPU -- \
    "${VP_PY}" "${SHARE}/samples/cropandresize/python/cropandresize.py" --input-dir "${SD}/batch_hd" --output-dir "${SO}/py_cropandresize"
  run_expect samples.py::multi_op_1 300 "${SO}/py_multi_op_1/output_2.png" --backend GPU -- \
    "${VP_PY}" "${SHARE}/samples/pipeline/multi_op_1.py" --input_dir "${SD}/batch" --output_dir "${SO}/py_multi_op_1"
else
  block_all "python3-opencv (cv2) or the sample inputs are not available" "${PY_SAMPLE_IDS[@]}"
fi

# ---------------------------------------------------------------------------
# full: extended-image checks (torch) before the perf runs
# ---------------------------------------------------------------------------
TORCH_IDS=(dlpack.torch::torch_cpu_to_roccv dlpack.torch::roccv_cpu_to_torch dlpack.torch::torch_gpu_to_roccv
  dlpack.torch::roccv_gpu_to_torch dlpack.torch::roccv_op_on_torch_gpu_tensor dlpack.torch::torch_gpu_noncontig_to_roccv)
if vp_tier_ge full; then
  if [[ "${VP_EXTENDED}" != 1 ]]; then
    block_all "needs the extended image (ROCm torch); VP_EXTENDED=${VP_EXTENDED}" "${TORCH_IDS[@]}" samples.torch::classification
  else
    if vp_require_py harness::dlpack-torch torch; then
      harness dlpack-torch 900 dlpack_interop.py --torch
    else
      block_all "torch not importable" "${TORCH_IDS[@]}"
    fi
    if [[ "${samples_ok}" != 1 ]]; then
      vp_blocked samples.torch::classification "sample inputs could not be generated"
    elif vp_require_py samples.torch::classification torch torchvision cv2; then
      mkdir -p "${VP_CACHE:-${VP_WORK}}/torch-hub"
      vp_run samples.torch::classification --timeout 900 --retry --backend GPU --cwd "${SW}" \
        --env "TORCH_HOME=${VP_CACHE:-${VP_WORK}}/torch-hub" -- \
        "${VP_PY}" "${SHARE}/samples/classification/pytorch_classification.py" --input "${SD}/mug.jpg"
    fi
  fi
fi

# ---------------------------------------------------------------------------
# perf (last): perf_ops nightly; roccv_bench in the full tier
# ---------------------------------------------------------------------------
if harness perf-ops 1800 perf_ops.py "${VP_WORK}/perf_ops.json"; then
  vp_perf perf_ops "${VP_WORK}/perf_ops.json"
fi

if vp_tier_ge full; then
  commit="" url=""
  if [[ -f "${VP_MANIFEST}" ]]; then
    commit="$(jq -r '.submodules[]? | select(.path == "roccv") | .commit // empty' "${VP_MANIFEST}")"
    url="$(jq -r '.submodules[]? | select(.path == "roccv") | .url // empty' "${VP_MANIFEST}")"
  fi
  BENCH_IDS=(bench::fetch-source build.roccv-bench::configure build.roccv-bench::build bench::run bench::analyze bench::to-perf)
  if [[ -z "${commit}" || -z "${url}" ]]; then
    block_all "no roccv entry in the manifest submodules (${VP_MANIFEST})" "${BENCH_IDS[@]}"
  elif [[ -z "${VP_CACHE:-}" ]]; then
    block_all "VP_CACHE is not set (the rocCV source is fetched into the download cache)" "${BENCH_IDS[@]}"
  elif ! vp_require_py bench::analyze pandas; then
    block_all "pandas missing (analyze_results.py)" bench::fetch-source build.roccv-bench::configure \
      build.roccv-bench::build bench::run bench::to-perf
  elif vp_run bench::fetch-source --timeout 3000 -- "${HERE}/fetch_roccv_src.sh" "${commit}" "${url}" "${VP_CACHE}"; then
    src="${VP_CACHE}/roccv-src/${commit}"
    BENCH_B="${VP_WORK}/bench-build"
    cat >"${VP_WORK}/bench-ci.json" <<'JSON'
{
  "params": [
    {"samples": 1, "height": 1080, "width": 1920, "runs": 20, "warmup_runs": 5},
    {"samples": 16, "height": 1080, "width": 1920, "runs": 20, "warmup_runs": 5}
  ]
}
JSON
    if vp_cmake_build build.roccv-bench "${src}/benchmarks" "${BENCH_B}" \
        && vp_run bench::run --timeout 5400 --cwd "${VP_WORK}" -- "${BENCH_B}/bin/roccv_bench" \
          --config "${VP_WORK}/bench-ci.json" --output "${VP_WORK}/roccv_bench.csv" --types "${VP_ROCCV_BENCH_TYPES:-GPU}" \
        && vp_run bench::analyze --timeout 600 --cwd "${VP_WORK}" -- "${VP_PY}" "${src}/benchmarks/analyze_results.py" \
          "${VP_WORK}/roccv_bench.csv" --export "${VP_WORK}/roccv_bench_clean.csv"; then
      if vp_run bench::to-perf --timeout 120 --env "PYTHONPATH=${HPY}" -- "${VP_PY}" "${HERE}/harness/bench_to_perf.py" \
          "${VP_WORK}/roccv_bench_clean.csv" "${VP_WORK}/roccv_bench.json"; then
        vp_perf roccv_bench "${VP_WORK}/roccv_bench.json"
      fi
    fi
  fi
fi

vp_finish
exit 0
