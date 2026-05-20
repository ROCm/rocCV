---
name: new-operator
description: Scaffold a new rocCV operator end-to-end — public header, dispatch impl, device + host kernels, pybind11 binding, C++ test, Python test — plus auto-register it in roccv_operators.hpp and python/src/main.cpp. Use when adding a new image operator (e.g. "add a Sobel operator", "scaffold op_blur", "/new-operator my_op").
---

# new-operator — scaffold a new rocCV operator

You generate the 8 files and 2 registration edits needed for a new rocCV operator. CMake uses `GLOB_RECURSE`, so no CMakeLists edits are required.

This skill produces **scaffolding only**. Kernel bodies, golden models, and any operator-specific semantics are left as `// TODO:` markers for the user to fill in. Your job is to wire the structure correctly: signatures, validation, dispatch, registration, and a test harness that matches the spec the user gives you.

## Step 1 — collect the full spec

Before writing any file, you must have an explicit specification of what the operator accepts and rejects. Compile correctness depends on this — the validation block, the dispatch table, and the test cases all derive from the spec. **Do not guess defaults silently.** If the user has not provided a value, ask.

Collect (in order):

1. **Operator name (snake_case)** — `[a-z][a-z0-9_]*`, e.g. `sobel`, `warp_affine`. Used for filenames and the python function. Derive `OP_PASCAL` (e.g. `WarpAffine`) and ask if `OP_PY_FUNC` should differ from the snake form (e.g. some existing ops use `cvtcolor` not `cvt_color`).

2. **Extra parameters** beyond `input` + `output`. For each: C++ type, C++ name (camelCase), Python name (snake_case), one-sentence doc, optional default. Accept "none" to scaffold the no-extras case.

3. **Supported input data types.** Multi-select from: `U8, U16, U32, S8, S16, S32, F32, F64`. Default suggestion (only if the user explicitly accepts it): `U8, S32, F32`.

4. **Supported input tensor layouts.** Multi-select from: `NHWC, HWC, NCHW, CHW, NC, HW`, etc. Default suggestion: `NHWC, HWC`.

5. **Supported channel counts.** Multi-select from `{1, 2, 3, 4}`. Default suggestion: `1, 3, 4`. The dispatch table is a fixed 4-element `std::array` indexed by `channels - 1`, and HIP vector types only exist for widths 1–4 (e.g. `uchar1`–`uchar4`, `float1`–`float4`) — so do not accept channel counts outside this range. (Channel count drives both the validation and the dispatch table's per-row entries.)

6. **Output relationship to input** — for each of the following, ask whether output must match input:
   - layout (default: yes)
   - dtype (default: yes)
   - shape (default: yes — set to "no" for ops like `Resize`)
   - If shape differs, ask which dimensions can differ (width/height/channels/batch).

7. **Any extra preconditions** specific to this operator — e.g. "kernel size must be odd", "scale must be positive", "input and output must reside on the same device". If the user lists any, scaffold them as `CHECK_TENSOR_COMPARISON(...)` or plain `if (...) throw Exception(...)` lines.

Echo the collected spec back as a compact bullet list before generating files, and proceed only after confirmation. If the user replies with corrections, update and re-echo.

## Step 2 — generate the 8 files

Read each template from `templates/` and write the destination file with substitutions applied. Templates use these placeholders:

### Naming
- `{{OP_SNAKE}}` — snake_case operator name
- `{{OP_PASCAL}}` — PascalCase class name
- `{{OP_PY_FUNC}}` — python-facing function name

### Parameter wiring (comma-prefixed; empty string when no extras)
- `{{PARAMS_CPP_DECL}}` — C++ param list for declarations, e.g. `, int32_t flipCode, float scale`
- `{{PARAMS_CPP_FWD}}` — C++ forwarding list, e.g. `, flipCode, scale`
- `{{PARAMS_DOXYGEN}}` — doxygen `@param[in]` lines (one per param)
- `{{PARAMS_PY_DECL}}` — same as `PARAMS_CPP_DECL` for PyOp signatures (usually identical)
- `{{PARAMS_PY_DEF_ARGS}}` — pybind11 `.def()` args, e.g. `, "flip_code"_a, "scale"_a = 1.0f`
- `{{PARAMS_PY_DOCSTRING}}` — `Args:` lines, one per param
- `{{PARAMS_PYTEST_PARAMETRIZE}}` — `@pytest.mark.parametrize` decorators
- `{{PARAMS_PYTEST_NAMES}}` — comma-prefixed names appended to the test fn signature
- `{{PARAMS_PYTEST_FORWARD}}` — comma-prefixed names forwarded into the rocpycv call

### Spec-driven (derived from Step 1)
- `{{DTYPES_DOXYGEN}}` — short form for the doxygen Limitations block, e.g. `U8, S32, F32`
- `{{LAYOUTS_DOXYGEN}}` — e.g. `TENSOR_LAYOUT_NHWC, TENSOR_LAYOUT_HWC`
- `{{CHANNELS_DOXYGEN}}` — e.g. `1, 3, 4`
- `{{IO_DEPENDENCY_TABLE}}` — rendered rows of the Input/Output dependency table inside the doxygen block. One row per property (TensorLayout, DataType, Channels, Width, Height, Batch) with `Yes` or `No`.
- `{{VALIDATION_DTYPES}}` — comma-separated `DATA_TYPE_*` enum values for `CHECK_TENSOR_DATATYPES`
- `{{VALIDATION_LAYOUTS}}` — comma-separated `TENSOR_LAYOUT_*` enum values for `CHECK_TENSOR_LAYOUT`
- `{{VALIDATION_CHANNELS}}` — comma-separated integers for `CHECK_TENSOR_CHANNELS`
- `{{OUTPUT_VALIDATION}}` — the block of output-vs-input `CHECK_TENSOR_COMPARISON(...)` lines. Include layout/dtype/shape lines only when the spec says they must match. If shape may differ, comment out the shape check and add a `// TODO:` for the user to encode the per-dimension constraints. Append any extra preconditions from Step 1.7 here.
- `{{DISPATCH_TABLE}}` — the body of the `funcs` `unordered_map`. One row per supported dtype; each row is a **fixed 4-element** `std::array` indexed by `channels - 1`. The skill only supports channel counts in `{1, 2, 3, 4}` — Step 1.5 must reject anything outside that range, since wider channel counts would index out of bounds and have no matching HIP vector type. Use `dispatch_{{OP_SNAKE}}<vector_type>` for supported channel counts and `0` for unsupported. Map dtype → vector type prefix as:
  - `U8` → `uchar`, `U16` → `ushort`, `U32` → `uint`, `S8` → `char`, `S16` → `short`, `S32` → `int`, `F32` → `float`, `F64` → `double`
  - Then suffix with channel count: e.g. F32 + 3ch → `float3`, U8 + 1ch → `uchar1`.
- `{{TEST_CORRECTNESS_GPU}}` and `{{TEST_CORRECTNESS_CPU}}` — `TEST_CASE(TestCorrectness<...>(...));` lines. Generate one per supported dtype × representative channel count so every dispatch row is exercised at least once on both devices. Use `FMT_*` constants matching the dtype + channel combo (e.g. `FMT_RGB8` for uchar3, `FMT_RGBf32` for float3, `FMT_U8` for uchar1, `FMT_S32` for int1, `FMT_F32` for float1, `FMT_RGBA8` for uchar4, `FMT_RGBAf32` for float4).
- `{{PYTEST_DTYPES}}` — comma-separated `rocpycv.eDataType.*` values for the pytest parametrize over dtype
- `{{PYTEST_CHANNELS}}` — comma-separated integers for the pytest parametrize over channels
- `{{PYTEST_LAYOUT}}` — the layout label used to construct the golden tensor (use `NHWC` if NHWC is supported, otherwise the first supported layout)

When a placeholder represents an empty list (no extras), emit an empty string — not a stray comma.

### Destination paths (do NOT create directories — they all exist)

| Template                          | Destination                                                       |
| --------------------------------- | ----------------------------------------------------------------- |
| `op_NAME.hpp.template`            | `include/op_{{OP_SNAKE}}.hpp`                                     |
| `op_NAME.cpp.template`            | `src/op_{{OP_SNAKE}}.cpp`                                         |
| `NAME_device.hpp.template`        | `include/kernels/device/{{OP_SNAKE}}_device.hpp`                  |
| `NAME_host.hpp.template`          | `include/kernels/host/{{OP_SNAKE}}_host.hpp`                      |
| `py_op_NAME.hpp.template`         | `python/include/operators/py_op_{{OP_SNAKE}}.hpp`                 |
| `py_op_NAME.cpp.template`         | `python/src/operators/py_op_{{OP_SNAKE}}.cpp`                     |
| `test_op_NAME.cpp.template`       | `tests/roccv/cpp/src/tests/operators/test_op_{{OP_SNAKE}}.cpp`    |
| `test_op_NAME.py.template`        | `tests/roccv/python/test_op_{{OP_SNAKE}}.py`                      |

Refuse to overwrite if any destination already exists — print which file conflicts and stop.

## Step 3 — wire registration

Two edits, both via the Edit tool:

1. **`include/roccv_operators.hpp`** — append `#include "op_{{OP_SNAKE}}.hpp"` to the include list. Insert in roughly the same casual alphabetical order the existing entries use.

2. **`python/src/main.cpp`** — two inserts:
   - Add `#include "operators/py_op_{{OP_SNAKE}}.hpp"` to the includes block at the top.
   - Add `PyOp{{OP_PASCAL}}::Export(m);` to the registration block inside the PYBIND11 module body.

## Step 4 — report

Output a short summary:
- Spec used (one bullet list, so the user sees what got encoded)
- Files created (8 paths)
- Registration edits applied (2 files)
- TODOs the user must still fill in:
  - Kernel body in `{{OP_SNAKE}}_device.hpp` and `{{OP_SNAKE}}_host.hpp`
  - Golden model in the C++ test
  - Any operator-specific validation not captured in Step 1.7
- Reminder: rebuild with `cmake --build build --parallel`. Do not run benchmarks; the user runs those.

## Notes

- The dispatch table and the validation block must agree exactly. If `CHECK_TENSOR_DATATYPES` allows `F32` but the dispatch table has no `F32` row, you'll throw `Not mapped to a defined function`. Cross-check before writing.
- For ops where shape differs (e.g. Resize), do not emit the `CHECK_TENSOR_COMPARISON(input.shape() == output.shape())` line; emit a `// TODO:` instead so the user wires the per-dimension checks they need.
- The `0` entries in dispatch table rows represent channel counts the operator does not support. Don't replace them with stubs.
- Templates default the kernel block size to `(64, 16)` and the grid to `(width/bx, height/by, batches)` — fine for most pointwise ops. If the operator needs a different launch shape, leave a `// TODO:` in `op_NAME.cpp` near the `dim3` declarations.

## Invocation examples

- `/new-operator` — fully interactive
- `/new-operator sobel` — name given, prompts for spec
- "scaffold an operator called gaussian_blur that takes a float sigma, supports U8 and F32 on NHWC, channels 1 and 3, and produces same-shape output" — extract the spec from prose; only ask for what's missing.
