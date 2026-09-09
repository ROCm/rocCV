# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

rocCV is an AMD GPU-accelerated image pre/post-processing library built on HIP/ROCm. It ships as a C++ shared library (`libroccv`) plus a Python binding (`rocpycv`) via pybind11. Requires C++20 and AMD Clang (ROCm ≥ 7.0, gfx908+).

## Build Commands

```bash
# Standard build (library + Python bindings)
mkdir -p build && cd build
cmake ../
cmake --build . --parallel

# With tests
cmake -DTESTS=ON ../
cmake --build . --parallel

# With everything (tests + samples)
cmake -DFULL_BUILD=ON ../

# With benchmarks
cmake -DBENCHMARKS=ON ../

# Debug build
cmake -DCMAKE_BUILD_TYPE=Debug ../

# Target a specific GPU architecture
cmake -DGPU_TARGETS=gfx1100 ../

# Install
sudo make install
```

## Running Tests

**C++ tests** — after building with `-DTESTS=ON`, from the build directory:
```bash
ctest                          # run all tests
ctest -VV                      # verbose output
ctest -R test_op_resize        # run a single test by name
./bin/tests/operators/test_op_resize   # run executable directly
```

**Python tests** — requires `PYTHONPATH` pointing to the build's lib dir:
```bash
export PYTHONPATH=build/lib:$PYTHONPATH
python3 -m pytest tests/roccv/python/                     # all python tests
python3 -m pytest tests/roccv/python/test_op_resize.py    # single operator
python3 -m pytest tests/roccv/python/test_op_resize.py -v # verbose
# Or via make target from build dir:
make test_python
```

**Benchmarks** — after building with `-DBENCHMARKS=ON`, from build dir:
```bash
./bin/roccv_bench --config ../benchmarks/config.json
./bin/roccv_bench --help
```

## Architecture

### Layer Overview

```
User code
    ↓
include/op_*.hpp          ← Operator public API (IOperator subclasses)
    ↓
src/op_*.cpp              ← Operator implementations — type/layout dispatch → kernel calls
    ↓
include/kernels/device/   ← HIP __global__ kernels (header-only templates)
include/kernels/host/     ← CPU kernel implementations (header-only templates)
    ↓
include/core/wrappers/    ← Kernel accessor types (ImageWrapper, BorderWrapper, InterpolationWrapper)
    ↓
include/core/             ← Tensor, TensorShape, TensorLayout, DataType, ImageFormat
```

### Key Design Patterns

**Operators** are lightweight stateless classes inheriting `IOperator` (`include/i_operator.hpp`). Each is invoked as a functor accepting a `hipStream_t` stream, input tensor, output tensor, and operator-specific parameters. The `eDeviceType` enum selects GPU or CPU path at runtime.

**Tensor system**: `TensorRequirements` (shape + dtype + device) → `TensorStorage` (raw buffer) → `Tensor` (user-facing handle). The `IAllocator` interface allows custom memory management; the default uses `hipMalloc`/`hipFreeHost` etc. Tensors carry a `TensorLayout` (NHWC, NCHW, HWC, etc.) encoded as a string of dimension labels.

**Kernel wrappers** chain together to handle border and interpolation logic cleanly in device code:
- `ImageWrapper<T>` — bounds-unsafe direct accessor for NHWC/NCHW/HWC tensors
- `BorderWrapper<T, BT>` — adds border handling (replicate, constant, reflect, wrap)
- `InterpolationWrapper<T, BT, IT>` — adds interpolation (nearest, linear, cubic) on top of a border wrapper

Operator `.cpp` files do: validate inputs → build typed wrappers → dispatch to `Kernels::Device::*` or `Kernels::Host::*` based on `eDeviceType`.

**GPU kernels** are header-only in `include/kernels/device/` because they contain `__global__` template functions that must be instantiated from the `.cpp` TU that calls them.

### Directory Map

| Path | Purpose |
|------|---------|
| `include/op_*.hpp` | Public operator headers |
| `include/operator_types.h` | Shared enums (eInterpolationType, eBorderType, etc.) and structs |
| `include/i_operator.hpp` | IOperator base class |
| `include/roccv_operators.hpp` | Convenience header including all operators |
| `include/core/` | Tensor, TensorLayout, DataType, ImageFormat, and wrappers |
| `include/core/detail/` | Internal: casting, type_traits, math, allocator interface, context |
| `include/kernels/device/` | HIP GPU kernels (header-only `__global__` templates) |
| `include/kernels/host/` | CPU kernels (header-only templates) |
| `include/kernels/common/` | Coefficients and helpers shared by device/host |
| `include/common/` | `validation_helpers.hpp` used by operator implementations |
| `src/op_*.cpp` | Operator implementations |
| `src/core/` | Core type implementations |
| `python/src/` | pybind11 binding code (builds against the vendored pybind11 / dlpack) |
| `vendor/` | Checked-in third-party sources (pybind11 v3.0.0, dlpack v1.3, nlohmann/json v3.12.0) — no submodules, no network fetch; see `vendor/ORIGIN.md` |
| `tests/roccv/cpp/` | C++ tests — one executable per operator, custom test framework with `EXPECT_TEST_STATUS` macro |
| `tests/roccv/python/` | Python pytest tests — one file per operator |
| `benchmarks/` | `roccv_bench` executable and `config.json` |

### C++ Test Framework

Tests do **not** use Google Test or Catch2. Each test binary has a custom `main()`. Use `EXPECT_TEST_STATUS(call, eTestStatusType)` and `EXPECT_EXCEPTION(call, eStatusType)` macros defined in `tests/roccv/cpp/include/test_helpers.hpp`. Golden-model testing is the standard pattern: run operator on CPU, compare result byte-for-byte against the GPU output.

### Adding a New Operator

Use the `new-operator` skill (`.claude/skills/new-operator/`) to scaffold all of the below from a single spec. The skill prompts for supported dtypes / layouts / channels / output relationship / extra params, then generates files and wires registration. Invoke via `/new-operator [name]` or in prose ("scaffold an operator called X that ...").

The eight files it produces (and the two registration sites it edits):

1. `include/op_<name>.hpp` — operator class inheriting `IOperator`
2. `include/kernels/device/<name>_device.hpp` — `__global__` template kernel(s)
3. `include/kernels/host/<name>_host.hpp` — CPU implementation
4. `src/op_<name>.cpp` — type/layout dispatch wiring kernels to operator
5. `#include "op_<name>.hpp"` added to `include/roccv_operators.hpp`
6. C++ test at `tests/roccv/cpp/src/tests/operators/test_op_<name>.cpp`
7. Python test at `tests/roccv/python/test_op_<name>.py`
8. pybind11 binding in `python/src/operators/` (header in `python/include/operators/`) + `PyOp<Name>::Export(m)` registered in `python/src/main.cpp`

CMake uses `GLOB_RECURSE`, so no CMakeLists edits are required. Kernel bodies and golden models remain as `// TODO:` markers — the skill scaffolds structure, not semantics.
