# Vendored Dependencies

This directory holds third-party sources checked directly into the rocCV tree.

These are **not** git submodules. rocCV is consumed by a superbuild that neither permits submodules nor has network access at configure/build time, so the dependencies previously pulled by `FetchContent` in `python/CMakeLists.txt` are vendored here instead.

Every dependency here is header-only, so only its headers and its license file are checked in — no build system, tests, docs, or tooling from upstream. Each is consumed as a plain include directory; none of the upstream CMake projects are added via `add_subdirectory`. The headers themselves are unmodified upstream content, taken verbatim from the pinned release.

## Updating

Do not hand-edit these trees. To move to a new upstream version:

1. Download the release named below, at the new tag, from its upstream repository.
2. Replace the dependency's `include/` tree and license file with the new ones.
3. Delete everything else the release ships.
4. Update the version field in this file.

---

## pybind11

| | |
|---|---|
| Release | v3.0.0 |
| Upstream | https://github.com/pybind/pybind11 |
| Vendored | `include/pybind11/`, `LICENSE` |
| License | BSD 3-Clause |

Provides the C++/Python binding headers used to build the `rocpycv` extension module. The module target is created with CMake's own `Python3_add_library(... MODULE WITH_SOABI)`, which covers what upstream's `pybind11_add_module()` helper did for us, so pybind11's CMake package is not needed.

## dlpack

| | |
|---|---|
| Release | v1.3 |
| Upstream | https://github.com/dmlc/dlpack |
| Vendored | `include/dlpack/dlpack.h`, `LICENSE` |
| License | Apache-2.0 |

Defines the DLPack tensor exchange ABI, used by `rocpycv` for zero-copy interop with other frameworks.

## nlohmann/json

| | |
|---|---|
| Release | v3.12.0 |
| Upstream | https://github.com/nlohmann/json |
| Vendored | `include/nlohmann/{json.hpp,json_fwd.hpp}`, `LICENSE.MIT` |
| License | MIT |

The single-header release assets, used by the benchmarking suite (`benchmarks/`) to parse `config.json` and serialize results.
