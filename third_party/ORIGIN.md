# Vendored Dependencies

This directory holds third-party sources checked directly into the rocCV tree.

Every dependency here is header-only, so each is trimmed to its headers, its license file, and — where we consume the dependency through its upstream CMake project — the CMake files that project needs. Nothing else upstream ships is checked in. Every file is unmodified upstream content, taken verbatim from the pinned release; no rocCV-local patches have been applied.

## Updating

Do not hand-edit these trees. To move to a new upstream version:

1. Download the release named below, at the new tag, from its upstream repository.
2. Replace the paths listed under **Vendored** with the new ones.
3. Delete everything else the release ships.
4. Update the version field in this file.

To verify an existing tree, extract the pinned release and `diff -r` it against each **Vendored** path. It should report no differences.

---

## pybind11

| | |
|---|---|
| Release | v3.1.0 |
| Upstream | https://github.com/pybind/pybind11 |
| Vendored | `include/pybind11/`, `tools/`, `CMakeLists.txt`, `LICENSE` |
| License | BSD 3-Clause |

Provides the `pybind11::headers` target and the `pybind11_add_module()` helper used to build the `rocpycv` extension module. `CMakeLists.txt` and `tools/` are kept because that helper is what supplies LTO, section-stripping, hidden visibility, and the SOABI extension suffix; added via `add_subdirectory` with `PYBIND11_INSTALL` and `PYBIND11_TEST` off.

## dlpack

| | |
|---|---|
| Release | v1.3 |
| Upstream | https://github.com/dmlc/dlpack |
| Vendored | `include/dlpack/dlpack.h`, `LICENSE` |
| License | Apache-2.0 |

Defines the DLPack tensor exchange ABI, used by `rocpycv` for zero-copy interop with other frameworks. Upstream's CMake project only wraps this single header in an interface target, so it is consumed as a plain include directory instead.

## nlohmann/json

| | |
|---|---|
| Release | v3.12.0 |
| Upstream | https://github.com/nlohmann/json |
| Vendored | `include/nlohmann/{json.hpp,json_fwd.hpp}`, `LICENSE.MIT` |
| License | MIT |

The single-header release assets, used by the benchmarking suite (`benchmarks/`) to parse `config.json` and serialize results.
