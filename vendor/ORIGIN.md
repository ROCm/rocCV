# Vendored Dependencies

This directory holds third-party sources checked directly into the rocCV tree.

These are **not** git submodules. rocCV is consumed by a superbuild that neither permits submodules nor has network access at configure/build time, so the dependencies previously pulled by `FetchContent` in `python/CMakeLists.txt` are vendored here instead.

Each subdirectory is pristine, unmodified upstream content — either a release tarball with its top-level `<name>-<version>/` prefix stripped, or the release assets themselves. No rocCV-local patches have been applied. If a patch ever becomes necessary, record it in this file.

## Updating

Do not hand-edit these trees. To move to a new upstream version:

1. Download the release tarball from the URL below (with the new tag).
2. Verify its SHA256 against the checksum published by upstream.
3. Replace the subdirectory wholesale with the extracted contents.
4. Update the version, URL, SHA256, and commit fields in this file.

---

## pybind11

| | |
|---|---|
| Version | v3.0.0 |
| Upstream | https://github.com/pybind/pybind11 |
| Source | https://github.com/pybind/pybind11/archive/refs/tags/v3.0.0.tar.gz |
| SHA256 | `453b1a3e2b266c3ae9da872411cadb6d693ac18063bd73226d96cfb7015a200c` |
| Tag commit | `ed5057ded698e305210269dafa57574ecf964483` |
| License | BSD 3-Clause (`pybind11/LICENSE`) |
| Retrieved | 2026-09-09 |

Provides the `pybind11::headers` target and the `pybind11_add_module()` helper used to build the `rocpycv` Python extension module.

## dlpack

| | |
|---|---|
| Version | v1.3 |
| Upstream | https://github.com/dmlc/dlpack |
| Source | https://github.com/dmlc/dlpack/archive/refs/tags/v1.3.tar.gz |
| SHA256 | `f3d567f885f6c142183afc91a58873b31d0e0b36faa2e45c232b98c74596404f` |
| Tag commit | `84d107bf416c6bab9ae68ad285876600d230490d` |
| License | Apache-2.0 (`dlpack/LICENSE`) |
| Retrieved | 2026-09-09 |

Provides the `dlpack::dlpack` interface target defining the DLPack tensor exchange ABI, used by `rocpycv` for zero-copy interop with other frameworks. Built with `BUILD_MOCK=OFF`.

## nlohmann/json

| | |
|---|---|
| Version | v3.12.0 |
| Upstream | https://github.com/nlohmann/json |
| Source | https://github.com/nlohmann/json/releases/download/v3.12.0/json.hpp |
| | https://github.com/nlohmann/json/releases/download/v3.12.0/json_fwd.hpp |
| SHA256 | `aaf127c04cb31c406e5b04a63f1ae89369fccde6d8fa7cdda1ed4f32dfc5de63` (json.hpp) |
| | `fb6aa70cbece087f37ab4685c182b287c53be54f785f981b9db9d30d2d028b37` (json_fwd.hpp) |
| License | MIT (`nlohmann_json/LICENSE.MIT`) |
| Retrieved | 2026-09-09 |

Used by the benchmarking suite (`benchmarks/`) to parse `config.json` and serialize results.
