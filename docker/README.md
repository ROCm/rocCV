[![MIT licensed](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

# rocCV Docker

Build and run [rocCV](https://github.com/ROCm/rocCV) in a container on AMD GPUs using the ROCm stack.

## Prerequisites

- **Host:** Linux with the ROCm kernel driver (`amdgpu-dkms`) installed. See [ROCm installation](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/).
- **Docker** installed, with permission to run containers with `--device=/dev/kfd` and `--device=/dev/dri`.

## Build

Build from the **repository root** (parent of `docker/`). The Dockerfile copies the current repository from the build context; run `git submodule update --init --recursive` first so submodules are included.

### Using the build script (recommended)

`build.sh` derives the image tag suffix from `BASE_DOCKER_IMAGE` so you only set the base image:

```bash
# Default base (rocm/dev-ubuntu-24.04) → e.g. roccv:dev-ubuntu-24.04-20260211
./docker/build.sh

# Custom base image (suffix derived automatically)
BASE_DOCKER_IMAGE=rocm/dev-ubuntu-22.04:7.1.1-complete ./docker/build.sh

# Pass extra build args to docker build
./docker/build.sh --build-arg GPU_ARCH=gfx90a
```

### Using docker build directly

Tag format: `roccv:<IMAGE_TAG_SUFFIX>-<date>` (date = `$(date +%Y%m%d)`).

```bash
# Default base and tag
docker build -t roccv:dev-ubuntu-24.04-$(date +%Y%m%d) -f docker/Dockerfile .

# Custom base: derive IMAGE_TAG_SUFFIX (strip repo prefix, replace ':' with '-')
BASE_DOCKER_IMAGE=rocm/dev-ubuntu-22.04:7.1.1-complete
SUFFIX=$(echo "${BASE_DOCKER_IMAGE}" | sed 's|^[^/]*/||' | tr ':' '-')
docker build -t roccv:${SUFFIX}-$(date +%Y%m%d) \
  --build-arg BASE_DOCKER_IMAGE=${BASE_DOCKER_IMAGE} \
  --build-arg IMAGE_TAG_SUFFIX=${SUFFIX} \
  -f docker/Dockerfile .
```

### Build arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `BASE_DOCKER_IMAGE` | `rocm/dev-ubuntu-24.04` | Base ROCm image. |
| `IMAGE_TAG_SUFFIX` | `dev-ubuntu-24.04` | Used in the image tag; should match the base (e.g. `dev-ubuntu-22.04-7.1.1`). Use `build.sh` to derive from `BASE_DOCKER_IMAGE`. |
| `GPU_ARCH` | *(all defaults)* | Single GPU target (e.g. `gfx90a`, `gfx908`, `gfx1030`) for a faster build. |
| `BUILD_DATE` | — | Optional; e.g. `$(date +%Y%m%d)` for image labels. |

## Run

Use a tag that matches your build (e.g. `roccv:dev-ubuntu-24.04-20260211` or the output of `build.sh`).

### Interactive shell (with GPU)

```bash
docker run -it --rm \
  --device=/dev/kfd --device=/dev/dri \
  --security-opt seccomp=unconfined \
  --group-add=video \
  roccv:dev-ubuntu-24.04-$(date +%Y%m%d)
```

### Run C++ tests

```bash
docker run --rm \
  --device=/dev/kfd --device=/dev/dri \
  --security-opt seccomp=unconfined \
  roccv:dev-ubuntu-24.04-$(date +%Y%m%d) \
  ctest -R '^test_' --output-on-failure -V
```

### Run Python tests

Python tests use pytest and need GPU access so `rocpycv` can run on the device:

```bash
docker run --rm \
  --device=/dev/kfd --device=/dev/dri \
  --security-opt seccomp=unconfined \
  --group-add=video \
  -w /workspace/roccv/tests/roccv/python \
  roccv:dev-ubuntu-24.04-$(date +%Y%m%d) \
  python3 -m pytest -v
```

Run a single test file:

```bash
docker run --rm \
  --device=/dev/kfd --device=/dev/dri \
  --security-opt seccomp=unconfined \
  -w /workspace/roccv/tests/roccv/python \
  roccv:dev-ubuntu-24.04-$(date +%Y%m%d) \
  python3 -m pytest -v test_op_resize.py
```

`PYTHONPATH` is set in the image so `rocpycv` is found; the working directory is the Python test directory.

### Run benchmarks

Benchmarks need GPU access. Mount a directory to get results on the host:

```bash
mkdir -p results
docker run --rm \
  --device=/dev/kfd --device=/dev/dri \
  --security-opt seccomp=unconfined \
  --group-add=video \
  -v $(pwd)/results:/workspace/roccv/build \
  -w /workspace/roccv/build \
  roccv:dev-ubuntu-24.04-$(date +%Y%m%d) \
  ./bin/roccv_bench --config ../benchmarks/config.json
```

Results are in `results/roccv_bench_results.json`. For more options:

```bash
docker run --rm roccv:dev-ubuntu-24.04-$(date +%Y%m%d) ./bin/roccv_bench --help
```

## Image contents

- rocCV built from the repository in the build context, with tests, samples, benchmarks, and Python bindings.
- Installed under `/opt/rocm`; `PYTHONPATH` includes the rocCV Python module path.
- Working directory in the container: `/workspace/roccv/build` (for `ctest` or `roccv_bench`).

## See also

- [rocCV README](../README.md) — project overview and dependencies.
- [benchmarks/README.md](../benchmarks/README.md) — benchmark configuration and graphing.
