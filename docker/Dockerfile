# ##############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# ##############################################################################

# Dockerfile to build and test rocCV on AMD GPU (ROCm)
# Build from repo root. Tag format: roccv:<IMAGE_TAG_SUFFIX>-<date>
#
# Docker cannot derive one ARG from another. To have IMAGE_TAG_SUFFIX follow
# BASE_DOCKER_IMAGE, use docker/build.sh (it sets IMAGE_TAG_SUFFIX from
# BASE_DOCKER_IMAGE) or pass both when calling docker build:
#   SUFFIX=$(echo "${BASE_DOCKER_IMAGE}" | sed 's|^[^/]*/||' | tr ':' '-')
#   docker build -t roccv:${SUFFIX}-$(date +%Y%m%d) --build-arg BASE_DOCKER_IMAGE=... --build-arg IMAGE_TAG_SUFFIX=${SUFFIX} -f docker/Dockerfile .

ARG BASE_DOCKER_IMAGE=rocm/dev-ubuntu-24.04
ARG IMAGE_TAG_SUFFIX=dev-ubuntu-24.04
ARG GPU_ARCH
ARG BUILD_DATE

FROM ${BASE_DOCKER_IMAGE}

LABEL org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.base="${BASE_DOCKER_IMAGE}" \
      roccv.image.tag.suffix="${IMAGE_TAG_SUFFIX}"

ENV ROCM_PATH=/opt/rocm
ENV PATH="${ROCM_PATH}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${ROCM_PATH}/lib:${LD_LIBRARY_PATH}"
ENV HIP_PLATFORM=amd

# Build deps and test deps (OpenCV for samples/tests, pytest, numpy)
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    git \
    libopencv-dev \
    python3-dev \
    python3-pip \
    python3-pytest \
    python3-numpy \
    python3-opencv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Copy repo from build context (build from repo root: docker build -f docker/Dockerfile .)
COPY . roccv/

# Initialize submodules (pybind11, dlpack)
RUN cd roccv && git submodule update --init --recursive

WORKDIR /workspace/roccv

# Configure and build rocCV with tests and Python bindings.
# GPU_ARCH: when set, build only for that GPU (e.g. gfx90a); otherwise use default targets.
ARG GPU_ARCH
RUN mkdir -p build && cd build \
    && cmake .. \
        -D CMAKE_BUILD_TYPE=Release \
        -D BUILD_PYPACKAGE=ON \
        -D TESTS=ON \
        -D SAMPLES=ON \
        -D BENCHMARKS=ON \
        $( [ -n "${GPU_ARCH}" ] && echo "-D GPU_TARGETS=${GPU_ARCH}" ) \
    && cmake --build . --parallel $(nproc)

# Install into image (optional; tests can run from build dir)
RUN cd build && cmake --install . --prefix /opt/rocm

# For runtime: ensure Python can find rocpycv
ENV PYTHONPATH="${ROCM_PATH}/lib:${PYTHONPATH}"

# Default: run C++ and Python tests
WORKDIR /workspace/roccv/build