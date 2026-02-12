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

#!/usr/bin/env bash
# Build rocCV Docker image. IMAGE_TAG_SUFFIX is derived from BASE_DOCKER_IMAGE
# (strip repo prefix, replace ':' with '-') so you only need to set the base image.
#
# Usage (from repo root):
#   ./docker/build.sh
#   BASE_DOCKER_IMAGE=rocm/dev-ubuntu-22.04:7.1.1-complete ./docker/build.sh
#   ./docker/build.sh --build-arg GPU_ARCH=gfx90a

set -e
BASE_DOCKER_IMAGE=${BASE_DOCKER_IMAGE:-rocm/dev-ubuntu-24.04}
IMAGE_TAG_SUFFIX=$(echo "${BASE_DOCKER_IMAGE}" | sed 's|^[^/]*/||' | tr ':' '-')
DATE=$(date +%Y%m%d)
IMAGE_TAG="roccv:${IMAGE_TAG_SUFFIX}-${DATE}"

cd "$(dirname "$0")/.."
docker build -t "${IMAGE_TAG}" \
  --build-arg BASE_DOCKER_IMAGE="${BASE_DOCKER_IMAGE}" \
  --build-arg IMAGE_TAG_SUFFIX="${IMAGE_TAG_SUFFIX}" \
  --build-arg BUILD_DATE="${DATE}" \
  -f docker/Dockerfile \
  "$@" .

echo "Built ${IMAGE_TAG}"
