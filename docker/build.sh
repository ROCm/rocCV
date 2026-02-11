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
