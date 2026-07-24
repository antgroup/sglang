#!/usr/bin/env bash
set -euo pipefail

FAST_ULYSSES_COMMIT="6e5dcb24dc44e781ac3091d1d9b3f9fef314fb87"
FAST_ULYSSES_REPOSITORY="https://github.com/triple-mu/fast-ulysses.git"
SOURCE_DIR="${FAST_ULYSSES_SOURCE_DIR:-/home/admin/fast-ulysses}"
NVSHMEM_HOME="${FAST_ULYSSES_NVSHMEM_HOME:-/home/admin/nvshmem-3.7.2-cu12}"
CUDA_ARCH="${FAST_ULYSSES_CUDA_ARCH:-90}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PATCH_FILE="${SCRIPT_DIR}/patches/0001-stream-ordered-pre-write-barrier.patch"

for package in \
  libnvshmem3-cuda-12-3.7.2-1 \
  libnvshmem3-devel-cuda-12-3.7.2-1 \
  libnvshmem3-static-cuda-12-3.7.2-1; do
  if ! rpm -q "${package}" >/dev/null 2>&1; then
    echo "Missing ${package}." >&2
    echo "Install with: dnf install -y libnvshmem3-cuda-12-3.7.2-1 libnvshmem3-devel-cuda-12-3.7.2-1 libnvshmem3-static-cuda-12-3.7.2-1" >&2
    exit 1
  fi
done

mkdir -p "${NVSHMEM_HOME}"
if [[ ! -e "${NVSHMEM_HOME}/include" ]]; then
  ln -s /usr/include/nvshmem_12 "${NVSHMEM_HOME}/include"
fi
if [[ ! -e "${NVSHMEM_HOME}/lib" ]]; then
  ln -s /usr/lib64/nvshmem/12 "${NVSHMEM_HOME}/lib"
fi
test -f "${NVSHMEM_HOME}/include/nvshmem.h"
test -f "${NVSHMEM_HOME}/lib/cmake/nvshmem/NVSHMEMVersion.cmake"

if [[ ! -d "${SOURCE_DIR}/.git" ]]; then
  if [[ -e "${SOURCE_DIR}" ]]; then
    echo "${SOURCE_DIR} exists but is not a git checkout; refusing to overwrite it." >&2
    exit 1
  fi
  git clone "${FAST_ULYSSES_REPOSITORY}" "${SOURCE_DIR}"
fi

actual_remote="$(git -C "${SOURCE_DIR}" remote get-url origin)"
if [[ "${actual_remote}" != "${FAST_ULYSSES_REPOSITORY}" ]]; then
  echo "Unexpected fast-ulysses origin: ${actual_remote}" >&2
  exit 1
fi

git -C "${SOURCE_DIR}" fetch origin "${FAST_ULYSSES_COMMIT}"
git -C "${SOURCE_DIR}" checkout --detach "${FAST_ULYSSES_COMMIT}"

if git -C "${SOURCE_DIR}" apply --unidiff-zero --reverse --check "${PATCH_FILE}" >/dev/null 2>&1; then
  echo "Stream-ordered pre-write barrier patch is already applied."
elif git -C "${SOURCE_DIR}" apply --unidiff-zero --check "${PATCH_FILE}"; then
  git -C "${SOURCE_DIR}" apply --unidiff-zero "${PATCH_FILE}"
else
  echo "fast-ulysses checkout has unexpected changes; refusing to alter it." >&2
  git -C "${SOURCE_DIR}" status --short >&2
  exit 1
fi

NVSHMEM_HOME="${NVSHMEM_HOME}" \
FAST_ULYSSES_CUDA_ARCH="${CUDA_ARCH}" \
CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-8}" \
python -m pip install -e "${SOURCE_DIR}" --no-build-isolation

export LD_LIBRARY_PATH="${NVSHMEM_HOME}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export LD_PRELOAD="/usr/lib64/libcuda.so.1${LD_PRELOAD:+:${LD_PRELOAD}}"
python - <<'PY'
import fast_ulysses
from fast_ulysses import UlyssesGroup

assert hasattr(UlyssesGroup, "pre_write_barrier")
print("fast_ulysses_version=", fast_ulysses.__version__)
print("fast_ulysses_module=", fast_ulysses.__file__)
print("pre_write_barrier=available")
PY
