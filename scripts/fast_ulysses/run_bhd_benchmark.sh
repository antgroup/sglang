#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "Usage: $0 <ulysses_degree> <N_global_sequence> <H_global_heads> <D_head_dim>" >&2
  echo "Example: $0 8 75600 40 128" >&2
  exit 2
fi

DEGREE="$1"
GLOBAL_SEQ="$2"
HEADS="$3"
HEAD_DIM="$4"
for value_name in DEGREE GLOBAL_SEQ HEADS HEAD_DIM; do
  value="${!value_name}"
  if ! [[ "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "${value_name} must be a positive integer, got ${value}" >&2
    exit 2
  fi
done
if ((DEGREE < 2 || DEGREE > 8)); then
  echo "ulysses_degree must be in [2, 8]" >&2
  exit 2
fi
if ((GLOBAL_SEQ % DEGREE != 0)); then
  echo "N=${GLOBAL_SEQ} must be divisible by ulysses_degree=${DEGREE}" >&2
  exit 2
fi
if ((HEADS % DEGREE != 0)); then
  echo "H=${HEADS} must be divisible by ulysses_degree=${DEGREE}" >&2
  exit 2
fi
if ((HEAD_DIM % 8 != 0)); then
  echo "D=${HEAD_DIM} must be a multiple of 8 for BF16 16-byte row alignment" >&2
  exit 2
fi

SGLANG_DIR="${SGLANG_DIR:-/home/admin/sglang}"
NVSHMEM_HOME="${FAST_ULYSSES_NVSHMEM_HOME:-/home/admin/nvshmem-3.7.2-cu12}"
LOG_DIR="${FAST_ULYSSES_LOG_DIR:-/home/admin/fast_ulysses_experiment/logs}"
WARMUP="${FAST_ULYSSES_BENCH_WARMUP:-5}"
ITERATIONS="${FAST_ULYSSES_BENCH_ITERATIONS:-20}"
STAMP="$(date +%Y%m%d-%H%M%S)"
LOG_FILE="${LOG_DIR}/bhd-ws${DEGREE}-N${GLOBAL_SEQ}-H${HEADS}-D${HEAD_DIM}-${STAMP}.log"

test -d "${SGLANG_DIR}"
test -d "${NVSHMEM_HOME}/lib"
mkdir -p "${LOG_DIR}"

export PYTHONPATH="${SGLANG_DIR}/python${PYTHONPATH:+:${PYTHONPATH}}"
export LD_PRELOAD="/usr/lib64/libcuda.so.1${LD_PRELOAD:+:${LD_PRELOAD}}"
export LD_LIBRARY_PATH="${NVSHMEM_HOME}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export NVSHMEM_DISABLE_NVLS="${NVSHMEM_DISABLE_NVLS:-1}"
export NVSHMEM_REMOTE_TRANSPORT="${NVSHMEM_REMOTE_TRANSPORT:-none}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"

cd "${SGLANG_DIR}"
torchrun --standalone --nproc-per-node="${DEGREE}" \
  scripts/fast_ulysses/benchmark_bhd.py \
  --global-seq "${GLOBAL_SEQ}" \
  --heads "${HEADS}" \
  --head-dim "${HEAD_DIM}" \
  --warmup "${WARMUP}" \
  --iterations "${ITERATIONS}" \
  2>&1 | tee "${LOG_FILE}"

echo "FAST_ULYSSES_BHD_LOG=${LOG_FILE}"
