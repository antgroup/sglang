#!/usr/bin/env bash
set -euo pipefail

SGLANG_DIR="${SGLANG_DIR:-/home/admin/sglang}"
NVSHMEM_HOME="${FAST_ULYSSES_NVSHMEM_HOME:-/home/admin/nvshmem-3.7.2-cu12}"
LOG_DIR="${FAST_ULYSSES_LOG_DIR:-/home/admin/fast_ulysses_experiment/logs}"
TRANSFER="${FAST_ULYSSES_TRANSFER:-auto}"
BENCH_ARGS=("$@")
for ((i = 0; i < ${#BENCH_ARGS[@]}; i++)); do
  case "${BENCH_ARGS[i]}" in
    --transfer)
      TRANSFER="${BENCH_ARGS[i + 1]}"
      ;;
    --transfer=*)
      TRANSFER="${BENCH_ARGS[i]#--transfer=}"
      ;;
  esac
done
STAMP="$(date +%Y%m%d-%H%M%S)"
LOG_FILE="${LOG_DIR}/microbenchmark-${TRANSFER}-${STAMP}.log"

mkdir -p "${LOG_DIR}"
export PYTHONPATH="${SGLANG_DIR}/python${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="${NVSHMEM_HOME}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export LD_PRELOAD="/usr/lib64/libcuda.so.1${LD_PRELOAD:+:${LD_PRELOAD}}"
export NVSHMEM_DISABLE_NVLS="${NVSHMEM_DISABLE_NVLS:-1}"
export NVSHMEM_REMOTE_TRANSPORT="${NVSHMEM_REMOTE_TRANSPORT:-none}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export SGLANG_FAST_ULYSSES_TAG_POOL_SIZE="${SGLANG_FAST_ULYSSES_TAG_POOL_SIZE:-32}"

cd "${SGLANG_DIR}"
torchrun --standalone --nproc-per-node=8 \
  scripts/fast_ulysses/benchmark_a2a.py \
  "${BENCH_ARGS[@]}" \
  --transfer "${TRANSFER}" 2>&1 | tee "${LOG_FILE}"

echo "FAST_ULYSSES_MICROBENCHMARK_LOG=${LOG_FILE}"
