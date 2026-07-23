#!/usr/bin/env bash
set -euo pipefail

world_size="${1:-4}"
timeout_seconds="${TIMEOUT_SECONDS:-300}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"
export PYTHONPATH="${repo_root}/python${PYTHONPATH:+:${PYTHONPATH}}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

timeout "${timeout_seconds}s" torchrun \
  --standalone \
  --nproc-per-node="${world_size}" \
  scripts/ulysses_a2a/benchmark_lingbot_shapes.py \
  --iterations "${ITERATIONS:-200}" \
  --warmup "${WARMUP:-20}" \
  --trials "${TRIALS:-7}"
