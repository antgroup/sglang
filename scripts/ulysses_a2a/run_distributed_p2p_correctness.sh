#!/usr/bin/env bash
set -euo pipefail

world_size="${1:-2}"
kernel_style="${2:-tk}"
timeout_seconds="${TIMEOUT_SECONDS:-180}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"
export PYTHONPATH="${repo_root}/python${PYTHONPATH:+:${PYTHONPATH}}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

timeout "${timeout_seconds}s" torchrun \
  --standalone \
  --nproc-per-node="${world_size}" \
  scripts/ulysses_a2a/distributed_p2p_correctness.py \
  --kernel-style "${kernel_style}"
