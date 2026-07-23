#!/usr/bin/env bash
set -euo pipefail

timeout_seconds="${TIMEOUT_SECONDS:-180}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"
export PYTHONPATH="${repo_root}/python${PYTHONPATH:+:${PYTHONPATH}}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

timeout "${timeout_seconds}s" torchrun \
  --standalone \
  --nproc-per-node=2 \
  scripts/ulysses_a2a/distributed_cuda_graph.py
