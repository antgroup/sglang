#!/usr/bin/env bash
set -euo pipefail

backend="${1:-sgl_p2p}"
cycles="${2:-20}"
timeout_seconds="${TIMEOUT_SECONDS:-300}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"
export PYTHONPATH="${repo_root}/python${PYTHONPATH:+:${PYTHONPATH}}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

timeout "${timeout_seconds}s" torchrun \
  --standalone \
  --nproc-per-node=4 \
  scripts/ulysses_a2a/distributed_backend_memory_soak.py \
  --backend "${backend}" \
  --cycles "${cycles}"
