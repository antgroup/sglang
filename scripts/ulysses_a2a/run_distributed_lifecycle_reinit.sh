#!/usr/bin/env bash
set -euo pipefail

backend="${1:-nccl}"
topology="${2:-tp2_sp2}"
timeout_seconds="${TIMEOUT_SECONDS:-240}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"
export PYTHONPATH="${repo_root}/python${PYTHONPATH:+:${PYTHONPATH}}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

timeout "${timeout_seconds}s" torchrun \
  --standalone \
  --nproc-per-node=4 \
  scripts/ulysses_a2a/distributed_lifecycle_reinit.py \
  --backend "${backend}" \
  --topology "${topology}"
