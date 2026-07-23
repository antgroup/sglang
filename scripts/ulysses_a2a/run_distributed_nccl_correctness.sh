#!/usr/bin/env bash
set -euo pipefail

world_size="${1:-2}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"
export PYTHONPATH="${repo_root}/python${PYTHONPATH:+:${PYTHONPATH}}"

torchrun --standalone --nproc-per-node "${world_size}" \
  scripts/ulysses_a2a/distributed_nccl_correctness.py
