#!/usr/bin/env bash
set -euo pipefail

timeout_seconds="${TIMEOUT_SECONDS:-90}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
artifact_dir="${ARTIFACT_DIR:-${repo_root}/artifacts/ulysses_a2a}"
log_path="${artifact_dir}/distributed_post_commit_failure_w2.log"
mkdir -p "${artifact_dir}"
cd "${repo_root}"
export PYTHONPATH="${repo_root}/python${PYTHONPATH:+:${PYTHONPATH}}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

set +e
timeout "${timeout_seconds}s" torchrun \
  --standalone \
  --nproc-per-node=2 \
  scripts/ulysses_a2a/distributed_post_commit_failure.py \
  2>&1 | tee "${log_path}"
status=${PIPESTATUS[0]}
set -e

if [[ "${status}" -eq 0 ]]; then
  echo "expected post-commit worker-group failure, got exit 0" >&2
  exit 1
fi
if [[ "${status}" -eq 124 ]]; then
  echo "post-commit worker-group failure hit the hard timeout" >&2
  exit 1
fi
if ! grep -q "injected_post_commit_signal_allocation_failure" "${log_path}"; then
  echo "injected failure marker missing from log" >&2
  exit 1
fi
if ! grep -q "UlyssesA2ACommitError" "${log_path}"; then
  echo "post-commit error was not classified as UlyssesA2ACommitError" >&2
  exit 1
fi
if ! grep -q "POST_COMMIT_RANK_FATAL rank=1 nccl_calls=0 p2p_calls=0" "${log_path}"; then
  echo "failing rank attempted an unexpected backend data path" >&2
  exit 1
fi

echo "DISTRIBUTED_POST_COMMIT_FAILURE_OK exit=${status} fallback=none watchdog=clean"
